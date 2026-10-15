"""Tools (and their JSON schemas) for the agent.

Read-only tools run automatically. Mutating tools (edit_file, write_file, run_shell,
web_fetch, MCP tools) ask for approval on every call unless the user opted a prefix
into the allow list. Every approved file change is checkpointed first so /undo can
restore it.
"""
from __future__ import annotations

import codecs
import difflib
import fnmatch
import html
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_OUTPUT_CHARS = 12000  # truncate tool output fed back to the model
SHELL_TIMEOUT = int(os.environ.get("AGENT_SHELL_TIMEOUT", "120"))  # default seconds per shell command
IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
                ".pytest_cache", ".ruff_cache", "dist", "build", ".idea"}
READ_ONLY_TOOLS = {"list_dir", "find_files", "read_file", "search_text", "todo_write"}
PARALLEL_TOOLS = {"list_dir", "find_files", "read_file", "search_text"}  # safe to run concurrently in one step
FILE_TOOLS = {"edit_file", "write_file", "apply_patch", "multi_edit", "insert_at_line"}
TODO_STATUSES = ("pending", "in_progress", "completed")
PLAN_MODE_MSG = ("Plan mode is active (read-only). Do not try to change anything; explore and present "
                 "a step-by-step plan as text. The user will approve it before any edits.")
READ_ONLY_MSG = "Error: this tool is not available to a sub-agent (read-only)."
SHELL_META = re.compile(r"[;&|<>`]|\$\(")

CONFIG_DIR = Path(os.environ.get("AGENT_HOME", str(Path.home() / ".anyagent")))
CONFIG_PATH = CONFIG_DIR / "config.json"
PROJECT_DIR = ".anyagent"   # per-project folder: config.json, HARNESS.md, memory.md, commands/
DEFAULT_DENY = ["rm -rf /", "rm -rf ~", "git push --force", "git push -f", "sudo ", "mkfs", "dd if="]


def _read_json(path) -> dict:
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load_config(root=None) -> dict:
    """Global config, with the project's .anyagent/config.json merged on top (lists are unioned)."""
    cfg = _read_json(CONFIG_PATH)
    if root:
        for k, v in _read_json(Path(root) / PROJECT_DIR / "config.json").items():
            if isinstance(v, list) and isinstance(cfg.get(k), list):
                cfg[k] = cfg[k] + [x for x in v if x not in cfg[k]]
            elif isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
    cfg.setdefault("deny", list(DEFAULT_DENY))
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n")


def unified(a: str, b: str, name_a: str, name_b: str) -> str:
    return "\n".join(difflib.unified_diff(a.splitlines(), b.splitlines(),
                                          fromfile=name_a, tofile=name_b, lineterm=""))


# ------------------------------------------------------------ text encoding
def decode_bytes(data: bytes):
    """Decode a file preserving its encoding. Returns (text, encoding, newline).

    `text` always uses "\\n"; `newline` is what the file used ("\\r\\n" or "\\n").
    """
    if data.startswith(codecs.BOM_UTF8):
        text, enc = data[len(codecs.BOM_UTF8):].decode("utf-8", errors="replace"), "utf-8-sig"
    elif data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text, enc = data.decode("utf-16", errors="replace"), "utf-16"
    else:
        for enc in ("utf-8", "cp1252", "latin-1"):  # latin-1 never fails
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
    newline = "\r\n" if "\r\n" in text else "\n"
    return text.replace("\r\n", "\n"), enc, newline


def encode_text(text: str, enc: str, newline: str) -> bytes:
    if newline != "\n":
        text = text.replace("\n", newline)
    return text.encode(enc, errors="replace")


def _leading_ws(s: str) -> str:
    return s[:len(s) - len(s.lstrip(" \t"))]


def fuzzy_find(text: str, old: str, new: str):
    """Whitespace-tolerant match for edit_file.

    Returns (start, end, adjusted_new, note) or an error string, or None when nothing
    is close. First tries ignoring trailing whitespace, then ignoring indentation
    (re-indenting `new` to match the file).
    """
    lines = old.split("\n")
    # 1. trailing whitespace only
    pat = "\n".join(re.escape(l.rstrip()) + "[ \t]*" for l in lines)
    hits = list(re.finditer(pat, text))
    if len(hits) == 1:
        m = hits[0]
        return m.start(), m.end(), new, "matched ignoring trailing whitespace"
    if len(hits) > 1:
        return f"Error: old_string matches {len(hits)} places (ignoring trailing whitespace). Add more context."
    # 2. indentation too
    pat = "\n".join("[ \t]*" + re.escape(l.strip()) + "[ \t]*" if l.strip() else "[ \t]*" for l in lines)
    hits = list(re.finditer(pat, text))
    if len(hits) > 1:
        return f"Error: old_string matches {len(hits)} places (ignoring indentation). Add more context."
    if not hits:
        return None
    m = hits[0]
    first_old = next((l for l in lines if l.strip()), "")
    first_hit = next((l for l in text[m.start():m.end()].split("\n") if l.strip()), "")
    old_indent, file_indent = _leading_ws(first_old), _leading_ws(first_hit)
    fixed = []
    for l in new.split("\n"):
        if l.startswith(old_indent):
            l = file_indent + l[len(old_indent):]
        fixed.append(l)
    return m.start(), m.end(), "\n".join(fixed), "matched ignoring indentation"


def closest_lines(text: str, old: str, n: int = 3) -> str:
    """Up to n file lines that look most like the first line of old_string."""
    probe = next((l.strip() for l in old.split("\n") if l.strip()), "")
    if not probe:
        return ""
    lines = text.split("\n")
    stripped = [l.strip() for l in lines]
    close = difflib.get_close_matches(probe, [s for s in stripped if s], n=n, cutoff=0.5)
    out = []
    for c in close:
        i = stripped.index(c)
        out.append(f"  {i + 1}: {lines[i][:160]}")
    return ("\nClosest lines in the file:\n" + "\n".join(out)) if out else ""


# ----------------------------------------------------------- shell helpers
def outside_paths(command: str, root: Path) -> list:
    """Paths mentioned in a command that lie outside root (heuristic, for warnings)."""
    root = Path(root).resolve()
    found = []
    command = os.path.expandvars(command.replace("$HOME", "~").replace("${HOME}", "~"))
    for tok in re.findall(r"(?<![\w.])(?:~|/|\.\./)[^\s;&|'\"`<>()]*", command):
        p = Path(os.path.expanduser(tok)) if tok.startswith("~") else (root / tok if tok.startswith("..") else Path(tok))
        try:
            rp = p.resolve()
        except (OSError, RuntimeError):
            rp = p
        if rp != root and root not in rp.parents:
            found.append(tok)
    if re.search(r"(?:^|[;&|]\s*)cd\s+(?:/|~|\.\.)", command.strip()):
        pass  # already captured by the token scan above
    return sorted(set(found))


def sandbox_argv(command: str, root: Path, platform: str = sys.platform) -> list:
    """Wrap a command so it can only write under root and temp dirs.

    macOS: sandbox-exec. Linux: bubblewrap (bwrap) with / read-only and root/tmp writable.
    """
    allow = [str(Path(root).resolve()), "/private/tmp", "/tmp", "/dev", "/var/folders", "/private/var/folders"]
    tmp = os.environ.get("TMPDIR")
    if tmp:
        allow += [tmp, str(Path(tmp).resolve())]
    allow = list(dict.fromkeys(allow))
    if platform.startswith("linux"):
        argv = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
        for p in allow:
            if p not in ("/dev", "/tmp", "/private/tmp", "/var/folders", "/private/var/folders") and os.path.isdir(p):
                argv += ["--bind", p, p]
        return argv + ["--die-with-parent", "--", "/bin/sh", "-c", command]
    if platform != "darwin":
        raise RuntimeError("--sandbox needs macOS (sandbox-exec) or Linux (bwrap)")
    rules = " ".join(f'(subpath "{p}")' for p in allow)
    profile = f"(version 1)(allow default)(deny file-write*)(allow file-write* {rules})"
    return ["sandbox-exec", "-p", profile, "/bin/sh", "-c", command]


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h\d>|</li>|</tr>", "\n", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(raw)
    text = "\n".join(re.sub(r"[ \t]+", " ", l).strip() for l in text.split("\n"))
    return re.sub(r"\n{2,}", "\n\n", text).strip()


def parse_ddg(page: str, n: int = 5) -> str:
    out = []
    for m in re.finditer(r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>.*?'
                         r'class="result__snippet"[^>]*>(.*?)</a>', page, re.S):
        href = html.unescape(m.group(1))
        q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg")
        url = q[0] if q else href
        out.append(f"{html_to_text(m.group(2))}\n{url}\n{html_to_text(m.group(3))}")
        if len(out) >= n:
            break
    return "\n\n".join(out)


class Tools:
    def __init__(self, root, approve=None, ui=None, allow=None, sandbox=False, shell_timeout=None, deny=None,
                 hooks=None):
        from anyagent.ui import BaseUI  # lazy: ui imports this module
        self.root = Path(root).resolve()
        self.ui = ui or BaseUI()
        self.approve = approve or self.ui.approve
        self.allow: list = list(allow or [])
        self.deny: list = list(DEFAULT_DENY if deny is None else deny)
        self.hooks: dict = dict(hooks or {})   # {"after_edit": "ruff check {path}", "after_shell": "..."}
        self.git = (self.root / ".git").exists()
        self.sandbox = sandbox
        self.shell_timeout = shell_timeout or SHELL_TIMEOUT
        self.todos: list = []
        self.checkpoints: list = []   # (relative path, previous bytes or None) per approved file change
        self.jobs: dict = {}          # background shell jobs
        self.mcp: dict = {}           # tool name -> (server, remote tool name, schema)
        self.plan_mode = False
        self.read_only = False        # set while a read-only sub-agent runs
        self.in_subagent = False      # set while any sub-agent runs (no nesting)
        self.subagent = None          # callable(task, quiet) -> str, installed by the session
        self.searcher = None          # callable(query, n) -> [(title, url, snippet)], e.g. Ollama web_search
        self._preapproved = False
        if hasattr(self.ui, "allow_cb"):
            self.ui.allow_cb = self.add_allow

    # ------------------------------------------------------------ helpers
    def _resolve(self, p: str) -> Path:
        path = Path(p).expanduser()
        path = (path if path.is_absolute() else self.root / path).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"Path {p!r} is outside the workspace root {self.root}")
        return path

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root)) or "."
        except ValueError:
            return str(path)

    @staticmethod
    def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
        if len(text) <= limit:
            return text
        head, tail = text[: limit * 2 // 3], text[-limit // 3:]
        return f"{head}\n... [truncated {len(text) - limit} chars] ...\n{tail}"

    @staticmethod
    def _is_binary(path: Path) -> bool:
        try:
            with open(path, "rb") as f:
                return b"\0" in f.read(2048)
        except OSError:
            return True

    def _read(self, p: Path):
        """(text with \\n newlines, encoding, original newline)."""
        return decode_bytes(p.read_bytes())

    def _walk_files(self, base: Path):
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
            for name in sorted(filenames):
                yield Path(dirpath) / name

    def _blocked(self):
        if self.read_only:
            return READ_ONLY_MSG
        if self.plan_mode:
            return PLAN_MODE_MSG
        return None

    # ----------------------------------------------------- approval / allow
    def add_allow(self, prefix: str) -> None:
        prefix = prefix.strip()
        if prefix and prefix not in self.allow:
            self.allow.append(prefix)
            cfg = load_config()
            cfg["allow"] = self.allow
            save_config(cfg)

    def remove_allow(self, prefix: str) -> bool:
        if prefix in self.allow:
            self.allow.remove(prefix)
            cfg = load_config()
            cfg["allow"] = self.allow
            save_config(cfg)
            return True
        return False

    def _denied(self, command: str):
        c = " ".join(command.split())
        for p in self.deny:
            if c == p.strip() or c.startswith(p) or f" {p}" in f" {c}":
                return f"Error: refused by the deny list (matches {p.strip()!r}). Ask the user to run it themselves."
        return None

    def _allowed(self, key: str, kind: str) -> bool:
        key = key.strip()
        if kind == "command" and SHELL_META.search(key):
            return False  # chained/redirected commands always ask
        return any(key == p or key.startswith(p + " ") or key.startswith(p) and kind != "command"
                   for p in self.allow)

    def _ask(self, title: str, body: str, kind: str, key: str = ""):
        if self._preapproved:
            return True, ""
        if key and self._allowed(key, kind):
            self.ui.info("  ⎿  auto-approved (allow list)")
            return True, ""
        return self.approve(title, body, kind=kind)

    def _checkpoint(self, p: Path) -> None:
        self.checkpoints.append((self._rel(p), p.read_bytes() if p.exists() else None))

    def _git(self, *args, timeout=30):
        r = subprocess.run(["git", *args], cwd=str(self.root), capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()

    def tree_checkpoint(self) -> str:
        """Snapshot the whole git working tree (tracked files) before a shell command. Returns the hash or ''."""
        if not self.git:
            return ""
        try:
            rc, out = self._git("stash", "create")
            if rc != 0:
                return ""
            if not out:
                rc, out = self._git("rev-parse", "HEAD")
            if rc == 0 and out:
                self.checkpoints.append(("", out.split()[0]))
                return out
        except (OSError, subprocess.TimeoutExpired):
            pass
        return ""

    def undo(self) -> str:
        """Restore the file state from before the most recent approved edit/write (or shell command, in git)."""
        if not self.checkpoints:
            return "Nothing to undo."
        rel, data = self.checkpoints.pop()
        if rel == "":  # whole-tree checkpoint taken before a shell command
            ok, why = self.approve("Restore tracked files from the checkpoint taken before the last command",
                                   f"git checkout {data} -- .   (new untracked files are kept)", kind="command")
            if not ok:
                self.checkpoints.append((rel, data))
                return "Undo cancelled."
            rc, out = self._git("checkout", data, "--", ".")
            return f"Restored tracked files to checkpoint {data[:10]}." if rc == 0 else f"Error: {out}"
        p = self.root / rel
        if data is None:
            if p.exists():
                p.unlink()
            return f"Removed {rel} (it was created by the agent)."
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return f"Restored {rel}."

    # ------------------------------------------- read-only tools (no approval)
    def list_dir(self, path: str = ".", max_depth: int = 2) -> str:
        base = self._resolve(path)
        if not base.is_dir():
            return f"Error: {path} is not a directory"
        max_depth = max(1, min(int(max_depth), 6))
        lines: list = []

        def rec(d: Path, depth: int):
            try:
                entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except OSError as e:
                lines.append(f"{'  ' * depth}[error: {e}]")
                return
            for e in entries:
                if e.name in IGNORED_DIRS:
                    continue
                if e.is_dir():
                    lines.append(f"{'  ' * depth}{e.name}/")
                    if depth + 1 < max_depth:
                        rec(e, depth + 1)
                else:
                    lines.append(f"{'  ' * depth}{e.name}")
                if len(lines) > 400:
                    return

        rec(base, 0)
        if len(lines) > 400:
            lines = lines[:400] + ["... [listing truncated]"]
        return "\n".join(lines) or "(empty directory)"

    def find_files(self, pattern: str) -> str:
        """Glob by filename pattern, e.g. '*.py' or 'tests/test_*.py'."""
        results = []
        for f in self._walk_files(self.root):
            rel = self._rel(f)
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(f.name, pattern):
                results.append(rel)
                if len(results) >= 200:
                    results.append("... [truncated]")
                    break
        return "\n".join(results) or "No files matched."

    def read_file(self, path: str, offset: int = 1, limit: int = 400) -> str:
        p = self._resolve(path)
        if not p.is_file():
            return f"Error: {path} is not a file"
        if self._is_binary(p):
            return f"Error: {path} looks like a binary file"
        offset, limit = max(1, int(offset)), max(1, min(int(limit), 2000))
        text, enc, nl = self._read(p)
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        chunk = lines[offset - 1: offset - 1 + limit]
        body = "\n".join(f"{i:>5}  {line}" for i, line in enumerate(chunk, start=offset))
        end = offset - 1 + len(chunk)
        note = "" if enc == "utf-8" and nl == "\n" else f" · {enc}{', CRLF' if nl != chr(10) else ''}"
        footer = f"\n[showing lines {offset}-{end} of {len(lines)}{note}]"
        return self._truncate(body) + footer

    def search_text(self, pattern: str, path: str = ".", glob: str = "") -> str:
        """Regex search across files; returns file:line:text."""
        base = self._resolve(path)
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"Error: invalid regex: {e}"
        files = [base] if base.is_file() else self._walk_files(base)
        hits: list = []
        for f in files:
            if glob and not (fnmatch.fnmatch(f.name, glob) or fnmatch.fnmatch(self._rel(f), glob)):
                continue
            if self._is_binary(f):
                continue
            try:
                for i, line in enumerate(self._read(f)[0].split("\n"), 1):
                    if rx.search(line):
                        hits.append(f"{self._rel(f)}:{i}: {line.strip()[:200]}")
                        if len(hits) >= 100:
                            hits.append("... [truncated at 100 matches]")
                            return "\n".join(hits)
            except OSError:
                continue
        return "\n".join(hits) or "No matches."

    def todo_write(self, todos) -> str:
        """Replace the task list shown to the user (planning aid; no side effects)."""
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except json.JSONDecodeError:
                return "Error: todos must be a list of {content, status}"
        if not isinstance(todos, list):
            return "Error: todos must be a list of {content, status}"
        clean = []
        for t in todos:
            if isinstance(t, str):
                t = {"content": t, "status": "pending"}
            if not isinstance(t, dict) or not str(t.get("content", "")).strip():
                return "Error: each todo needs a non-empty 'content'"
            status = t.get("status", "pending")
            clean.append({"content": str(t["content"]).strip(),
                          "status": status if status in TODO_STATUSES else "pending"})
        self.todos = clean
        self.ui.show_todos(clean)
        return "Todo list updated. Continue with the next pending item."

    # ------------------------------------ mutating tools (approval every time)
    def _plan_edit(self, path: str, old_string: str, new_string: str, replace_all: bool = False):
        """Compute an edit without applying it. Returns dict or an error string."""
        p = self._resolve(path)
        if not p.is_file():
            return f"Error: {path} does not exist (use write_file to create it)"
        if old_string == new_string:
            return "Error: old_string and new_string are identical"
        text, enc, nl = self._read(p)
        old, new = old_string.replace("\r\n", "\n"), new_string.replace("\r\n", "\n")
        count = text.count(old)
        note = ""
        if count == 0:
            found = fuzzy_find(text, old, new)
            if isinstance(found, str):
                return found
            if found is None:
                return ("Error: old_string not found. Re-read the file and copy the text exactly, "
                        "including whitespace." + closest_lines(text, old))
            start, end, new, note = found
            updated, count = text[:start] + new + text[end:], 1
        elif count > 1 and not replace_all:
            return (f"Error: old_string matches {count} places. Include more surrounding context "
                    "to make it unique, or set replace_all=true.")
        else:
            updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
            count = count if replace_all else 1
        return {"path": p, "updated": updated, "enc": enc, "nl": nl, "count": count, "note": note,
                "title": f"Edit {self._rel(p)}",
                "diff": unified(text, updated, f"a/{self._rel(p)}", f"b/{self._rel(p)}")}

    def _plan_write(self, path: str, content: str):
        p = self._resolve(path)
        existed = p.exists()
        if existed and not p.is_file():
            return f"Error: {path} is not a file"
        old, enc, nl = self._read(p) if existed else ("", "utf-8", "\n")
        return {"path": p, "updated": content.replace("\r\n", "\n"), "enc": enc, "nl": nl, "existed": existed,
                "title": f"{'Overwrite' if existed else 'Create'} {self._rel(p)}",
                "diff": unified(old, content, f"a/{self._rel(p)}" if existed else "/dev/null", f"b/{self._rel(p)}")}

    def preview(self, name: str, args: dict):
        """Dry-run a file tool call: (title, diff, path) or None (error / not a file tool)."""
        planners = {"edit_file": self._plan_edit, "write_file": self._plan_write, "multi_edit": self._plan_multi,
                    "apply_patch": self._plan_patch}
        if name not in planners:
            return None
        try:
            plan = planners[name](**args)
        except (TypeError, ValueError):
            return None
        return None if isinstance(plan, str) else (plan["title"], plan["diff"], plan["path"])

    def _apply(self, plan: dict) -> str:
        """Write the planned content. Returns hook output ('' when no hook is configured)."""
        self._checkpoint(plan["path"])
        plan["path"].parent.mkdir(parents=True, exist_ok=True)
        plan["path"].write_bytes(encode_text(plan["updated"], plan["enc"], plan["nl"]))
        return self._run_hook("after_edit", path=self._rel(plan["path"]))

    def _run_hook(self, name: str, **fmt) -> str:
        cmd = self.hooks.get(name)
        if not cmd:
            return ""
        try:
            cmd = cmd.format(**fmt)
            r = subprocess.run(cmd, shell=True, cwd=str(self.root), capture_output=True, text=True, timeout=120)
            out = (r.stdout + r.stderr).strip()
        except (OSError, subprocess.TimeoutExpired, KeyError) as e:
            return f"\n[hook {name} failed: {e}]"
        return f"\n[hook {name}: `{cmd}` exit {r.returncode}]\n{self._truncate(out, 2000)}" if (out or r.returncode) else ""

    # --- patch-style editing
    def _plan_multi(self, path: str, edits):
        """Sequential old/new replacements on one file, one approval."""
        if isinstance(edits, str):
            try:
                edits = json.loads(edits)
            except ValueError:
                return "Error: edits must be a list of {old_string, new_string}"
        if not isinstance(edits, list) or not edits:
            return "Error: edits must be a non-empty list of {old_string, new_string}"
        p = self._resolve(path)
        if not p.is_file():
            return f"Error: {path} does not exist"
        text, enc, nl = self._read(p)
        updated, notes = text, []
        for i, e in enumerate(edits, 1):
            if not isinstance(e, dict) or "old_string" not in e or "new_string" not in e:
                return f"Error: edit {i} needs old_string and new_string"
            old, new = str(e["old_string"]).replace("\r\n", "\n"), str(e["new_string"]).replace("\r\n", "\n")
            n = updated.count(old)
            if n == 1:
                updated = updated.replace(old, new, 1)
            elif n > 1 and e.get("replace_all"):
                updated = updated.replace(old, new)
            elif n > 1:
                return f"Error: edit {i}: old_string matches {n} places; add context or set replace_all"
            else:
                found = fuzzy_find(updated, old, new)
                if isinstance(found, str) or found is None:
                    return f"Error: edit {i}: old_string not found." + closest_lines(updated, old)
                a, b, new2, note = found
                updated = updated[:a] + new2 + updated[b:]
                notes.append(f"edit {i} {note}")
        return {"path": p, "updated": updated, "enc": enc, "nl": nl, "count": len(edits), "note": "; ".join(notes),
                "title": f"Edit {self._rel(p)} ({len(edits)} edits)",
                "diff": unified(text, updated, f"a/{self._rel(p)}", f"b/{self._rel(p)}")}

    def _plan_patch(self, patch: str, path: str = ""):
        """Apply a unified diff (one file) with fuzzy hunk placement."""
        lines = patch.replace("\r\n", "\n").split("\n")
        hunks, cur, target = [], None, path
        for ln in lines:
            if ln.startswith("+++ "):
                if not target:
                    target = re.sub(r"^[ab]/", "", ln[4:].strip().split("\t")[0])
                continue
            if ln.startswith("--- ") or ln.startswith("diff ") or ln.startswith("index "):
                continue
            if ln.startswith("@@"):
                m = re.match(r"@@ -(\d+)", ln)
                cur = {"start": int(m.group(1)) if m else 1, "old": [], "new": []}
                hunks.append(cur)
            elif cur is not None:
                if ln.startswith("+"):
                    cur["new"].append(ln[1:])
                elif ln.startswith("-"):
                    cur["old"].append(ln[1:])
                elif ln.startswith("\\"):
                    continue
                else:
                    body = ln[1:] if ln.startswith(" ") else ln
                    cur["old"].append(body)
                    cur["new"].append(body)
        if not target:
            return "Error: patch has no '+++ path' header; pass `path` explicitly"
        if not hunks:
            return "Error: no @@ hunks found in the patch"
        p = self._resolve(target)
        if not p.is_file():
            return f"Error: {target} does not exist"
        text, enc, nl = self._read(p)
        file_lines = text.split("\n")
        offset, notes = 0, []
        for hi, h in enumerate(hunks, 1):
            old = h["old"]
            while old and old[-1] == "" and len(old) > 1 and h["new"] and h["new"][-1] == "":
                old.pop(); h["new"].pop()  # trailing blank context from a trailing newline
            pos = self._find_block(file_lines, old, h["start"] - 1 + offset)
            if pos is None:
                return (f"Error: hunk {hi} (around line {h['start']}) does not match the file. "
                        "Re-read the file and regenerate the patch." + closest_lines(text, "\n".join(old)))
            file_lines[pos:pos + len(old)] = h["new"]
            offset += len(h["new"]) - len(old)
            if pos != h["start"] - 1:
                notes.append(f"hunk {hi} applied at line {pos + 1}")
        updated = "\n".join(file_lines)
        return {"path": p, "updated": updated, "enc": enc, "nl": nl, "count": len(hunks), "note": "; ".join(notes),
                "title": f"Patch {self._rel(p)} ({len(hunks)} hunks)",
                "diff": unified(text, updated, f"a/{self._rel(p)}", f"b/{self._rel(p)}")}

    @staticmethod
    def _find_block(lines, block, hint, radius=400):
        if not block:
            return max(0, min(hint, len(lines)))
        n = len(block)
        stripped = [l.rstrip() for l in block]
        for exact in (True, False):
            for delta in range(0, radius + 1):
                for pos in {hint - delta, hint + delta}:
                    if pos < 0 or pos + n > len(lines):
                        continue
                    window = lines[pos:pos + n]
                    if (window == block) if exact else ([l.rstrip() for l in window] == stripped):
                        return pos
        return None

    def multi_edit(self, path: str, edits) -> str:
        if self._blocked():
            return self._blocked()
        plan = self._plan_multi(path, edits)
        if isinstance(plan, str):
            return plan
        ok, why = self._ask(plan["title"], plan["diff"], "diff")
        if not ok:
            return why
        hook = self._apply(plan)
        note = f" ({plan['note']})" if plan["note"] else ""
        return f"Applied {plan['count']} edits to {self._rel(plan['path'])}{note}.{hook}"

    def apply_patch(self, patch: str, path: str = "") -> str:
        if self._blocked():
            return self._blocked()
        plan = self._plan_patch(patch, path)
        if isinstance(plan, str):
            return plan
        ok, why = self._ask(plan["title"], plan["diff"], "diff")
        if not ok:
            return why
        hook = self._apply(plan)
        note = f" ({plan['note']})" if plan["note"] else ""
        return f"Patched {self._rel(plan['path'])} ({plan['count']} hunks){note}.{hook}"

    def insert_at_line(self, path: str, line: int, text: str) -> str:
        """Insert text before 1-based `line` (line = 0 or past the end appends)."""
        if self._blocked():
            return self._blocked()
        p = self._resolve(path)
        if not p.is_file():
            return f"Error: {path} does not exist"
        content, enc, nl = self._read(p)
        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        at = len(lines) if int(line) <= 0 or int(line) > len(lines) else int(line) - 1
        new = text.replace("\r\n", "\n").rstrip("\n").split("\n")
        updated = "\n".join(lines[:at] + new + lines[at:]) + "\n"
        plan = {"path": p, "updated": updated, "enc": enc, "nl": nl,
                "title": f"Insert into {self._rel(p)} at line {at + 1}",
                "diff": unified(content, updated, f"a/{self._rel(p)}", f"b/{self._rel(p)}")}
        ok, why = self._ask(plan["title"], plan["diff"], "diff")
        if not ok:
            return why
        hook = self._apply(plan)
        return f"Inserted {len(new)} line(s) into {self._rel(p)} at line {at + 1}.{hook}"

    def edit_file(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        if self._blocked():
            return self._blocked()
        plan = self._plan_edit(path, old_string, new_string, replace_all)
        if isinstance(plan, str):
            return plan
        ok, why = self._ask(plan["title"], plan["diff"], "diff")
        if not ok:
            return why
        hook = self._apply(plan)
        note = f" ({plan['note']})" if plan["note"] else ""
        return f"Edited {self._rel(plan['path'])} ({plan['count']} replacement(s)){note}.{hook}"

    def write_file(self, path: str, content: str) -> str:
        if self._blocked():
            return self._blocked()
        plan = self._plan_write(path, content)
        if isinstance(plan, str):
            return plan
        ok, why = self._ask(plan["title"], plan["diff"], "diff")
        if not ok:
            return why
        hook = self._apply(plan)
        return f"{'Overwrote' if plan['existed'] else 'Created'} {self._rel(plan['path'])} ({len(content.splitlines())} lines).{hook}"

    @staticmethod
    def _kill(proc) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # whole process group (POSIX)
        except (ProcessLookupError, PermissionError, OSError, AttributeError):
            try:
                proc.kill()  # Windows / fallback: the direct child only
            except OSError:
                pass

    def _popen(self, command: str, stdout):
        env = dict(os.environ, PAGER="cat", GIT_PAGER="cat", NO_COLOR="1")
        if self.sandbox:
            args, shell = sandbox_argv(command, self.root), False
        else:
            args, shell = command, True
        return subprocess.Popen(args, shell=shell, cwd=str(self.root), env=env, stdin=subprocess.DEVNULL,
                                stdout=stdout, stderr=subprocess.STDOUT, text=True, errors="replace",
                                bufsize=1, start_new_session=(os.name != "nt"))

    def run_shell(self, command: str, timeout: int = 0, background: bool = False) -> str:
        if self._blocked():
            return self._blocked()
        if self._denied(command):
            return self._denied(command)
        ok, why = self._ask("Bash command", command, "command", key=command)
        if not ok:
            return why
        self.tree_checkpoint()
        if background:
            return self._start_job(command)
        timeout = int(timeout) if timeout and int(timeout) > 0 else self.shell_timeout
        lines: list = []
        proc = self._popen(command, subprocess.PIPE)

        def pump():
            for line in proc.stdout:  # stdout and stderr are merged, so order is preserved
                lines.append(line)

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        timed_out = False
        deadline = time.time() + timeout
        try:
            with self.ui.live_output(command) as live:
                while proc.poll() is None:
                    if time.time() > deadline:
                        timed_out = True
                        self._kill(proc)
                        break
                    live.update(lines)
                    time.sleep(0.1)
        except BaseException:  # includes Ctrl-C: don't leave the process running
            self._kill(proc)
            raise
        t.join(1)
        body = "".join(lines)
        if timed_out:
            return self._truncate(f"Error: command timed out after {timeout}s and was killed. Pass a larger "
                                  f"`timeout` or run it with background=true.\n{body}")
        return self._truncate(f"[exit code {proc.returncode}]\n{body}".rstrip()) + self._run_hook("after_shell", command=command)

    # ------------------------------------------------------ background jobs
    def _start_job(self, command: str) -> str:
        jid = len(self.jobs) + 1
        log_dir = CONFIG_DIR / "jobs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"job-{os.getpid()}-{jid}.log"
        fh = open(log, "w", encoding="utf-8", errors="replace")
        proc = self._popen(command, fh)
        self.jobs[jid] = {"proc": proc, "log": log, "command": command, "pos": 0, "fh": fh}
        return (f"Started background job {jid} (pid {proc.pid}). Output is logged to {log}. "
                f"Call check_job(job_id={jid}) to read new output or kill it.")

    def check_job(self, job_id: int, kill: bool = False) -> str:
        job = self.jobs.get(int(job_id))
        if not job:
            return f"Error: no job {job_id}. Known jobs: {sorted(self.jobs) or 'none'}"
        proc = job["proc"]
        if kill and proc.poll() is None:
            self._kill(proc)
            proc.wait(5)
        rc = proc.poll()
        job["fh"].flush()
        with open(job["log"], "r", encoding="utf-8", errors="replace") as f:
            f.seek(job["pos"])
            new = f.read()
            job["pos"] = f.tell()
        status = "running" if rc is None else f"exited with code {rc}" + (" (killed)" if kill else "")
        return self._truncate(f"[job {job_id}: {status}]\n{new}".rstrip() or f"[job {job_id}: {status}]")

    def list_jobs(self) -> list:
        return [(jid, "running" if j["proc"].poll() is None else f"exit {j['proc'].returncode}", j["command"])
                for jid, j in self.jobs.items()]

    def kill_jobs(self) -> None:
        for j in self.jobs.values():
            if j["proc"].poll() is None:
                self._kill(j["proc"])
            j["fh"].close()

    # ----------------------------------------------------------- web fetch
    def web_fetch(self, url: str, max_chars: int = 8000) -> str:
        if self._blocked():
            return self._blocked()
        if not re.match(r"https?://", url):
            return "Error: only http(s) URLs are supported"
        ok, why = self._ask("Fetch URL", url, "text", key=f"web_fetch {url}")
        if not ok:
            return why
        req = urllib.request.Request(url, headers={"User-Agent": "anyagent/0.2 (+local coding agent)"})
        try:
            with self.ui.working(f"Fetching {url}…"):
                with urllib.request.urlopen(req, timeout=30) as resp:
                    ctype = resp.headers.get("Content-Type", "")
                    raw = resp.read(2_000_000)
        except Exception as e:
            return f"Error: fetch failed: {e}"
        text = decode_bytes(raw)[0]
        if "html" in ctype or re.search(r"(?i)<html|<body", text[:2000]):
            text = html_to_text(text)
        max_chars = max(500, min(int(max_chars), 50000))
        return self._truncate(text, max_chars) or "(empty response)"

    def web_search(self, query: str, max_results: int = 5) -> str:
        """DuckDuckGo HTML results (no key needed); Ollama's hosted search when OLLAMA_API_KEY is set."""
        if self._blocked():
            return self._blocked()
        ok, why = self._ask("Web search", query, "text", key=f"web_search {query}")
        if not ok:
            return why
        max_results = max(1, min(int(max_results), 10))
        if self.searcher:
            try:
                hits = self.searcher(query, max_results)
                return "\n\n".join(f"{t}\n{u}\n{s}" for t, u, s in hits) or "No results."
            except Exception as e:  # fall through to DuckDuckGo
                self.ui.warn(f"hosted search failed ({e}); using DuckDuckGo")
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (anyagent)"})
        try:
            with self.ui.working("Searching…"):
                with urllib.request.urlopen(req, timeout=30) as resp:
                    page = resp.read(2_000_000).decode("utf-8", "replace")
        except Exception as e:
            return f"Error: search failed: {e}"
        return self._truncate(parse_ddg(page, max_results)) or "No results."

    # ------------------------------------------------------------ sub-agent
    def run_subagent(self, task: str = "", tasks=None, write: bool = False) -> str:
        if self.read_only or self.in_subagent:
            return "Error: a sub-agent cannot spawn another sub-agent."
        if not self.subagent:
            return "Error: sub-agents are not available in this session."
        jobs = [t for t in ([task] + list(tasks or [])) if str(t).strip()]
        if not jobs:
            return "Error: give a task (or a list of tasks)"
        if write and self.plan_mode:
            return PLAN_MODE_MSG
        if write or len(jobs) == 1:
            out = []
            for t in jobs:
                self.read_only = not write
                try:
                    out.append(self.subagent(t, quiet=False))
                finally:
                    self.read_only = False
            return "\n\n".join(out)
        self.read_only = True
        try:
            with self.ui.working(f"Running {len(jobs)} sub-agents…"):
                with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as ex:
                    results = list(ex.map(lambda t: self.subagent(t, quiet=True), jobs))
        finally:
            self.read_only = False
        return "\n\n".join(f"## Task: {t}\n{r}" for t, r in zip(jobs, results))

    # ------------------------------------------------------------------ MCP
    def register_mcp(self, server) -> list:
        """Expose an MCP server's tools as mcp__<server>__<tool>. Returns the tool names."""
        names = []
        for t in server.tools:
            name = f"mcp__{server.name}__{t['name']}"
            schema = t.get("inputSchema") or {}
            self.mcp[name] = (server, t["name"], _tool(name, f"[MCP {server.name}] {t.get('description', '')}",
                                                       schema.get("properties", {}), schema.get("required", [])))
            names.append(name)
        if getattr(server, "resources", None):
            name = f"mcp__{server.name}__read_resource"
            listing = ", ".join(r.get("uri", "") for r in server.resources[:20])
            self.mcp[name] = (server, "__resource__", _tool(
                name, f"[MCP {server.name}] Read a resource by URI. Available: {listing}",
                {"uri": {"type": "string"}}, ["uri"]))
            names.append(name)
        return names

    def _call_mcp(self, name: str, args: dict) -> str:
        if self._blocked():
            return self._blocked()
        server, remote, _ = self.mcp[name]
        label = "read_resource" if remote == "__resource__" else remote
        ok, why = self._ask(f"MCP {server.name}: {label}", json.dumps(args, indent=2), "text", key=name)
        if not ok:
            return why
        with self.ui.working(f"MCP {server.name}…"):
            if remote == "__resource__":
                return self._truncate(server.read_resource(str(args.get("uri", ""))))
            return self._truncate(server.call_tool(remote, args))

    def schemas(self, read_only: bool = False) -> list:
        if read_only:
            return [t for t in TOOLS if t["function"]["name"] in READ_ONLY_TOOLS]
        return TOOLS + [s for _, _, s in self.mcp.values()]

    # ------------------------------------------------------------ dispatch
    def _invoke(self, name: str, args) -> str:
        if name in self.mcp:
            return self._call_mcp(name, args)
        fn = getattr(self, name, None)
        if name not in TOOL_NAMES or fn is None:
            return f"Error: unknown tool {name!r}. Available: {', '.join(sorted(TOOL_NAMES | set(self.mcp)))}"
        if self.read_only and name not in READ_ONLY_TOOLS:
            return READ_ONLY_MSG
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return "Error: tool arguments were not valid JSON"
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:  # report back to the model so it can recover
            return f"Error: {type(e).__name__}: {e}"

    def dispatch(self, name: str, args, preapproved: bool = False) -> str:
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        args = args if isinstance(args, dict) else {}
        self.ui.tool_call(name, args)
        self._preapproved = preapproved
        try:
            result = self._invoke(name, args)
        finally:
            self._preapproved = False
        self.ui.tool_result(name, args, result)
        return result

    def dispatch_parallel(self, calls: list) -> list:
        """Run read-only calls concurrently (results printed in order). calls: [(name, args)]."""
        with ThreadPoolExecutor(max_workers=min(8, len(calls))) as ex:
            results = list(ex.map(lambda c: self._invoke(c[0], c[1]), calls))
        for (name, args), r in zip(calls, results):
            self.ui.tool_call(name, args)
            self.ui.tool_result(name, args, r)
        return results

    def batch_approve(self, calls: list):
        """Offer one approval for several file edits in a single model step.

        calls: [(name, args)]. Returns {index: True} for pre-approved calls, {index: "denial"} for
        rejected ones, or {} to fall back to one-by-one approval.
        """
        if self._blocked():
            return {}
        items = []
        for i, (name, args) in enumerate(calls):
            if name in FILE_TOOLS:
                pv = self.preview(name, args)
                if pv is None:
                    return {}
                items.append((i, pv))
        if len(items) < 2 or len({pv[2] for _, pv in items}) != len(items):
            return {}  # two edits to the same file depend on each other: review one by one
        choice = self.ui.approve_batch([(t, d) for _, (t, d, _) in items])
        if choice == "all":
            return {i: True for i, _ in items}
        if choice == "none":
            return {i: "User denied this action (rejected the whole batch). Ask what to do instead."
                    for i, _ in items}
        return {}


def _tool(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOLS = [
    _tool("list_dir", "List files and folders (tree) under a directory.",
          {"path": {"type": "string", "description": "Directory, relative to root. Default '.'"},
           "max_depth": {"type": "integer", "description": "Depth 1-6. Default 2"}}, []),
    _tool("find_files", "Find files by glob pattern on name or relative path, e.g. '*.py' or 'tests/*.py'.",
          {"pattern": {"type": "string"}}, ["pattern"]),
    _tool("read_file", "Read a text file with line numbers. Use offset/limit for large files.",
          {"path": {"type": "string"}, "offset": {"type": "integer", "description": "1-based start line"},
           "limit": {"type": "integer", "description": "Number of lines (max 2000)"}}, ["path"]),
    _tool("search_text", "Regex search across files (like grep -rn). Returns file:line: text.",
          {"pattern": {"type": "string", "description": "Python regex"},
           "path": {"type": "string", "description": "File or directory. Default '.'"},
           "glob": {"type": "string", "description": "Optional filename filter, e.g. '*.py'"}}, ["pattern"]),
    _tool("todo_write", "Create or update a short task list for multi-step work. Send the FULL list each time, "
                        "with exactly one item in_progress. Mark items completed as you finish them.",
          {"todos": {"type": "array", "items": {"type": "object", "properties": {
              "content": {"type": "string"},
              "status": {"type": "string", "enum": list(TODO_STATUSES)}},
              "required": ["content", "status"]}}}, ["todos"]),
    _tool("edit_file", "Replace exact text in an existing file. old_string must match exactly once unless "
                       "replace_all. Requires user approval.",
          {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"},
           "replace_all": {"type": "boolean"}}, ["path", "old_string", "new_string"]),
    _tool("write_file", "Create a new file or fully overwrite one. Prefer edit_file for existing files. "
                        "Requires user approval.",
          {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _tool("run_shell", "Run a shell command in the project root (tests, linters, git, installs). Requires user "
                       "approval. Output streams live; no stdin. Default timeout 120s: set `timeout` for slow "
                       "commands, or background=true for servers/watchers (then poll with check_job).",
          {"command": {"type": "string"},
           "timeout": {"type": "integer", "description": "Seconds before the command is killed"},
           "background": {"type": "boolean", "description": "Start and return immediately with a job id"}},
          ["command"]),
    _tool("check_job", "Read new output from a background job started by run_shell, optionally killing it.",
          {"job_id": {"type": "integer"}, "kill": {"type": "boolean"}}, ["job_id"]),
    _tool("web_fetch", "Fetch a URL (docs, issues, APIs) and return its text. Requires user approval.",
          {"url": {"type": "string"}, "max_chars": {"type": "integer", "description": "Default 8000"}}, ["url"]),
    _tool("run_subagent", "Delegate work to a sub-agent with a fresh context and get its report back. Default is "
                          "read-only research (explore code, find usages, summarize a module); pass several "
                          "`tasks` to run them in parallel. write=true lets one sub-agent edit files (each edit "
                          "still asks the user).",
          {"task": {"type": "string", "description": "Self-contained description of the job"},
           "tasks": {"type": "array", "items": {"type": "string"}, "description": "Several independent tasks"},
           "write": {"type": "boolean", "description": "Allow the sub-agent to edit files / run commands"}}, []),
    _tool("multi_edit", "Apply several old_string/new_string replacements to ONE file in order, with a single "
                        "approval. Each old_string must match once (whitespace-tolerant).",
          {"path": {"type": "string"}, "edits": {"type": "array", "items": {"type": "object", "properties": {
              "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}},
              "required": ["old_string", "new_string"]}}}, ["path", "edits"]),
    _tool("apply_patch", "Apply a unified diff (one file, '@@' hunks with context lines). Hunks are placed "
                         "fuzzily near their line numbers. Requires user approval.",
          {"patch": {"type": "string"}, "path": {"type": "string", "description": "Target file if the patch has no +++ header"}},
          ["patch"]),
    _tool("insert_at_line", "Insert text before the given 1-based line (0 or past the end appends). "
                            "Requires user approval.",
          {"path": {"type": "string"}, "line": {"type": "integer"}, "text": {"type": "string"}}, ["path", "line", "text"]),
    _tool("web_search", "Search the web (DuckDuckGo). Returns titles, URLs and snippets; follow up with web_fetch. "
                        "Requires user approval.",
          {"query": {"type": "string"}, "max_results": {"type": "integer", "description": "1-10, default 5"}}, ["query"]),
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
