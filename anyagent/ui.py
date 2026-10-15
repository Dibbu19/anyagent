"""Claude-Code-style terminal UI (rich + prompt_toolkit).

BaseUI is a silent no-op implementation (used in tests / headless runs).
UI is the interactive one: streamed markdown, tool bullets, diff panels,
an arrow-key approval menu, slash-command and @file completion.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path

from prompt_toolkit.completion import Completer, Completion
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from anyagent.tools import IGNORED_DIRS, SHELL_META, outside_paths

ACCENT = "#d97757"
ADD_STYLE = "on #1f3d2a"
DEL_STYLE = "on #4a2025"
APP_NAME = "AnyAgent"


# --------------------------------------------------------------------- helpers
def fmt_tokens(n) -> str:
    n = int(n or 0)
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _one_line(s: str, limit: int = 100) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def tool_label(name: str, args: dict):
    """(display name, argument summary) in the style of Claude Code."""
    a = args or {}
    if name == "list_dir":
        return "List", _one_line(a.get("path", "."))
    if name == "find_files":
        return "Glob", _one_line(a.get("pattern", ""))
    if name == "read_file":
        return "Read", _one_line(a.get("path", ""))
    if name == "search_text":
        where = a.get("path")
        extra = f" in {where}" if where and where != "." else ""
        return "Search", _one_line(f'"{a.get("pattern", "")}"{extra}')
    if name == "edit_file":
        return "Update", _one_line(a.get("path", ""))
    if name == "write_file":
        return "Write", _one_line(a.get("path", ""))
    if name == "run_shell":
        return "Bash", _one_line(a.get("command", ""))
    if name == "check_job":
        return "Job", _one_line(f"#{a.get('job_id', '?')}" + (" kill" if a.get("kill") else ""))
    if name == "web_fetch":
        return "Fetch", _one_line(a.get("url", ""))
    if name == "run_subagent":
        tasks = a.get("tasks") or []
        return "Agent", _one_line(a.get("task") or f"{len(tasks)} tasks: " + "; ".join(map(str, tasks)))
    if name == "multi_edit":
        return "Update", _one_line(f"{a.get('path', '')} ({len(a.get('edits') or [])} edits)")
    if name == "apply_patch":
        return "Patch", _one_line(a.get("path") or re.sub(r"^.*?\+\+\+ (?:[ab]/)?(\S+).*$", r"\1", str(a.get("patch", "")), flags=re.S)[:80])
    if name == "insert_at_line":
        return "Insert", _one_line(f"{a.get('path', '')}:{a.get('line', '')}")
    if name == "web_search":
        return "Search web", _one_line(a.get("query", ""))
    if name == "todo_write":
        return "Update Todos", ""
    if name.startswith("mcp__"):
        _, server, tool = name.split("__", 2)
        return f"MCP {server}:{tool}", _one_line(", ".join(f"{k}={v}" for k, v in a.items()))
    return name, _one_line(", ".join(f"{k}={v}" for k, v in a.items()))


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def summarize_result(name: str, result: str):
    """(short text, rich style, extra lines) shown under a tool call."""
    r = (result or "").strip("\n")
    lines = r.splitlines()
    first = lines[0] if lines else ""
    if r.startswith("User denied"):
        fb = r.split("feedback:", 1)[1].strip() if "feedback:" in r else ""
        return ("Rejected" + (f" — {fb}" if fb else ""), "red", [])
    if r.startswith("Error"):
        return (first[:200], "red", [])
    if name == "read_file":
        m = re.search(r"\[showing lines (\d+)-(\d+) of (\d+)([^\]]*)\]\s*$", r)
        if m:
            a, b, total = (int(x) for x in m.groups()[:3])
            count = max(0, b - a + 1)
            return (f"Read {_plural(count, 'line')}" + (f" (of {total})" if count != total else "") + m.group(4),
                    "dim", [])
    if name == "list_dir":
        return (f"Listed {_plural(len([l for l in lines if l.strip()]), 'entry').replace('entrys', 'entries')}", "dim", [])
    if name == "find_files":
        if r.startswith("No files"):
            return ("No files found", "dim", [])
        return (f"Found {_plural(len([l for l in lines if not l.startswith('...')]), 'file')}", "dim", [])
    if name == "search_text":
        if r.startswith("No matches"):
            return ("No matches", "dim", [])
        return (f"Found {_plural(len([l for l in lines if not l.startswith('...')]), 'match').replace('matchs', 'matches')}", "dim", [])
    if name in ("run_shell", "check_job"):
        code = re.match(r"\[(?:exit code (-?\d+)|job \d+: ([^\]]+))\]", first)
        body = [l for l in (lines[1:] if code else lines)]
        extra = body[:12]
        if len(body) > 12:
            extra.append(f"… +{len(body) - 12} lines")
        if code and code.group(2):
            return (code.group(2), "green" if "running" in code.group(2) or "code 0" in code.group(2) else "red",
                    extra or ["(no new output)"])
        if code:
            rc = int(code.group(1))
            head = "exit code 0" if rc == 0 else f"exit code {rc}"
            return (head, "green" if rc == 0 else "red", extra or ["(no output)"])
        return ("done", "dim", extra)
    if name == "web_fetch":
        return (f"Fetched {_plural(len(r), 'char')}", "dim", [])
    if name == "run_subagent":
        return (f"Sub-agent finished ({_plural(len(lines), 'line')})", "dim", [])
    if name == "web_search":
        return (f"Found {_plural(len([l for l in lines if l.startswith('http')]), 'result')}", "dim", [])
    return (first[:200], "dim", [])


def render_diff(diff_text: str, max_lines=60):
    """Unified diff -> rich renderable with line numbers and red/green rows."""
    lines = diff_text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("--- ") and lines[1].startswith("+++ "):
        lines = lines[2:]
    if not lines:
        return Text("(no changes)", style="dim")
    added = sum(1 for l in lines if l.startswith("+"))
    removed = sum(1 for l in lines if l.startswith("-"))
    body = Text()
    old_n = new_n = 0
    shown = 0
    for idx, line in enumerate(lines):
        if max_lines and shown >= max_lines:
            body.append(f"… +{len(lines) - idx} more lines (choose \"View full diff\" to see everything)\n", style="dim")
            break
        if line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", line)
            if m:
                old_n, new_n = int(m.group(1)), int(m.group(2))
            if shown:
                body.append("      ⋮\n", style="dim")
            continue
        if line.startswith("+"):
            body.append(f"{new_n:>5} + {line[1:]}\n", style=ADD_STYLE)
            new_n += 1
        elif line.startswith("-"):
            body.append(f"{old_n:>5} - {line[1:]}\n", style=DEL_STYLE)
            old_n += 1
        elif line.startswith("\\"):
            body.append(f"      {line}\n", style="dim")
        else:
            body.append(f"{new_n:>5}   {line[1:] if line.startswith(' ') else line}\n")
            old_n += 1
            new_n += 1
        shown += 1
    body.rstrip()
    summary = Text(f"Added {_plural(added, 'line')}, removed {_plural(removed, 'line')}", style="dim")
    return Group(summary, body)


def split_committed(buf: str):
    """Split streamed markdown into (complete blocks, still-growing tail).

    A block is complete once a blank line follows it outside a code fence.
    """
    in_fence = False
    pos = 0
    cut = 0
    for line in buf.splitlines(keepends=True):
        stripped = line.strip()
        pos += len(line)
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if not in_fence and stripped == "" and line.endswith("\n") and buf[:pos].strip():
            cut = pos
    return buf[:cut], buf[cut:]


def allow_prefix(command: str) -> str:
    """Suggested allow-list prefix for a command: first two words, e.g. 'git status'."""
    words = command.strip().split()
    return " ".join(words[:2]) if len(words) >= 2 and not words[1].startswith("-") else (words[0] if words else "")


class StreamResult:
    def __init__(self):
        self.content = ""
        self.thinking = ""
        self.calls: list = []
        self.last = None
        self.t0 = time.time()


class ThinkingStream:
    """Dim, scrolling tail of the model's reasoning while it streams."""

    def __init__(self, tail: int = 6):
        self.spinner = Thinking("Thinking…")
        self.text = ""
        self.tail = tail

    def add(self, piece: str):
        self.text += piece

    def __rich__(self):
        lines = self.text.strip().splitlines()[-self.tail:]
        return Group(self.spinner, Text("\n".join(lines), style="dim italic", overflow="ellipsis")) if lines else self.spinner


class Thinking:
    """Animated '✻ Thinking… (12s · ctrl+c to interrupt)' line."""

    def __init__(self, label: str = "Thinking…"):
        self.label = label
        self.t0 = time.time()
        self.spinner = Spinner("dots", style=ACCENT)

    def __rich__(self):
        secs = int(time.time() - self.t0)
        self.spinner.text = Text.assemble((" " + self.label, ACCENT), (f" ({secs}s · ctrl+c to interrupt)", "dim"))
        return self.spinner


class LiveOutput:
    """Spinner plus the last few lines of a running command."""

    def __init__(self, command: str, console, tail: int = 8):
        self.thinking = Thinking(_one_line(command, 60))
        self.tail = tail
        self.lines: list = []
        self.live = Live(self, console=console, refresh_per_second=8, transient=True)

    def update(self, lines):
        self.lines = lines[-self.tail:]
        self.live.refresh()

    def __rich__(self):
        body = Text("".join(self.lines).rstrip("\n"), style="dim", no_wrap=True, overflow="ellipsis")
        return Group(self.thinking, body) if self.lines else self.thinking

    def __enter__(self):
        self.live.start()
        return self

    def __exit__(self, *exc):
        self.live.stop()


# ----------------------------------------------------------------- select menu
def select(options, title="Do you want to proceed?", input=None, output=None):
    """Inline arrow-key menu. Returns the chosen index, or None if cancelled (Esc).

    Raises KeyboardInterrupt on Ctrl-C.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    idx = [0]
    kb = KeyBindings()

    def render():
        frags = [("bold", f" {title}\n")]
        for i, opt in enumerate(options):
            if i == idx[0]:
                frags.append((f"fg:{ACCENT} bold", f" ❯ {i + 1}. {opt}\n"))
            else:
                frags.append(("", f"   {i + 1}. {opt}\n"))
        frags.append(("fg:#808080", " ↑/↓ move · enter select · y/n quick answer · esc cancel"))
        return frags

    @kb.add("up")
    @kb.add("k")
    def _up(event):
        idx[0] = (idx[0] - 1) % len(options)

    @kb.add("down")
    @kb.add("j")
    def _down(event):
        idx[0] = (idx[0] + 1) % len(options)

    @kb.add("enter")
    def _enter(event):
        event.app.exit(result=idx[0])

    @kb.add("escape", eager=True)
    def _esc(event):
        event.app.exit(result=None)

    @kb.add("c-c")
    def _ctrl_c(event):
        event.app.exit(exception=KeyboardInterrupt)

    for n in range(min(len(options), 9)):
        kb.add(str(n + 1))(lambda event, n=n: event.app.exit(result=n))
    kb.add("y")(lambda event: event.app.exit(result=0))
    kb.add("n")(lambda event: event.app.exit(result=1))

    app = Application(
        layout=Layout(Window(FormattedTextControl(render, focusable=True, show_cursor=False))),
        key_bindings=kb, full_screen=False, erase_when_done=True, input=input, output=output)
    return app.run()


# ------------------------------------------------------------------ completion
class InputCompleter(Completer):
    def __init__(self, root: Path, commands, dynamic=None):
        self.root = Path(root)
        self.commands = commands
        self.dynamic = dynamic or {}   # "/model" -> callable() -> [candidates]
        self._cache = ([], 0.0)

    def _files(self):
        now = time.time()
        if now - self._cache[1] > 5:
            files = []
            for dp, dn, fn in os.walk(self.root):
                dn[:] = sorted(d for d in dn if d not in IGNORED_DIRS)
                for f in sorted(fn):
                    files.append(os.path.relpath(os.path.join(dp, f), self.root))
                if len(files) > 5000:
                    break
            self._cache = (files, now)
        return self._cache[0]

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/") and " " not in text:
            for name, desc in self.commands:
                if name.startswith(text):
                    yield Completion(name, start_position=-len(text), display_meta=desc)
            return
        if text.startswith("/") and " " in text:
            cmd, _, frag = text.partition(" ")
            fn = self.dynamic.get(cmd)
            if fn:
                try:
                    cands = fn()
                except Exception:
                    cands = []
                for c in cands:
                    if c.lower().startswith(frag.lower()):
                        yield Completion(c, start_position=-len(frag))
            return
        m = re.search(r"(?:^|\s)@([^\s@]*)$", text)
        if m:
            frag = m.group(1).lower()
            files = self._files()
            starts = [f for f in files if f.lower().startswith(frag)]
            contains = [f for f in files if frag in f.lower() and not f.lower().startswith(frag)]
            for f in (starts + contains)[:20]:
                yield Completion("@" + f, start_position=-len(m.group(1)) - 1)


class _NullLive:
    def update(self, lines): pass
    def __enter__(self): return self
    def __exit__(self, *exc): pass


# ----------------------------------------------------------------------- UIs
class BaseUI:
    """Silent UI: all output is dropped, approvals are denied."""

    commands: list = []
    status: dict = {}
    allow_cb = None

    def banner(self, *a, **k): pass
    def info(self, msg): pass
    def warn(self, msg): pass
    def error(self, msg): pass
    def show_help(self, commands): pass
    def tool_call(self, name, args): pass
    def tool_result(self, name, args, result): pass
    def show_todos(self, todos): pass
    def read_input(self): return None
    def confirm_plan(self): return False

    def approve(self, title, body, kind="text"):
        return False, "User denied this action (no interactive UI)."

    def approve_batch(self, items):
        return "each"

    def working(self, message="Working…"):
        return contextlib.nullcontext()

    def live_output(self, command):
        return _NullLive()

    def pick(self, options, title):
        return None

    def consume_stream(self, stream, result):
        for chunk in stream:
            result.last = chunk
            m = chunk.message
            if m.content:
                result.content += m.content
            if getattr(m, "thinking", None):
                result.thinking += m.thinking
            if m.tool_calls:
                result.calls.extend(m.tool_calls)


class HeadlessUI(BaseUI):
    """For --json / scripts: approves everything, prints only warnings/errors to stderr."""

    def __init__(self):
        self.log: list = []

    def warn(self, msg):
        print(str(msg), file=sys.stderr)

    def error(self, msg):
        print(str(msg), file=sys.stderr)

    def tool_call(self, name, args):
        self.log.append({"tool": name, "args": args})

    def tool_result(self, name, args, result):
        self.log[-1]["result"] = result[:2000] if isinstance(result, str) else result

    def approve(self, title, body, kind="text"):
        return True, ""

    def approve_batch(self, items):
        return "all"


class UI(BaseUI):
    def __init__(self, root=".", console=None, pt_input=None, pt_output=None, history_path=None):
        self.root = Path(root).resolve()
        self.console = console or Console(highlight=False)
        self.commands = []
        self.status = {}
        self.dynamic = {}
        self.select_fn = self._select
        self.allow_cb = None
        self._first = True
        self._armed = False
        self._pt_input = pt_input
        self._pt_output = pt_output
        self._history_path = history_path
        self._session = None

    # ----------------------------------------------------------- messages
    def banner(self, model, root, num_ctx, provider="ollama", resumed=False, sandbox=False):
        notes = ["Every edit and shell command asks for your approval."]
        if sandbox:
            notes.append("Shell commands run under sandbox-exec (writes limited to the project and temp dirs).")
        if resumed:
            notes.append("Resumed the previous session for this directory.")
        body = Group(
            Text.assemble(("✻ ", ACCENT), (f"Welcome to {APP_NAME}", "bold")),
            Text(""),
            Text("/help for commands · @file to attach · \\⏎ for a newline", style="dim"),
            Text(""),
            Text(f"cwd:   {root}", style="dim"),
            Text(f"model: {model} ({provider}) · context {fmt_tokens(num_ctx)}", style="dim"),
            *[Text(n, style="dim") for n in notes],
        )
        self.console.print(Panel(body, border_style=ACCENT, expand=False, padding=(0, 2)))

    def info(self, msg):
        self.console.print(Text(str(msg), style="dim"))

    def warn(self, msg):
        self.console.print(Text(str(msg), style="yellow"))

    def error(self, msg):
        self.console.print(Text(str(msg), style="red"))

    def show_help(self, commands):
        table = Table.grid(padding=(0, 3))
        for name, desc in commands:
            table.add_row(Text(name, style=ACCENT), Text(desc))
        keys = Table.grid(padding=(0, 3))
        for k, d in [("Enter", "send message"), ("\\ + Enter / Esc Enter / Ctrl-J", "newline"),
                     ("@path", "attach a file (Tab completes)"), ("Ctrl-C", "interrupt a running turn"),
                     ("Ctrl-C twice / Ctrl-D", "exit")]:
            keys.add_row(Text(k, style=ACCENT), Text(d))
        self.console.print(Panel(Group(Text("Commands", style="bold"), table, Text(""),
                                       Text("Keys", style="bold"), keys),
                                 border_style="dim", expand=False, padding=(0, 2)))

    # ------------------------------------------------------ assistant text
    def _bulleted(self, renderable):
        grid = Table.grid(padding=(0, 1), expand=True)
        grid.add_column(width=1, no_wrap=True)
        grid.add_column(ratio=1)
        grid.add_row(Text("●" if self._first else " ", style=ACCENT), renderable)
        return grid

    def _print_block(self, text):
        self.console.print(self._bulleted(Markdown(text.strip("\n"), code_theme="monokai")))
        self._first = False

    def _pending(self, buf):
        if not buf.strip():
            return Text("")
        lines = buf.splitlines()
        room = max(3, self.console.size.height - 6)
        if len(lines) > room:  # too tall for a live region: show the tail as plain text
            return self._bulleted(Text("\n".join(lines[-room:]), style="dim"))
        return self._bulleted(Markdown(buf.strip("\n"), code_theme="monokai"))

    def consume_stream(self, stream, result):
        self._first = True
        reasoning = ThinkingStream()
        thinking = Live(reasoning, console=self.console, refresh_per_second=10, transient=True)
        live = Live(Text(""), console=self.console, refresh_per_second=12, transient=True)
        thinking.start()
        state = "thinking"
        buf = ""
        try:
            for chunk in stream:
                result.last = chunk
                m = chunk.message
                if m.tool_calls:
                    result.calls.extend(m.tool_calls)
                if getattr(m, "thinking", None):
                    result.thinking += m.thinking
                    if state == "thinking":
                        reasoning.add(m.thinking)
                        thinking.refresh()
                if m.content:
                    result.content += m.content
                    if state == "thinking":
                        thinking.stop()
                        self.console.print()
                        live.start()
                        state = "live"
                    buf += m.content
                    committed, buf = split_committed(buf)
                    if committed.strip():
                        self._print_block(committed)
                    live.update(self._pending(buf), refresh=True)
        finally:
            if state == "thinking":
                thinking.stop()
            else:
                live.stop()
            if buf.strip():
                self._print_block(buf)

    @contextlib.contextmanager
    def working(self, message="Working…"):
        live = Live(Thinking(message), console=self.console, refresh_per_second=10, transient=True)
        live.start()
        try:
            yield
        finally:
            live.stop()

    def live_output(self, command):
        return LiveOutput(command, self.console)

    # ------------------------------------------------------------- tools
    def tool_call(self, name, args):
        label, arg = tool_label(name, args)
        line = Text()
        line.append("● ", style="green")
        line.append(label, style="bold")
        if arg:
            line.append(f"({arg})", style="dim")
        self.console.print()
        self.console.print(line, no_wrap=True, overflow="ellipsis", crop=True)

    def tool_result(self, name, args, result):
        if name == "todo_write" and not str(result).startswith("Error"):
            return
        text, style, extra = summarize_result(name, result)
        line = Text("  ⎿  ", style="dim")
        line.append(text, style=style)
        self.console.print(line, no_wrap=True, overflow="ellipsis", crop=True)
        for e in extra:
            self.console.print(Text("     " + e, style="dim"), no_wrap=True, overflow="ellipsis", crop=True)

    def show_todos(self, todos):
        out = Text()
        for i, td in enumerate(todos):
            done = td["status"] == "completed"
            active = td["status"] == "in_progress"
            out.append("  ⎿  " if i == 0 else "     ", style="dim")
            style = "dim strike" if done else (f"bold {ACCENT}" if active else "")
            out.append(("☒ " if done else "☐ ") + td["content"] + "\n", style=style)
        out.rstrip()
        self.console.print()
        self.console.print(Text.assemble(("● ", "green"), ("Update Todos", "bold")))
        self.console.print(out)

    # ---------------------------------------------------------- approvals
    def _body(self, body, kind, full=False):
        if kind == "diff":
            return render_diff(body, max_lines=None if full else 60)
        if kind == "command":
            parts = [Syntax(body, "bash", theme="ansi_dark", background_color="default", word_wrap=True),
                     Text(f"cwd: {self.root}", style="dim")]
            outside = outside_paths(body, self.root)
            if outside:
                parts.append(Text("⚠ touches paths outside the project: " + ", ".join(outside), style="yellow"))
            return Group(*parts)
        return Text(str(body))

    def _panel(self, title, body, kind, full=False):
        self.console.print(Panel(self._body(body, kind, full), title=Text(title, style="bold"), title_align="left",
                                 border_style=ACCENT, padding=(0, 1)))

    def _select(self, options, title):
        return select(options, title, input=self._pt_input, output=self._pt_output)

    def _ask_text(self, prompt):
        from prompt_toolkit import PromptSession
        session = PromptSession(input=self._pt_input, output=self._pt_output)
        return session.prompt([(f"fg:{ACCENT}", prompt)]).strip()

    def approve(self, title, body, kind="text"):
        self._panel(title, body, kind)
        options = ["Yes", "No", "No, and tell the agent what to do differently"]
        extra = None
        if kind == "diff" and len(body.splitlines()) > 62:
            extra = "view"
            options.append("View full diff")
        elif kind == "command" and self.allow_cb and not SHELL_META.search(body) and allow_prefix(body):
            extra = "allow"
            options.append(f"Yes, and always allow commands starting with '{allow_prefix(body)}'")
        while True:
            choice = self.select_fn(options, "Do you want to proceed?")
            if choice == 3 and extra == "view":
                self._panel(title, body, kind, full=True)
                continue
            break
        if choice == 3 and extra == "allow":
            self.allow_cb(allow_prefix(body))
            self.info(f"Commands starting with '{allow_prefix(body)}' will run without asking (/allow to manage).")
            return True, ""
        if choice == 0:
            return True, ""
        if choice == 2:
            self.console.print(Text("What should the agent do instead?", style="dim"))
            fb = self._ask_text("  ❯ ")
            if fb:
                return False, f"User denied this action with feedback: {fb}"
        return False, "User denied this action. Do not retry it unchanged; ask what to do instead."

    def approve_batch(self, items):
        """Show several diffs at once. Returns 'all', 'each' or 'none'."""
        for title, diff in items:
            self._panel(title, diff, "diff")
        n = len(items)
        choice = self.select_fn([f"Yes, apply all {n} changes", "No, review them one by one", "No, reject all"],
                                f"The agent proposes {n} file changes. Apply them?")
        return {0: "all", 1: "each"}.get(choice, "none")

    def confirm_plan(self):
        choice = self.select_fn(["Approve the plan and start implementing", "Keep planning (stay read-only)"],
                                "Plan mode: the agent has finished its turn.")
        return choice == 0

    def pick(self, options, title):
        """Generic menu; returns the chosen index or None."""
        return self.select_fn(list(options), title)

    # -------------------------------------------------------------- input
    def _build_session(self):
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory, InMemoryHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.styles import Style as PTStyle

        history = InMemoryHistory()
        path = Path(self._history_path) if self._history_path else Path.home() / ".anyagent" / "history"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(path))
        except OSError:
            pass

        kb = KeyBindings()

        @kb.add("enter")
        def _enter(event):
            buf = event.current_buffer
            if buf.document.text_before_cursor.endswith("\\"):
                buf.delete_before_cursor(1)
                buf.insert_text("\n")
            else:
                buf.validate_and_handle()

        @kb.add("escape", "enter")
        @kb.add("c-j")
        def _newline(event):
            event.current_buffer.insert_text("\n")

        self._session = PromptSession(
            history=history, completer=InputCompleter(self.root, self.commands, self.dynamic), complete_while_typing=True,
            key_bindings=kb, input=self._pt_input, output=self._pt_output,
            style=PTStyle.from_dict({"bottom-toolbar": "noreverse fg:#808080", "bottom-toolbar.text": "noreverse fg:#808080"}))

    def _toolbar(self):
        s = self.status
        parts = [f" {self.root.name}"]
        if s.get("model"):
            parts.append(s["model"])
        if s.get("ctx"):
            used, ctx = s.get("used") or 0, s["ctx"]
            parts.append(f"ctx {fmt_tokens(used)}/{fmt_tokens(ctx)} ({int(100 * used / ctx)}%)")
        if s.get("tps"):
            parts.append(f"{s['tps']:.0f} tok/s")
        if s.get("cost"):
            parts.append(f"${s['cost']:.3f}")
        if s.get("think"):
            parts.append("think")
        if s.get("plan"):
            parts.append("PLAN MODE (read-only)")
        return " · ".join(parts)

    def read_input(self):
        if self._session is None:
            self._build_session()
        self.console.print()
        self.console.print(Rule(style="grey37"))
        try:
            text = self._session.prompt(
                [(f"fg:{ACCENT} bold", "> ")], bottom_toolbar=self._toolbar,
                placeholder=[("fg:#6c6c6c", "Ask anything · / for commands · @ to attach a file")])
        except EOFError:
            return None
        except KeyboardInterrupt:
            if self._armed:
                return None
            self._armed = True
            self.warn("Press Ctrl-C again to exit (or Ctrl-D).")
            return ""
        self._armed = False
        return text.strip()
