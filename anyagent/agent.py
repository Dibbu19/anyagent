#!/usr/bin/env python3
"""AnyAgent: a local coding agent with a Claude Code-style terminal UI.

- Read-only exploration tools run automatically.
- EVERY file edit and EVERY shell command requires your explicit approval (unless you opt a
  command prefix into the allow list, or run with --yes).
- All file access is confined to the --root directory (default: current directory).

Usage:
    any                                  # interactive, in the current directory
    any --root ~/code/myproject          # interactive, in another directory
    any "fix the failing tests"          # one-shot
    any --continue                       # resume the last session for this directory
    any --yes --sandbox --json "task"    # headless, for scripts and CI
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

from anyagent import __version__
from anyagent.providers import NS, PROVIDERS, ProviderError, make_provider, with_retries
from anyagent.tools import (CONFIG_DIR, FILE_TOOLS, PARALLEL_TOOLS, PROJECT_DIR, TOOL_NAMES, TOOLS, Tools,
                            load_config)
from anyagent.ui import UI, BaseUI, HeadlessUI, StreamResult, fmt_tokens

DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "qwen3-coder:30b")
DEFAULT_NUM_CTX = int(os.environ.get("AGENT_NUM_CTX", "32768"))
DEFAULT_TEMPERATURE = os.environ.get("AGENT_TEMPERATURE")  # unset = the model's own Modelfile default
MAX_STEPS = 40        # max model calls per user request
SUBAGENT_STEPS = 20
COMPACT_AT = 0.85     # auto-compact when the prompt fills this fraction of the context window
LOOP_NUDGE_AT = 3     # identical call + identical result this many times -> warn the model
LOOP_STOP_AT = 6      # ... this many times -> stop the turn
ERRORS_NUDGE_AT = 4   # consecutive tool errors -> warn
ERRORS_STOP_AT = 8    # consecutive tool errors -> stop the turn
KEEP_SESSIONS = 20    # per project directory
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
SESSIONS_DIR = CONFIG_DIR / "sessions"
HARNESS_FILE = Path(__file__).with_name("HARNESS.md")

COMMANDS = [
    ("/help", "Show commands and key bindings"),
    ("/clear", "Clear the conversation history"),
    ("/compact", "Summarize the conversation to free up context"),
    ("/context", "Show context window usage"),
    ("/stats", "Tokens, speed and cost for this session"),
    ("/model", "Show or switch the model"),
    ("/think", "Toggle the model's thinking mode"),
    ("/plan", "Toggle plan mode (read-only until you approve the plan)"),
    ("/undo", "Revert the most recent approved file change"),
    ("/diff", "Show what changed since the session started (git)"),
    ("/commit", "Stage everything and commit (message generated unless given)"),
    ("/remember", "Save a fact to .anyagent/memory.md for future sessions"),
    ("/resume", "List saved sessions for this directory; /resume 2 loads one"),
    ("/allow", "Show/add/remove auto-approved command prefixes: /allow, /allow git status, /allow rm git status"),
    ("/jobs", "List background jobs (/jobs kill <id> to stop one)"),
    ("/mcp", "List MCP servers, their tools, and recent stderr"),
    ("/init", "Create an AGENTS.md describing this project"),
    ("/exit", "Exit"),
]

SUBAGENT_PROMPT = """You are a sub-agent inside a coding project with a fresh context. Complete the task you are given, then reply with a precise report: relevant paths with line numbers, the facts you verified, and a short conclusion.
"""
SUBAGENT_READONLY = "You can list, find, read and search files only. Do not propose edits you cannot make; just report.\n"

INIT_PROMPT = (
    "Explore this project and create an AGENTS.md at the project root for future coding sessions. "
    "Include: what the project is, how to build/run/test/lint it (exact commands), the architecture in a few "
    "bullets, and code style conventions you can verify from the code. Keep it under 60 lines. "
    "Base everything on files you actually read.")

COMPACT_PROMPT = (
    "Summarize the coding session below so work can continue seamlessly in a fresh context. Include: the user's "
    "goals and requests; files read, created or modified and the key changes; commands run and their results; "
    "errors hit and how they were fixed; the current state and the next steps. Be specific (paths, function "
    "names, exact commands). Under 600 words. Output only the summary.\n\n=== SESSION ===\n")

COMMIT_PROMPT = ("Write a git commit message for the diff below: a Conventional Commits subject line under 72 "
                 "characters, then a blank line, then at most 3 short bullets. Output only the message.\n\n")

PLAN_NOTE = ("[Plan mode is on: you are read-only. Explore, then present a numbered implementation plan as text. "
             "Do not call edit_file/write_file/run_shell; the user will approve the plan first.]\n\n")


# ------------------------------------------------------------------ system prompt
def load_harness(root: Path) -> str:
    """Model-facing rules: the part of HARNESS.md above the policy marker (project override wins)."""
    for f in (root / PROJECT_DIR / "HARNESS.md", HARNESS_FILE):
        if f.is_file():
            text = f.read_text(errors="replace")
            return text.split("<!-- policy -->")[0].strip()
    return "You are a careful coding agent. Explore before editing, keep edits small, verify with tests."


def load_project_context(root: Path) -> str:
    out = ""
    for name in ("AGENTS.md", "CLAUDE.md"):
        f = root / name
        if f.is_file():
            out += f"\n\nProject instructions from {name}:\n{f.read_text(errors='replace')[:6000]}"
            break
    mem = root / PROJECT_DIR / "memory.md"
    if mem.is_file():
        out += f"\n\nProject memory (facts saved in earlier sessions):\n{mem.read_text(errors='replace')[:4000]}"
    return out


def build_system(root: Path, tools: Tools) -> str:
    names = [t["function"]["name"] for t in tools.schemas()]
    return (load_harness(root) + "\n\nTools available: " + ", ".join(names) +
            f"\nWorkspace root: {root}" + load_project_context(root))


MENTION_RE = re.compile(r"(?<!\S)@(\S+)")


def expand_mentions(text: str, tools: Tools, max_chars: int = 20000, images_ok: bool = False):
    """Attach files referenced as @path. Returns (new_text, [attached names], [base64 images])."""
    blocks, names, images, seen = [], [], [], set()
    for m in MENTION_RE.finditer(text):
        token = m.group(1)
        for cand in (token, token.rstrip(",.;:!?)")):
            try:
                p = tools._resolve(cand)
            except ValueError:
                continue
            if not p.is_file():
                continue
            rel = tools._rel(p)
            if rel in seen:
                break
            if p.suffix.lower() in IMAGE_EXT:
                if images_ok:
                    seen.add(rel)
                    images.append(base64.b64encode(p.read_bytes()).decode())
                    names.append(rel)
                break
            if not tools._is_binary(p):
                seen.add(rel)
                data = tools._read(p)[0]
                note = "" if len(data) <= max_chars else f"\n[truncated: file has {len(data)} chars]"
                blocks.append(f'<file path="{rel}">\n{data[:max_chars]}{note}\n</file>')
                names.append(rel)
                break
    if blocks:
        text = text + "\n\nAttached files:\n" + "\n".join(blocks)
    return text, names, images


def model_options(num_ctx: int, temperature=None) -> dict:
    opts = {"num_ctx": num_ctx}
    if temperature is not None and temperature != "":
        opts["temperature"] = float(temperature)
    return opts


class TokenMeter:
    """Character-based prompt estimate, calibrated from the server's reported counts.

    Ollama's prompt_eval_count can exclude cached prompt tokens, so the agent always uses
    max(reported, estimate) and only tightens the ratio when the report looks complete.
    """

    def __init__(self, chars_per_token: float = 4.0):
        self.ratio = chars_per_token

    @staticmethod
    def chars(messages, tools_schema=None) -> int:
        n = len(json.dumps(tools_schema or [], ensure_ascii=False))
        for m in messages:
            n += len(str(m.get("content") or ""))
            if m.get("tool_calls"):
                n += len(json.dumps(m["tool_calls"], ensure_ascii=False))
            n += 1000 * len(m.get("images") or [])
        return n

    def estimate(self, messages, tools_schema=None) -> int:
        return int(self.chars(messages, tools_schema) / self.ratio)

    def observe(self, messages, tools_schema, reported: int) -> int:
        est = self.estimate(messages, tools_schema)
        if reported and reported >= est * 0.5:  # a count far below the estimate is a cache artefact
            self.ratio = min(8.0, max(2.0, 0.7 * self.ratio + 0.3 * self.chars(messages, tools_schema) / reported))
        return max(reported or 0, est)


METER = TokenMeter()


def estimate_tokens(messages, tools_schema=None) -> int:
    return METER.estimate(messages, tools_schema)


# ------------------------------------------------- tool calls sent as text
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_XML_FUNC = re.compile(r"<function=([\w.\-]+)>(.*?)</function>", re.S)
_XML_PARAM = re.compile(r"<parameter=([\w.\-]+)>\s*(.*?)\s*</parameter>", re.S)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _param_types(name: str, extra_schemas=None) -> dict:
    for t in list(TOOLS) + list(extra_schemas or []):
        if t["function"]["name"] == name:
            return {k: v.get("type", "string") for k, v in t["function"]["parameters"]["properties"].items()}
    return {}


def _coerce(value: str, typ: str):
    if typ == "string":
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _mk_call(name: str, args: dict):
    return NS(function=NS(name=name, arguments=args))


def parse_text_tool_calls(content: str, known=None, extra_schemas=None):
    """Recover tool calls some servers emit as plain text.

    Handles `<tool_call>{json}</tool_call>`, Qwen3-Coder's `<tool_call><function=x><parameter=y>…`
    form, and a lone ```json {"name":…, "arguments":…}``` fence. Returns (clean_text, calls).
    """
    known = known or (TOOL_NAMES | {t["function"]["name"] for t in (extra_schemas or [])})
    calls = []

    def from_json(obj):
        if isinstance(obj, dict) and obj.get("name") in known:
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            calls.append(_mk_call(obj["name"], args if isinstance(args, dict) else {}))
            return True
        return False

    def handle_block(m):
        inner = m.group(1).strip()
        try:
            if from_json(json.loads(inner)):
                return ""
        except ValueError:
            pass
        found = False
        for fm in _XML_FUNC.finditer(inner):
            name = fm.group(1)
            if name not in known:
                continue
            types = _param_types(name, extra_schemas)
            args = {k: _coerce(v, types.get(k, "string")) for k, v in _XML_PARAM.findall(fm.group(2))}
            calls.append(_mk_call(name, args))
            found = True
        return "" if found else m.group(0)

    text = _TOOL_CALL_BLOCK.sub(handle_block, content or "")
    if not calls:
        def handle_fence(m):
            try:
                return "" if from_json(json.loads(m.group(1))) else m.group(0)
            except ValueError:
                return m.group(0)
        text = _JSON_FENCE.sub(handle_fence, text)
    return (text.strip() if calls else content), calls


def _transcript(messages, limit_chars: int) -> str:
    parts = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            parts.append(f"USER: {m.get('content', '')}")
        elif role == "assistant":
            calls = "; ".join(f"{c['function']['name']}({json.dumps(c['function']['arguments'])[:300]})"
                              for c in m.get("tool_calls", []))
            parts.append("ASSISTANT: " + (m.get("content") or "") + (f"\n[tool calls: {calls}]" if calls else ""))
        elif role == "tool":
            parts.append(f"TOOL RESULT ({m.get('tool_name')}): {str(m.get('content', ''))[:1500]}")
    return "\n\n".join(parts)[-limit_chars:]


def _tail_cut(messages, window: int = 8) -> int:
    """Index of the most recent user message (within `window`) so the latest exchange survives compaction."""
    for i in range(len(messages) - 1, max(0, len(messages) - window) - 1, -1):
        if messages[i].get("role") == "user":
            return i
    return len(messages)


def compact(provider, messages, model: str, num_ctx: int, ui=None, focus: str = "", temperature=None) -> str:
    """Replace the history with a model-written summary, keeping the latest exchange verbatim."""
    ui = ui or BaseUI()
    cut = _tail_cut(messages[1:]) + 1
    if cut <= 1:
        cut = len(messages)
    prompt = COMPACT_PROMPT + _transcript(messages[1:cut], limit_chars=num_ctx * 2)
    if focus:
        prompt += f"\n\n=== EXTRA FOCUS FOR THE SUMMARY ===\n{focus}"
    with ui.working("Compacting conversation…"):
        resp = with_retries(lambda: provider.chat(
            model=model, options=model_options(num_ctx, temperature), keep_alive="30m",
            messages=[{"role": "system", "content": "You write precise summaries of coding sessions."},
                      {"role": "user", "content": prompt}]), ui)
    summary = (resp.message.content or "").strip() or "(no summary produced)"
    tail = messages[cut:]
    messages[1:] = [
        {"role": "user", "content": "This session continues from an earlier conversation. Summary of it:\n\n" + summary},
        {"role": "assistant", "content": "Understood. I have the summary and will continue from there."},
    ] + tail
    return summary


def run_turn(provider, tools: Tools, messages: list, model: str, num_ctx: int, ui=None, temperature=None,
             max_steps: int = MAX_STEPS, read_only: bool = False, think=None, stats=None) -> dict:
    """Run the model/tool loop until the model answers without tool calls.

    Returns {"prompt_tokens": last prompt size or None, "completion_tokens": total generated}.
    """
    ui = ui or tools.ui
    usage = {"prompt_tokens": None, "completion_tokens": 0}
    schemas = tools.schemas(read_only=read_only)
    options = model_options(num_ctx, temperature)
    seen: dict = {}  # (tool, args, result) -> repeat count, for loop detection
    errors_in_a_row = 0
    for _ in range(max_steps):
        result = StreamResult()

        def step():
            result.content, result.thinking, result.calls = "", "", []
            stream = provider.chat(model=model, messages=messages, tools=schemas, stream=True, keep_alive="30m",
                                   options=options, think=think)
            ui.consume_stream(stream, result)

        try:
            with_retries(step, ui)
        except KeyboardInterrupt:
            partial = (result.content.rstrip() + "\n" if result.content.strip() else "") + "[Request interrupted by user]"
            messages.append({"role": "assistant", "content": partial})
            ui.warn("Interrupted. Tell the agent what to do next.")
            return usage

        if not result.calls and result.content:
            result.content, result.calls = parse_text_tool_calls(result.content, extra_schemas=schemas)

        pt = getattr(result.last, "prompt_eval_count", None) or 0
        ct = getattr(result.last, "eval_count", None) or 0
        usage["prompt_tokens"] = METER.observe(messages, schemas, pt)
        usage["completion_tokens"] += ct
        if stats is not None:
            stats.append({"prompt": pt, "completion": ct, "seconds": time.time() - result.t0})

        assistant = {"role": "assistant", "content": result.content}
        if result.calls:
            assistant["tool_calls"] = [
                {"function": {"name": tc.function.name, "arguments": dict(tc.function.arguments or {})}}
                for tc in result.calls]
        messages.append(assistant)
        if not result.calls:
            return usage

        calls = [(tc.function.name, dict(tc.function.arguments or {})) for tc in result.calls]
        pre = tools.batch_approve(calls) if sum(n in FILE_TOOLS for n, _ in calls) >= 2 else {}
        outputs: list = [None] * len(calls)
        if len(calls) > 1 and all(n in PARALLEL_TOOLS for n, _ in calls):
            try:
                outputs = tools.dispatch_parallel(calls)
            except KeyboardInterrupt:
                for name, _ in calls:
                    messages.append({"role": "tool", "tool_name": name, "content": "Interrupted by user."})
                ui.warn("Interrupted. Tell the agent what to do next.")
                return usage
        stop = False
        for i, (name, args) in enumerate(calls):
            if outputs[i] is not None:
                out = outputs[i]
            elif isinstance(pre.get(i), str):
                out = pre[i]
                ui.tool_call(name, args)
                ui.tool_result(name, args, out)
            else:
                try:
                    out = tools.dispatch(name, args, preapproved=pre.get(i) is True)
                except KeyboardInterrupt:
                    for rest_name, _ in calls[i:]:
                        messages.append({"role": "tool", "tool_name": rest_name,
                                         "content": "Interrupted by user before this tool finished."})
                    ui.warn("Interrupted. Tell the agent what to do next.")
                    return usage
            key = (name, json.dumps(args, sort_keys=True, default=str), out)
            seen[key] = seen.get(key, 0) + 1
            errors_in_a_row = errors_in_a_row + 1 if str(out).startswith("Error") else 0
            if seen[key] >= LOOP_STOP_AT or errors_in_a_row >= ERRORS_STOP_AT:
                stop = True
            elif seen[key] >= LOOP_NUDGE_AT:
                out += (f"\n\n[Loop detected: this exact call has returned the same result {seen[key]} times. "
                        "Do not repeat it. Change your approach or ask the user for help.]")
            elif errors_in_a_row >= ERRORS_NUDGE_AT:
                out += (f"\n\n[{errors_in_a_row} tool calls in a row have failed. Stop and re-read the file or "
                        "the error carefully, or ask the user for help, instead of trying variations.]")
            messages.append({"role": "tool", "tool_name": name, "content": out})
        if stop:
            ui.warn("Stopped: the agent kept failing the same way. Tell it what to do differently.")
            messages.append({"role": "assistant", "content": "[Stopped: repeated failing tool calls.]"})
            return usage

        if usage["prompt_tokens"] and usage["prompt_tokens"] >= COMPACT_AT * num_ctx and not read_only:
            ui.warn("Context is nearly full — compacting the conversation…")
            compact(provider, messages, model, num_ctx, ui, temperature=temperature)
            messages.append({"role": "user", "content": "Continue with the task you were working on."})
            usage["prompt_tokens"] = METER.estimate(messages, schemas)
    ui.warn(f"Stopped after {max_steps} steps. Send another message to continue.")
    return usage


# ------------------------------------------------------------------ session
def session_dir(root: Path) -> Path:
    return SESSIONS_DIR / re.sub(r"[^\w.-]+", "%", str(root)).strip("%")


def list_sessions(root: Path) -> list:
    """Saved sessions for a directory, newest first: [(path, started, first user message)]."""
    out = []
    for f in sorted(session_dir(root).glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            first = next((m["content"] for m in data.get("messages", []) if m.get("role") == "user"), "")
            out.append((f, data.get("started", f.stem), " ".join(first.split())[:80]))
        except (OSError, ValueError):
            continue
    return out


def load_custom_commands(root: Path) -> dict:
    """<root>/.anyagent/commands/<name>.md -> /name; $ARGUMENTS is replaced by what follows the command."""
    cmds = {}
    for f in sorted((root / PROJECT_DIR / "commands").glob("*.md")):
        try:
            cmds["/" + f.stem] = f.read_text(errors="replace")
        except OSError:
            continue
    return cmds


class Session:
    def __init__(self, provider, tools: Tools, ui, model: str, num_ctx: int, system: str, temperature=None,
                 provider_name: str = "ollama", think=None, pricing=None):
        self.provider, self.tools, self.ui = provider, tools, ui
        self.model, self.num_ctx, self.temperature = model, num_ctx, temperature
        self.provider_name, self.think = provider_name, think
        self.pricing = pricing or {}   # {"model": {"input": $/1M, "output": $/1M}}
        self.messages = [{"role": "system", "content": system}]
        self.prompt_tokens = None
        self.plan_mode = False
        self.stats: list = []
        self.custom = load_custom_commands(tools.root)
        self.file = None               # set on first save
        self.started = time.strftime("%Y%m%d-%H%M%S")
        self.start_tree = self._git_head()
        self.tools.subagent = self.run_subagent
        self.sync_status()

    def _git_head(self) -> str:
        if not self.tools.git:
            return ""
        try:
            rc, out = self.tools._git("stash", "create")
            if rc == 0 and not out:
                rc, out = self.tools._git("rev-parse", "HEAD")
            return out.split()[0] if rc == 0 and out else ""
        except Exception:
            return ""

    # ------------------------------------------------------------------ stats
    def cost(self) -> float:
        price = self.pricing.get(self.model) or {}
        if not price:
            return 0.0
        p = sum(s["prompt"] for s in self.stats)
        c = sum(s["completion"] for s in self.stats)
        return (p * price.get("input", 0) + c * price.get("output", 0)) / 1_000_000

    def sync_status(self):
        last = self.stats[-1] if self.stats else None
        tps = last["completion"] / last["seconds"] if last and last["seconds"] > 0 and last["completion"] else None
        self.ui.status = {"model": self.model, "ctx": self.num_ctx, "used": self.prompt_tokens,
                          "plan": self.plan_mode, "think": bool(self.think), "tps": tps, "cost": self.cost()}

    # ------------------------------------------------------------ persistence
    def save(self):
        try:
            d = session_dir(self.tools.root)
            d.mkdir(parents=True, exist_ok=True)
            if self.file is None:
                self.file = d / f"{self.started}.json"
            self.file.write_text(json.dumps({"root": str(self.tools.root), "model": self.model, "started": self.started,
                                             "time": time.time(), "messages": self.messages,
                                             "todos": self.tools.todos, "stats": self.stats}, ensure_ascii=False))
            old = sorted(d.glob("*.json"))[:-KEEP_SESSIONS]
            for f in old:
                f.unlink(missing_ok=True)
                f.with_suffix(".md").unlink(missing_ok=True)
        except OSError as e:
            self.ui.warn(f"Could not save the session: {e}")

    def resume(self, which=None) -> bool:
        """Load a saved session: the newest, or an index into list_sessions(), or a path."""
        sessions = list_sessions(self.tools.root)
        if isinstance(which, Path):
            path = which
        elif sessions:
            path = sessions[min(int(which or 0), len(sessions) - 1)][0]
        else:
            return False
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return False
        if len(data.get("messages") or []) < 2:
            return False
        self.messages[1:] = data["messages"][1:]
        self.tools.todos = data.get("todos") or []
        self.stats = data.get("stats") or []
        self.file, self.started = path, data.get("started", path.stem)
        self.prompt_tokens = METER.estimate(self.messages, self.tools.schemas())
        self.sync_status()
        return True

    def _log(self, role: str, text: str):
        try:
            d = session_dir(self.tools.root)
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{self.started}.md", "a", encoding="utf-8") as f:
                f.write(f"\n## {role} · {time.strftime('%Y-%m-%d %H:%M')}\n\n{text.rstrip()}\n")
        except OSError:
            pass

    # ------------------------------------------------------------------ turns
    def run_subagent(self, task: str, quiet: bool = False) -> str:
        read_only = self.tools.read_only
        system = SUBAGENT_PROMPT + (SUBAGENT_READONLY if read_only else "") + f"\nWorkspace root: {self.tools.root}"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
        if not quiet:
            self.ui.info(f"  ⎿  sub-agent started ({'read-only' if read_only else 'can edit'})")
        self.tools.in_subagent = True
        try:
            run_turn(self.provider, self.tools, messages, self.model, self.num_ctx, BaseUI() if quiet else self.ui,
                     self.temperature, max_steps=SUBAGENT_STEPS, read_only=read_only, think=self.think,
                     stats=self.stats)
        finally:
            self.tools.in_subagent = False
        final = next((m.get("content") for m in reversed(messages) if m["role"] == "assistant" and m.get("content")), "")
        return final or "(the sub-agent produced no report)"

    def turn(self, text: str) -> None:
        text, attached, images = expand_mentions(text, self.tools, images_ok=getattr(self.provider, "supports_images", False))
        for name in attached:
            self.ui.info(f"  ⎿  Attached {name}")
        self._log("user", text)
        msg = {"role": "user", "content": (PLAN_NOTE if self.plan_mode else "") + text}
        if images:
            msg["images"] = images
        self.messages.append(msg)
        try:
            usage = run_turn(self.provider, self.tools, self.messages, self.model, self.num_ctx, self.ui,
                             self.temperature, think=self.think, stats=self.stats)
            if usage["prompt_tokens"] is not None:
                self.prompt_tokens = usage["prompt_tokens"]
            if self.prompt_tokens and self.prompt_tokens >= COMPACT_AT * self.num_ctx:
                self.ui.warn("Context is nearly full — compacting the conversation…")
                compact(self.provider, self.messages, self.model, self.num_ctx, self.ui, temperature=self.temperature)
                self.prompt_tokens = METER.estimate(self.messages, self.tools.schemas())
        except KeyboardInterrupt:
            self.ui.warn("Interrupted.")
        except Exception as e:
            self.ui.error(f"Error talking to the model: {e}")
            if self.messages[-1]["role"] == "user":
                self.messages.pop()  # keep the history well-formed for the next attempt
        last = self.messages[-1]
        if last["role"] == "assistant" and last.get("content"):
            self._log("assistant", last["content"])
        self.sync_status()
        self.save()
        if self.plan_mode and self.ui.confirm_plan():
            self.set_plan_mode(False)
            self.turn("The plan is approved. Implement it now.")

    def set_plan_mode(self, on: bool):
        self.plan_mode = self.tools.plan_mode = on
        self.sync_status()
        self.ui.info("Plan mode ON: the agent is read-only until you approve its plan." if on
                     else "Plan mode OFF: edits and commands ask for approval as usual.")

    # ------------------------------------------------------------------ git
    def diff(self) -> str:
        if not self.tools.git:
            return "Not a git repository."
        base = self.start_tree or "HEAD"
        rc, out = self.tools._git("diff", "--stat", base)
        rc2, untracked = self.tools._git("ls-files", "--others", "--exclude-standard")
        rc3, full = self.tools._git("diff", base)
        text = (out or "(no changes to tracked files)")
        if untracked:
            text += "\n\nNew untracked files:\n" + untracked
        return text + ("\n\n" + full[:6000] + ("\n… (truncated)" if len(full) > 6000 else "") if full else "")

    def commit(self, message: str = "") -> str:
        if not self.tools.git:
            return "Not a git repository."
        rc, status = self.tools._git("status", "--short")
        if not status:
            return "Nothing to commit."
        if not message:
            rc, diff = self.tools._git("diff", "HEAD")
            rc2, untracked = self.tools._git("ls-files", "--others", "--exclude-standard")
            with self.ui.working("Writing a commit message…"):
                resp = with_retries(lambda: self.provider.chat(
                    model=self.model, options=model_options(self.num_ctx, self.temperature), keep_alive="30m",
                    messages=[{"role": "user", "content": COMMIT_PROMPT + status + "\n\n" + diff[:12000] +
                               (f"\n\nNew files:\n{untracked}" if untracked else "")}]), self.ui)
            message = (resp.message.content or "").strip().strip("`") or "Update"
        ok, why = self.tools.approve("Commit", f"git add -A && git commit\n\n{message}", kind="command")
        if not ok:
            return "Commit cancelled."
        rc, out = self.tools._git("add", "-A")
        if rc != 0:
            return f"Error: {out}"
        rc, out = self.tools._git("commit", "-m", message)
        return out if rc == 0 else f"Error: {out}"

    def remember(self, fact: str) -> str:
        mem = self.tools.root / PROJECT_DIR / "memory.md"
        mem.parent.mkdir(parents=True, exist_ok=True)
        with open(mem, "a", encoding="utf-8") as f:
            if mem.stat().st_size == 0:
                f.write("# Project memory\n\n")
            f.write(f"- {fact.strip()}\n")
        self.messages[0]["content"] = build_system(self.tools.root, self.tools)
        return f"Saved to {mem.relative_to(self.tools.root)} (loaded into every session for this project)."

    def command(self, line: str) -> bool:
        """Handle a slash command. Returns False when the app should exit."""
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        if cmd in ("/exit", "/quit"):
            return False
        if cmd in self.custom:
            self.turn(self.custom[cmd].replace("$ARGUMENTS", arg))
        elif cmd == "/help":
            self.ui.show_help(COMMANDS + [(c, "custom command") for c in sorted(self.custom)])
        elif cmd in ("/clear", "/reset"):
            del self.messages[1:]
            self.prompt_tokens = None
            self.tools.todos = []
            self.sync_status()
            self.save()
            self.ui.info("Conversation cleared.")
        elif cmd == "/compact":
            if len(self.messages) <= 1:
                self.ui.info("Nothing to compact yet.")
            else:
                try:
                    summary = compact(self.provider, self.messages, self.model, self.num_ctx, self.ui, arg,
                                      self.temperature)
                    self.prompt_tokens = METER.estimate(self.messages, self.tools.schemas())
                    self.sync_status()
                    self.save()
                    self.ui.info("Conversation compacted. Summary:\n" + summary[:600] + ("…" if len(summary) > 600 else ""))
                except KeyboardInterrupt:
                    self.ui.warn("Interrupted.")
                except Exception as e:
                    self.ui.error(f"Could not compact: {e}")
        elif cmd == "/context":
            used = self.prompt_tokens
            est = ""
            if used is None:
                used = METER.estimate(self.messages, self.tools.schemas())
                est = " (estimated)"
            pct = min(100, int(100 * used / self.num_ctx))
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            self.ui.info(f"Context: {bar} {pct}%  {fmt_tokens(used)} / {fmt_tokens(self.num_ctx)} tokens{est}\n"
                         f"Messages: {len(self.messages) - 1} · model: {self.model} · ~{METER.ratio:.1f} chars/token")
        elif cmd == "/stats":
            p = sum(s["prompt"] for s in self.stats)
            c = sum(s["completion"] for s in self.stats)
            secs = sum(s["seconds"] for s in self.stats)
            tps = c / secs if secs else 0
            cost = f" · cost ${self.cost():.4f}" if self.pricing.get(self.model) else ""
            self.ui.info(f"Model calls: {len(self.stats)} · prompt {fmt_tokens(p)} · generated {fmt_tokens(c)} · "
                         f"{secs:.0f}s · {tps:.1f} tok/s{cost}")
        elif cmd == "/model":
            if not arg:
                try:
                    names = self.provider.list_models()
                except Exception as e:
                    names = []
                    self.ui.error(f"Could not list models: {e}")
                for n in names:
                    self.ui.info(f"  {'●' if n == self.model else ' '} {n}")
                self.ui.info("Switch with: /model <name>")
            else:
                try:
                    self.provider.show(arg)
                except Exception as e:
                    self.ui.error(f"Cannot use model {arg!r}: {e}")
                else:
                    self.model = arg
                    self.sync_status()
                    self.ui.info(f"Model set to {arg}.")
        elif cmd == "/think":
            self.think = (arg == "on") if arg in ("on", "off") else (not self.think)
            self.sync_status()
            self.ui.info(f"Thinking {'ON' if self.think else 'OFF'}" +
                         (" (models without thinking support will error; turn it off again)" if self.think else ""))
        elif cmd == "/plan":
            self.set_plan_mode(not self.plan_mode if arg not in ("on", "off") else arg == "on")
        elif cmd == "/undo":
            self.ui.info(self.tools.undo())
        elif cmd == "/diff":
            self.ui.info(self.diff())
        elif cmd == "/commit":
            try:
                self.ui.info(self.commit(arg))
            except KeyboardInterrupt:
                self.ui.warn("Interrupted.")
            except Exception as e:
                self.ui.error(f"Commit failed: {e}")
        elif cmd == "/remember":
            self.ui.info(self.remember(arg) if arg else "Usage: /remember <fact>")
        elif cmd == "/resume":
            sessions = list_sessions(self.tools.root)
            if not sessions:
                self.ui.info("No saved sessions for this directory.")
            elif arg.isdigit():
                self.ui.info("Resumed." if self.resume(int(arg) - 1) else "Could not load that session.")
            else:
                shown = sessions[:10]
                idx = self.ui.pick([f"{s[1]}  {s[2] or '(empty)'}" for s in shown], "Resume which session?")
                if idx is None:
                    for i, s in enumerate(shown, 1):
                        self.ui.info(f"  {i}. {s[1]}  {s[2]}")
                    self.ui.info("Load one with: /resume <number>")
                else:
                    self.ui.info("Resumed." if self.resume(idx) else "Could not load that session.")
        elif cmd == "/allow":
            if not arg:
                self.ui.info("Auto-approved command prefixes:\n" + ("\n".join(f"  {p}" for p in self.tools.allow)
                                                                    or "  (none)"))
            elif arg.startswith("rm "):
                self.ui.info("Removed." if self.tools.remove_allow(arg[3:].strip()) else "Not in the list.")
            else:
                self.tools.add_allow(arg)
                self.ui.info(f"Commands starting with '{arg}' now run without asking.")
        elif cmd == "/jobs":
            if arg.startswith("kill "):
                self.ui.info(self.tools.check_job(arg[5:].strip(), kill=True).splitlines()[0])
            else:
                jobs = self.tools.list_jobs()
                self.ui.info("\n".join(f"  #{j} {s}: {c}" for j, s, c in jobs) or "No background jobs.")
        elif cmd == "/mcp":
            servers = {srv for srv, _, _ in self.tools.mcp.values()}
            if not servers:
                self.ui.info(f"No MCP servers. Configure them in .mcp.json or {CONFIG_DIR / 'config.json'}.")
            for srv in sorted(servers, key=lambda s: s.name):
                names = [n for n, (s, _, _) in self.tools.mcp.items() if s is srv]
                self.ui.info(f"{srv.name}: " + ", ".join(sorted(names)))
                if srv.log():
                    self.ui.info("  stderr: " + srv.log()[-500:].replace("\n", "\n          "))
        elif cmd == "/init":
            self.turn(INIT_PROMPT)
        else:
            self.ui.warn(f"Unknown command {cmd}. Type /help.")
        return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="any", description="AnyAgent: local coding agent")
    ap.add_argument("prompt", nargs="*", help="optional one-shot task; omit for interactive mode")
    ap.add_argument("--version", action="version", version=f"anyagent {__version__}")
    ap.add_argument("--root", default=".", help="project directory the agent may access (default: cwd)")
    ap.add_argument("--provider", default=os.environ.get("AGENT_PROVIDER", "ollama"), choices=sorted(PROVIDERS))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--num-ctx", type=int, default=None,
                    help=f"context window tokens (default: min(model's, {DEFAULT_NUM_CTX}))")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help="sampling temperature (default: the model's own)")
    ap.add_argument("--host", default=None, help="model server URL (default: provider's env var or local)")
    ap.add_argument("--think", action="store_true", help="enable the model's thinking mode")
    ap.add_argument("--continue", dest="resume", action="store_true", help="resume the last session for this directory")
    ap.add_argument("--sandbox", action="store_true", help="run shell commands under sandbox-exec (macOS) / bwrap (Linux)")
    ap.add_argument("--yes", action="store_true", help="approve every action without asking (use with --sandbox)")
    ap.add_argument("--json", action="store_true", help="one-shot, no UI: print a JSON result (implies --yes)")
    ap.add_argument("--shell-timeout", type=int, default=None, help="default seconds before a command is killed")
    ap.add_argument("--allow", action="append", default=[], help="auto-approve commands with this prefix (repeatable)")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    headless = args.json
    ui = HeadlessUI() if headless else UI(root=root)
    ui.commands = COMMANDS
    if not root.is_dir():
        ui.error(f"Root is not a directory: {root}")
        return 2
    if headless and not args.prompt:
        ui.error("--json needs a one-shot prompt.")
        return 2

    try:
        provider = make_provider(args.provider, args.host)
        provider.show(args.model)
    except Exception as e:
        ui.error(f"Cannot use model {args.model!r}: {e}")
        ui.info("Is the model server running and is the model available (`ollama list`, or the server's /models)?")
        return 2
    note = provider.check()
    if note:
        ui.warn(note)
    num_ctx = args.num_ctx
    if num_ctx is None:
        limit = provider.context_length(args.model)
        num_ctx = min(DEFAULT_NUM_CTX, int(limit)) if limit else DEFAULT_NUM_CTX
        if limit and limit < DEFAULT_NUM_CTX:
            ui.warn(f"Model context is {fmt_tokens(limit)} tokens; using that.")

    config = load_config(root)
    tools = Tools(root, ui=ui, allow=list(config.get("allow") or []) + args.allow, sandbox=args.sandbox,
                  shell_timeout=args.shell_timeout, deny=config.get("deny"), hooks=config.get("hooks"))
    if args.yes or headless:
        tools.approve = lambda title, body, kind="text": (True, "")
        if not args.sandbox and not headless:
            ui.warn("--yes: every edit and command will run without asking. Consider --sandbox.")
    if hasattr(provider, "web_search") and os.environ.get("OLLAMA_API_KEY"):
        tools.searcher = provider.web_search
    from anyagent.mcp import load_servers
    servers = []
    for srv, name, err in load_servers(config, root):
        if srv:
            servers.append(srv)
            ui.info(f"MCP {name}: {len(tools.register_mcp(srv))} tools")
        else:
            ui.warn(f"MCP {name} failed to start: {err}")

    session = Session(provider, tools, ui, args.model, num_ctx, build_system(root, tools), args.temperature,
                      args.provider, think=True if args.think else None, pricing=config.get("pricing"))
    if not headless:
        ui.dynamic = {"/model": lambda: provider.list_models(), "/allow": lambda: ["rm " + p for p in tools.allow],
                      "/resume": lambda: [str(i) for i in range(1, len(list_sessions(root)) + 1)]}
        ui.commands = COMMANDS + [(c, "custom command") for c in sorted(session.custom)]
    resumed = args.resume and session.resume()
    if args.resume and not resumed:
        ui.warn("No saved session for this directory; starting fresh.")
    ui.banner(args.model, root, num_ctx, provider=args.provider, resumed=resumed, sandbox=args.sandbox)

    try:
        if args.prompt:
            session.turn(" ".join(args.prompt))
            if headless:
                final = next((m.get("content") for m in reversed(session.messages)
                              if m["role"] == "assistant" and m.get("content")), "")
                print(json.dumps({"result": final, "model": args.model, "session": str(session.file),
                                  "usage": {"prompt_tokens": session.prompt_tokens,
                                            "completion_tokens": sum(s["completion"] for s in session.stats)},
                                  "tools": ui.log}, ensure_ascii=False, indent=2))
            return 0
        while True:
            text = ui.read_input()
            if text is None:
                break
            if not text:
                continue
            first = text.split()[0]
            if first.startswith("/") and "/" not in first[1:]:
                if not session.command(text):
                    break
            else:
                session.turn(text)
    finally:
        tools.kill_jobs()
        for s in servers:
            s.close()
    ui.info("Bye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
