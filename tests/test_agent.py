"""Tests that need no model server. Run with: python -m unittest -v tests/test_agent.py"""
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

import anyagent.agent as agent
import anyagent.tools as agent_tools
import anyagent.ui as agent_ui
from anyagent import mcp

# keep every test away from the real ~/.anyagent
_HOME = tempfile.TemporaryDirectory()
agent_tools.CONFIG_DIR = Path(_HOME.name)
agent_tools.CONFIG_PATH = agent_tools.CONFIG_DIR / "config.json"
agent.SESSIONS_DIR = agent_tools.CONFIG_DIR / "sessions"


def make_tools(root, answers, **kw):
    """Tools whose approval prompt returns scripted answers: True / False / 'feedback text'."""
    calls = []

    def approve(title, body, kind="text"):
        calls.append((title, kind))
        a = answers.pop(0)
        if a is True:
            return True, ""
        return False, ("User denied this action." if a is False else f"feedback: {a}")

    t = agent_tools.Tools(Path(root), approve=approve, ui=agent_ui.BaseUI(), **kw)
    t.approval_calls = calls
    return t


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        (self.root / "app.py").write_text("def add(a, b):\n    return a - b\n")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "x.txt").write_text("hello\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_path_confinement(self):
        t = make_tools(self.root, [])
        self.assertIn("outside the workspace", t.dispatch("read_file", {"path": "../etc/passwd"}))
        self.assertIn("outside the workspace", t.dispatch("read_file", {"path": "/etc/hosts"}))

    def test_read_search_list_find(self):
        t = make_tools(self.root, [])
        self.assertIn("return a - b", t.read_file("app.py"))
        self.assertIn("app.py:2", t.search_text("return"))
        self.assertIn("sub/", t.list_dir("."))
        self.assertIn("sub/x.txt", t.find_files("*.txt"))

    def test_edit_requires_approval_and_applies(self):
        t = make_tools(self.root, [False, True])
        self.assertIn("denied", t.edit_file("app.py", "a - b", "a + b"))
        self.assertIn("a - b", (self.root / "app.py").read_text())  # unchanged
        self.assertIn("Edited", t.edit_file("app.py", "a - b", "a + b"))
        self.assertIn("a + b", (self.root / "app.py").read_text())
        self.assertEqual(t.approval_calls, [("Edit app.py", "diff"), ("Edit app.py", "diff")])

    def test_edit_errors_do_not_prompt(self):
        t = make_tools(self.root, [])
        self.assertIn("not found", t.edit_file("app.py", "nope", "x"))
        (self.root / "dup.py").write_text("x\nx\n")
        self.assertIn("matches 2", t.edit_file("dup.py", "x", "y"))
        self.assertEqual(t.approval_calls, [])

    # --- problem 1: line endings are preserved
    def test_edit_keeps_crlf(self):
        (self.root / "win.txt").write_bytes(b"one\r\ntwo\r\nthree\r\n")
        t = make_tools(self.root, [True])
        self.assertIn("Edited", t.edit_file("win.txt", "two\n", "TWO\n"))
        self.assertEqual((self.root / "win.txt").read_bytes(), b"one\r\nTWO\r\nthree\r\n")
        self.assertIn("CRLF", t.read_file("win.txt"))
        t2 = make_tools(self.root, [True])
        t2.write_file("win.txt", "a\nb\n")
        self.assertEqual((self.root / "win.txt").read_bytes(), b"a\r\nb\r\n")

    # --- problem 2: non-UTF-8 files
    def test_edit_keeps_encoding(self):
        (self.root / "latin.txt").write_bytes("caf\xe9 one\n".encode("cp1252"))
        (self.root / "bom.txt").write_bytes(b"\xef\xbb\xbfhello\n")
        t = make_tools(self.root, [True, True])
        self.assertIn("caf\xe9", t.read_file("latin.txt"))
        self.assertNotIn("\ufffd", t.read_file("latin.txt"))
        self.assertIn("Edited", t.edit_file("latin.txt", "one", "two"))
        self.assertEqual((self.root / "latin.txt").read_bytes(), "caf\xe9 two\n".encode("cp1252"))
        t.edit_file("bom.txt", "hello", "bye")
        self.assertEqual((self.root / "bom.txt").read_bytes(), b"\xef\xbb\xbfbye\n")

    # --- problem 3: whitespace-tolerant matching + closest lines
    def test_edit_fuzzy_whitespace(self):
        (self.root / "ws.py").write_text("def f():   \n    return 1  \n")
        t = make_tools(self.root, [True, True])
        r = t.edit_file("ws.py", "def f():\n    return 1\n", "def f():\n    return 2\n")
        self.assertIn("trailing whitespace", r)
        self.assertEqual((self.root / "ws.py").read_text(), "def f():\n    return 2\n")
        (self.root / "ind.py").write_text("class A:\n        def g(self):\n            return 1\n")
        r = t.edit_file("ind.py", "    def g(self):\n        return 1\n", "    def g(self):\n        return 2\n")
        self.assertIn("indentation", r)
        self.assertEqual((self.root / "ind.py").read_text(), "class A:\n        def g(self):\n            return 2\n")
        r = t.edit_file("app.py", "    return a -- b\n", "x")
        self.assertIn("not found", r)
        self.assertIn("Closest lines", r)
        self.assertIn("2:     return a - b", r)

    def test_write_file_gated(self):
        t = make_tools(self.root, [False, True])
        self.assertIn("denied", t.write_file("new/a.py", "print(1)\n"))
        self.assertFalse((self.root / "new" / "a.py").exists())
        self.assertIn("Created", t.write_file("new/a.py", "print(1)\n"))
        self.assertTrue((self.root / "new" / "a.py").exists())

    def test_shell_gated(self):
        t = make_tools(self.root, ["use pytest instead", True])
        self.assertIn("feedback: use pytest instead", t.run_shell("touch marker"))
        self.assertFalse((self.root / "marker").exists())
        r = t.run_shell("echo hi && touch marker")
        self.assertIn("[exit code 0]", r)
        self.assertIn("hi", r)
        self.assertTrue((self.root / "marker").exists())
        self.assertEqual(t.approval_calls[0][1], "command")

    # --- problem 4: stdout/stderr interleaved in order
    def test_shell_merges_streams_in_order(self):
        t = make_tools(self.root, [True])
        r = t.run_shell("echo one; echo two >&2; echo three")
        self.assertIn("one\ntwo\nthree", r)
        self.assertNotIn("[stderr]", r)

    def test_shell_timeout_kills_process_group(self):
        t = make_tools(self.root, [True])
        r = t.run_shell("sleep 30 & sleep 30", timeout=1)
        self.assertIn("timed out after 1s", r)

    # --- problem 5: paths outside the project are flagged; sandbox confines writes
    def test_outside_paths_warning(self):
        self.assertEqual(agent_tools.outside_paths("cat /etc/hosts && ls ~/Desktop", self.root), ["/etc/hosts", "~/Desktop"])
        self.assertEqual(agent_tools.outside_paths("cd ../other && ls", self.root), ["../other"])
        self.assertEqual(agent_tools.outside_paths(f"ls {self.root}/sub ./x", self.root), [])
        self.assertEqual(agent_tools.outside_paths("pytest -q", self.root), [])

    @unittest.skipUnless(sys.platform == "darwin", "sandbox-exec is macOS only")
    def test_sandbox_blocks_writes_outside_root(self):
        t = make_tools(self.root, [True, True], sandbox=True)
        outside = Path(tempfile.gettempdir()).resolve().parent / f"oa-sandbox-{os.getpid()}"
        with tempfile.TemporaryDirectory() as d:
            # $TMPDIR is allowed; the user's home is not
            self.assertIn("[exit code 0]", t.run_shell(f"touch {d}/ok && touch inside"))
            self.assertTrue((self.root / "inside").exists())
        r = t.run_shell(f"touch ~/.oa-sandbox-test-{os.getpid()}")
        self.assertNotIn("[exit code 0]", r)
        self.assertFalse((Path.home() / f".oa-sandbox-test-{os.getpid()}").exists())

    # --- background jobs
    def test_background_job(self):
        t = make_tools(self.root, [True])
        r = t.run_shell("echo start; sleep 0.3; echo end", background=True)
        self.assertIn("Started background job 1", r)
        time.sleep(0.1)
        self.assertIn("running", t.check_job(1))
        time.sleep(0.5)
        out = t.check_job(1)
        self.assertIn("exited with code 0", out)
        self.assertIn("end", out)
        self.assertEqual(t.list_jobs()[0][1], "exit 0")
        self.assertIn("Error: no job", t.check_job(9))
        t.kill_jobs()

    # --- checkpoints / undo
    def test_undo_restores_previous_content(self):
        t = make_tools(self.root, [True, True])
        t.edit_file("app.py", "a - b", "a + b")
        t.write_file("fresh.txt", "new\n")
        self.assertIn("Removed fresh.txt", t.undo())
        self.assertFalse((self.root / "fresh.txt").exists())
        self.assertIn("Restored app.py", t.undo())
        self.assertIn("a - b", (self.root / "app.py").read_text())
        self.assertEqual(t.undo(), "Nothing to undo.")

    # --- allow list
    def test_allow_list_skips_prompt_for_simple_commands_only(self):
        t = make_tools(self.root, [True], allow=["git status", "echo"])
        self.assertIn("[exit code 0]", t.run_shell("echo allowed"))
        self.assertEqual(t.approval_calls, [])
        t.run_shell("echo chained; touch nope")  # metachars: must still ask
        self.assertEqual(len(t.approval_calls), 1)
        t.add_allow("ls")
        self.assertEqual(json.loads(agent_tools.CONFIG_PATH.read_text())["allow"], ["git status", "echo", "ls"])
        self.assertTrue(t.remove_allow("ls"))

    # --- batch approval
    def test_batch_approve(self):
        class BatchUI(agent_ui.BaseUI):
            def __init__(self): self.batches = []
            def approve_batch(self, items):
                self.batches.append([t for t, _ in items])
                return "all"
        ui = BatchUI()
        t = agent_tools.Tools(self.root, ui=ui)  # approve() would deny: only batch pre-approval can pass
        calls = [("write_file", {"path": "a.txt", "content": "a\n"}), ("write_file", {"path": "b.txt", "content": "b\n"})]
        pre = t.batch_approve(calls)
        self.assertEqual(pre, {0: True, 1: True})
        self.assertEqual(ui.batches, [["Create a.txt", "Create b.txt"]])
        for i, (n, a) in enumerate(calls):
            self.assertIn("Created", t.dispatch(n, a, preapproved=pre[i] is True))
        # two edits to the same file depend on each other -> one by one
        same = [("edit_file", {"path": "app.py", "old_string": "a - b", "new_string": "x"}),
                ("edit_file", {"path": "app.py", "old_string": "x", "new_string": "y"})]
        self.assertEqual(t.batch_approve(same), {})

    # --- plan mode / sub-agent read-only
    def test_plan_mode_and_read_only_block_mutations(self):
        t = make_tools(self.root, [])
        t.plan_mode = True
        self.assertIn("Plan mode", t.edit_file("app.py", "a - b", "a + b"))
        self.assertIn("Plan mode", t.run_shell("ls"))
        t.plan_mode, t.read_only = False, True
        self.assertIn("not available to a sub-agent", t.dispatch("write_file", {"path": "z", "content": ""}))
        self.assertIn("return a - b", t.dispatch("read_file", {"path": "app.py"}))
        self.assertEqual(t.approval_calls, [])
        self.assertEqual({s["function"]["name"] for s in t.schemas(read_only=True)}, agent_tools.READ_ONLY_TOOLS)

    def test_web_fetch_validation_and_html(self):
        t = make_tools(self.root, [])
        self.assertIn("only http", t.web_fetch("ftp://x"))
        self.assertEqual(agent_tools.html_to_text("<html><style>x{}</style><p>Hi&amp;<b>there</b></p><p>two</p>"),
                         "Hi& there\ntwo")

    def test_todo_write(self):
        t = make_tools(self.root, [])
        self.assertIn("updated", t.dispatch("todo_write", {"todos": [
            {"content": "a", "status": "in_progress"}, {"content": "b", "status": "weird"}]}))
        self.assertEqual([x["status"] for x in t.todos], ["in_progress", "pending"])
        self.assertIn("Error", t.dispatch("todo_write", {"todos": "nope"}))

    def test_unknown_tool_and_bad_args(self):
        t = make_tools(self.root, [])
        self.assertIn("unknown tool", t.dispatch("rm_rf", {}))
        self.assertIn("bad arguments", t.dispatch("read_file", {"wrong": 1}))

    def test_expand_mentions(self):
        t = make_tools(self.root, [])
        text, names, images = agent.expand_mentions("look at @app.py, and @sub/x.txt plus @missing.py and @../etc/passwd", t)
        self.assertEqual(sorted(names), ["app.py", "sub/x.txt"])
        self.assertIn('<file path="app.py">', text)
        self.assertIn("return a - b", text)
        self.assertEqual(agent.expand_mentions("no mentions here", t)[1], [])
        (self.root / "shot.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00fake")
        self.assertEqual(agent.expand_mentions("see @shot.png", t)[2], [])  # provider without vision
        text, names, images = agent.expand_mentions("see @shot.png", t, images_ok=True)
        self.assertEqual(names, ["shot.png"])
        self.assertEqual(len(images), 1)
        self.assertEqual(text, "see @shot.png")


FAKE_MCP_SERVER = r'''
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    m, i = msg.get("method"), msg.get("id")
    if m == "initialize":
        out = {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "fake"}}
    elif m == "tools/list":
        out = {"tools": [{"name": "echo", "description": "Echo text", "inputSchema": {
            "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]}
    elif m == "tools/call":
        a = msg["params"]["arguments"]
        out = {"content": [{"type": "text", "text": "echo: " + a.get("text", "")}], "isError": a.get("text") == "fail"}
    else:
        continue
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": i, "result": out}) + "\n"); sys.stdout.flush()
'''


class MCPTests(unittest.TestCase):
    def test_stdio_server_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "srv.py"
            script.write_text(FAKE_MCP_SERVER)
            (Path(d) / ".mcp.json").write_text(json.dumps({"mcpServers": {"fake": {"command": sys.executable, "args": [str(script)]}}}))
            started = mcp.load_servers({}, d)
            self.assertEqual([(n, e) for _, n, e in started], [("fake", "")])
            srv = started[0][0]
            try:
                t = make_tools(d, [True, True])
                self.assertEqual(t.register_mcp(srv), ["mcp__fake__echo"])
                self.assertIn("mcp__fake__echo", [s["function"]["name"] for s in t.schemas()])
                self.assertEqual(t.dispatch("mcp__fake__echo", {"text": "hi"}), "echo: hi")
                self.assertEqual(t.dispatch("mcp__fake__echo", {"text": "fail"}), "Error: echo: fail")
                self.assertEqual(t.approval_calls[0], ("MCP fake: echo", "text"))
                self.assertEqual(agent_ui.tool_label("mcp__fake__echo", {"text": "hi"}), ("MCP fake:echo", "text=hi"))
            finally:
                srv.close()


class TextToolCallTests(unittest.TestCase):
    def test_json_block(self):
        text, calls = agent.parse_text_tool_calls('Let me look.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py", "limit": 5}}\n</tool_call>')
        self.assertEqual(text, "Let me look.")
        self.assertEqual((calls[0].function.name, calls[0].function.arguments), ("read_file", {"path": "a.py", "limit": 5}))

    def test_qwen_xml_block(self):
        raw = ("<tool_call>\n<function=edit_file>\n<parameter=path>\napp.py\n</parameter>\n<parameter=old_string>\n"
               "a - b\n</parameter>\n<parameter=new_string>\na + b\n</parameter>\n<parameter=replace_all>\ntrue\n"
               "</parameter>\n</function>\n</tool_call>")
        text, calls = agent.parse_text_tool_calls(raw)
        self.assertEqual(text, "")
        self.assertEqual(calls[0].function.arguments, {"path": "app.py", "old_string": "a - b", "new_string": "a + b", "replace_all": True})

    def test_json_fence_and_unknown_names(self):
        text, calls = agent.parse_text_tool_calls('```json\n{"name": "search_text", "arguments": {"pattern": "x"}}\n```')
        self.assertEqual(calls[0].function.name, "search_text")
        text, calls = agent.parse_text_tool_calls('```json\n{"name": "nope", "arguments": {}}\n```\nplain **md**')
        self.assertEqual(calls, [])
        self.assertIn("plain **md**", text)
        self.assertEqual(agent.parse_text_tool_calls("no calls here")[1], [])


class RenderHelperTests(unittest.TestCase):
    def test_split_committed_respects_code_fences(self):
        done, pending = agent_ui.split_committed("Para one.\n\nPara two start")
        self.assertEqual(done, "Para one.\n\n")
        self.assertEqual(pending, "Para two start")
        buf = "Intro\n\n```py\nx = 1\n\ny = 2\n"  # blank line inside an open fence
        done, pending = agent_ui.split_committed(buf)
        self.assertEqual(done, "Intro\n\n")
        self.assertTrue(pending.startswith("```py"))
        done, pending = agent_ui.split_committed(buf + "```\n\nAfter")
        self.assertIn("y = 2", done)
        self.assertEqual(pending, "After")

    def test_render_diff_and_summaries(self):
        diff = agent_tools.unified("a\nb\nc\n", "a\nB\nc\nd\n", "a/f", "b/f")
        console = Console(file=io.StringIO(), width=80, record=True, force_terminal=False)
        console.print(agent_ui.render_diff(diff))
        out = console.export_text()
        self.assertIn("Added 2 lines, removed 1 line", out)
        self.assertIn("2 - b", out)
        self.assertIn("2 + B", out)
        self.assertEqual(agent_ui.tool_label("read_file", {"path": "x.py"}), ("Read", "x.py"))
        self.assertEqual(agent_ui.summarize_result("search_text", "a:1: x\nb:2: y")[0], "Found 2 matches")
        text, style, extra = agent_ui.summarize_result("run_shell", "[exit code 1]\nboom\nbad")
        self.assertEqual((text, style, extra), ("exit code 1", "red", ["boom", "bad"]))
        self.assertEqual(agent_ui.summarize_result("edit_file", "User denied this action with feedback: no")[0],
                         "Rejected — no")
        self.assertEqual(agent_ui.summarize_result("check_job", "[job 1: running]\nline")[:2], ("running", "green"))
        self.assertEqual(agent_ui.allow_prefix("git status --short"), "git status")
        self.assertEqual(agent_ui.allow_prefix("pytest -q"), "pytest")

    def test_model_options_and_estimate(self):
        self.assertEqual(agent.model_options(4096), {"num_ctx": 4096})
        self.assertEqual(agent.model_options(4096, "0.2"), {"num_ctx": 4096, "temperature": 0.2})
        msgs = [{"role": "user", "content": "x" * 4000}]
        self.assertGreaterEqual(agent.estimate_tokens(msgs), 1000)


class UITests(unittest.TestCase):
    def make_ui(self, root):
        console = Console(file=io.StringIO(), width=100, record=True, force_terminal=False, highlight=False)
        return agent_ui.UI(root=root, console=console, history_path=Path(root) / ".hist"), console

    def test_stream_renders_markdown_with_single_bullet(self):
        with tempfile.TemporaryDirectory() as d:
            ui, console = self.make_ui(d)
            chunks = ["Hello **world**.\n\n", "Second para\n\n```py\nprint(1)\n", "\nprint(2)\n```\n"]
            stream = iter([NS(message=NS(content=c, tool_calls=None)) for c in chunks])
            res = agent_ui.StreamResult()
            ui.consume_stream(stream, res)
            out = console.export_text()
            self.assertEqual(res.content, "".join(chunks))
            self.assertEqual(out.count("●"), 1)
            self.assertIn("Hello world.", out)
            self.assertIn("print(2)", out)

    def test_tool_lines_and_approval_flow(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "app.py").write_text("def add(a, b):\n    return a - b\n")
            ui, console = self.make_ui(root)
            picks = iter([0, 1])  # approve first edit, reject second
            ui.select_fn = lambda options, title: next(picks)
            tools = agent_tools.Tools(root, ui=ui)
            tools.dispatch("read_file", {"path": "app.py"})
            tools.dispatch("edit_file", {"path": "app.py", "old_string": "a - b", "new_string": "a + b"})
            tools.dispatch("edit_file", {"path": "app.py", "old_string": "a + b", "new_string": "b + a"})
            out = console.export_text()
            self.assertIn("● Read(app.py)", out)
            self.assertIn("Read 2 lines", out)
            self.assertIn("● Update(app.py)", out)
            self.assertIn("Edit app.py", out)
            self.assertIn("Edited app.py", out)
            self.assertIn("Rejected", out)
            self.assertIn("a + b", (root / "app.py").read_text())

    def test_feedback_denial(self):
        with tempfile.TemporaryDirectory() as d:
            ui, _ = self.make_ui(d)
            ui.select_fn = lambda options, title: 2
            ui._ask_text = lambda prompt: "use pytest"
            ok, why = ui.approve("Bash command", "pytest", kind="command")
            self.assertFalse(ok)
            self.assertIn("use pytest", why)

    # --- problem 6: full diff on demand
    def test_view_full_diff_option(self):
        with tempfile.TemporaryDirectory() as d:
            ui, console = self.make_ui(d)
            content = "\n".join(f"line {i}" for i in range(200)) + "\n"
            diff = agent_tools.unified("", content, "/dev/null", "b/big.txt")
            picks = iter([3, 0])
            seen_options = []
            ui.select_fn = lambda options, title: (seen_options.append(list(options)), next(picks))[1]
            self.assertEqual(ui.approve("Create big.txt", diff, kind="diff"), (True, ""))
            out = console.export_text()
            self.assertIn("View full diff", seen_options[0][3])
            self.assertIn("+140 more lines", out)
            self.assertIn("line 199", out)

    # --- "always allow" menu option + outside-path warning in the panel
    def test_always_allow_option_and_outside_warning(self):
        with tempfile.TemporaryDirectory() as d:
            ui, console = self.make_ui(d)
            tools = agent_tools.Tools(d, ui=ui)
            seen = []
            ui.select_fn = lambda options, title: (seen.append(list(options)), 3)[1]
            self.assertEqual(ui.approve("Bash command", "git status --short", kind="command"), (True, ""))
            self.assertEqual(tools.allow, ["git status"])
            self.assertIn("always allow commands starting with 'git status'", seen[0][3])
            ui.select_fn = lambda options, title: (seen.append(list(options)), 1)[1]
            ui.approve("Bash command", "cat /etc/hosts | head", kind="command")
            self.assertEqual(len(seen[1]), 3)  # piped command: no always-allow shortcut
            self.assertIn("outside the project: /etc/hosts", console.export_text())

    def test_select_menu_keys(self):
        opts = ["Yes", "No", "Tell me"]

        def run(keys):
            with create_pipe_input() as inp:
                inp.send_text(keys)
                return agent_ui.select(opts, input=inp, output=DummyOutput())

        self.assertEqual(run("\r"), 0)
        self.assertEqual(run("\x1b[B\r"), 1)       # down, enter
        self.assertEqual(run("\x1b[B\x1b[B\x1b[B\r"), 0)  # wraps around
        self.assertEqual(run("3"), 2)
        self.assertEqual(run("n"), 1)
        self.assertEqual(run("y"), 0)
        with self.assertRaises(KeyboardInterrupt):
            run("\x03")

    def test_read_input(self):
        with tempfile.TemporaryDirectory() as d:
            with create_pipe_input() as inp:
                console = Console(file=io.StringIO(), width=100)
                ui = agent_ui.UI(root=d, console=console, pt_input=inp, pt_output=DummyOutput(),
                                 history_path=Path(d) / ".hist")
                inp.send_text("hello there\r")
                self.assertEqual(ui.read_input(), "hello there")
                inp.send_text("line one\\\rline two\r")  # backslash + enter = newline
                self.assertEqual(ui.read_input(), "line one\nline two")
                inp.send_text("\x03")
                self.assertEqual(ui.read_input(), "")      # first Ctrl-C only warns
                inp.send_text("\x03")
                self.assertIsNone(ui.read_input())          # second Ctrl-C exits

    def test_completer(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "src").mkdir()
            (Path(d) / "src" / "main.py").write_text("x")
            from prompt_toolkit.document import Document
            comp = agent_ui.InputCompleter(Path(d), agent.COMMANDS)
            names = [c.text for c in comp.get_completions(Document("/co"), None)]
            self.assertEqual(sorted(names), ["/commit", "/compact", "/context"])
            comp.dynamic = {"/model": lambda: ["qwen3-coder:30b", "llama3"]}
            self.assertEqual([c.text for c in comp.get_completions(Document("/model qw"), None)], ["qwen3-coder:30b"])
            files = [c.text for c in comp.get_completions(Document("see @sr"), None)]
            self.assertIn("@src/main.py", files)

    def test_live_output_renders_tail(self):
        with tempfile.TemporaryDirectory() as d:
            ui, console = self.make_ui(d)
            with ui.live_output("make test") as live:
                live.update(["a\n", "b\n"])
            console.print(live)
            self.assertIn("b", console.export_text())


class FakeClient:
    """Scripted model: first turn calls edit_file, second turn gives a final answer."""

    def __init__(self, prompt_tokens=None):
        self.n = 0
        self.prompt_tokens = prompt_tokens
        self.kwargs = []

    def chat(self, **kwargs):
        self.n += 1
        self.kwargs.append(kwargs)
        if self.n == 1:
            tc = NS(function=NS(name="edit_file", arguments={
                "path": "app.py", "old_string": "a - b", "new_string": "a + b"}))
            return iter([NS(message=NS(content="Fixing the bug.", tool_calls=[tc]),
                            prompt_eval_count=self.prompt_tokens, eval_count=7)])
        return iter([NS(message=NS(content="Done.", tool_calls=None))])


class ScriptedClient:
    """Replays a list of (content, [(tool, args)]) steps; then answers 'Done.'"""

    def __init__(self, steps):
        self.steps = list(steps)
        self.kwargs = []

    def chat(self, **kwargs):
        self.kwargs.append(kwargs)
        if not self.steps:
            return iter([NS(message=NS(content="Done.", tool_calls=None))])
        content, calls = self.steps.pop(0)
        tcs = [NS(function=NS(name=n, arguments=a)) for n, a in calls]
        return iter([NS(message=NS(content=content, tool_calls=tcs or None), prompt_eval_count=10, eval_count=1)])


class LoopTests(unittest.TestCase):
    def test_agent_loop_edits_after_approval(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "app.py").write_text("def add(a, b):\n    return a - b\n")
            tools = make_tools(root, [True])
            messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "fix add"}]
            usage = agent.run_turn(FakeClient(prompt_tokens=100000), tools, messages, "fake", 400000)
            self.assertIn("a + b", (root / "app.py").read_text())
            self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool", "assistant"])
            self.assertEqual(messages[3]["tool_name"], "edit_file")
            self.assertEqual(usage["completion_tokens"], 7)
            self.assertGreater(usage["prompt_tokens"], 0)  # last chunk reports nothing: the estimate is used

    # --- problem 7: context meter never trusts an under-reported count
    def test_prompt_tokens_use_estimate_when_reported_count_is_low(self):
        with tempfile.TemporaryDirectory() as d:
            tools = make_tools(d, [])
            messages = [{"role": "system", "content": "x" * 8000}, {"role": "user", "content": "hi"}]
            client = ScriptedClient([])
            usage = agent.run_turn(client, tools, messages, "fake", 4096)
            self.assertGreaterEqual(usage["prompt_tokens"], 2000)  # reported None/0, estimate wins
            self.assertNotIn("temperature", client.kwargs[0]["options"])  # problem 9: model default
            agent.run_turn(ScriptedClient([]), tools, messages, "fake", 4096, temperature=0.2)

    # --- problem 8: text tool calls are executed
    def test_text_tool_calls_are_executed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "a.py").write_text("secret = 1\n")
            tools = make_tools(root, [])
            text = '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>'
            client = ScriptedClient([(text, [])])
            messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "read"}]
            agent.run_turn(client, tools, messages, "fake", 4096)
            self.assertEqual(messages[2]["tool_calls"][0]["function"]["name"], "read_file")
            self.assertEqual(messages[2]["content"], "")
            self.assertIn("secret = 1", messages[3]["content"])

    # --- problem 10: loop detection
    def test_loop_detection_nudges_then_stops(self):
        with tempfile.TemporaryDirectory() as d:
            tools = make_tools(d, [])
            call = ("read_file", {"path": "missing.py"})
            client = ScriptedClient([("", [call])] * 10)
            messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]
            agent.run_turn(client, tools, messages, "fake", 4096)
            tool_msgs = [m["content"] for m in messages if m["role"] == "tool"]
            self.assertEqual(len(tool_msgs), agent.LOOP_STOP_AT)
            self.assertNotIn("Loop detected", tool_msgs[1])
            self.assertIn("Loop detected", tool_msgs[agent.LOOP_NUDGE_AT - 1])
            self.assertIn("repeated failing tool calls", messages[-1]["content"])

    def test_batch_approval_in_loop(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            ui = agent_ui.BaseUI()
            ui.approve_batch = lambda items: "all"
            tools = agent_tools.Tools(root, ui=ui)
            client = ScriptedClient([("", [("write_file", {"path": "a.txt", "content": "a"}),
                                           ("write_file", {"path": "b.txt", "content": "b"})])])
            messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]
            agent.run_turn(client, tools, messages, "fake", 4096)
            self.assertTrue((root / "a.txt").exists() and (root / "b.txt").exists())
            ui.approve_batch = lambda items: "none"
            agent.run_turn(ScriptedClient([("", [("write_file", {"path": "c.txt", "content": "c"}),
                                                 ("write_file", {"path": "d.txt", "content": "d"})])]),
                           tools, messages, "fake", 4096)
            self.assertFalse((root / "c.txt").exists())
            self.assertIn("rejected the whole batch", messages[-2]["content"])

    def test_interrupt_during_approval_keeps_history_consistent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "app.py").write_text("def add(a, b):\n    return a - b\n")

            def approve(title, body, kind="text"):
                raise KeyboardInterrupt

            tools = agent_tools.Tools(root, approve=approve, ui=agent_ui.BaseUI())
            messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "fix add"}]
            agent.run_turn(FakeClient(), tools, messages, "fake", 4096)
            self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool"])
            self.assertIn("Interrupted", messages[-1]["content"])
            self.assertIn("a - b", (root / "app.py").read_text())

    def test_compact_replaces_history(self):
        class Summarizer:
            def chat(self, **kw):
                self.kw = kw
                return NS(message=NS(content="SUMMARY TEXT"))

        c = Summarizer()
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "do a thing"},
                    {"role": "assistant", "content": "ok", "tool_calls": [
                        {"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]},
                    {"role": "tool", "tool_name": "read_file", "content": "file body"}]
        out = agent.compact(c, messages, "m", 4096)
        self.assertEqual(out, "SUMMARY TEXT")
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["content"], "sys")
        self.assertIn("SUMMARY TEXT", messages[1]["content"])
        self.assertIn("do a thing", c.kw["messages"][1]["content"])
        self.assertIn("read_file", c.kw["messages"][1]["content"])

    def test_auto_compact_mid_turn(self):
        class Client(FakeClient):
            def chat(self, **kw):
                if kw.get("stream"):
                    return super().chat(**kw)
                return NS(message=NS(content="SUM"))

        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "app.py").write_text("def add(a, b):\n    return a - b\n")
            tools = make_tools(root, [True])
            messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "fix add"}]
            agent.run_turn(Client(prompt_tokens=4000), tools, messages, "fake", 4096)
            self.assertIn("SUM", messages[1]["content"])
            self.assertEqual(messages[-1]["content"], "Done.")

    def test_subagent_runs_read_only_with_fresh_context(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "a.py").write_text("x = 1\n")
            client = ScriptedClient([("", [("run_subagent", {"task": "what is in a.py?"})]),      # main
                                     ("", [("read_file", {"path": "a.py"}), ("write_file", {"path": "b", "content": ""})]),  # sub
                                     ("Report: a.py sets x = 1", [])])                              # sub final
            ui = agent_ui.BaseUI()
            tools = agent_tools.Tools(root, ui=ui)
            s = agent.Session(client, tools, ui, "m", 4096, "system")
            s.turn("investigate")
            sub_msgs = client.kwargs[1]["messages"]
            self.assertIn("sub-agent", sub_msgs[0]["content"])
            self.assertEqual({t["function"]["name"] for t in client.kwargs[1]["tools"]}, agent_tools.READ_ONLY_TOOLS)
            self.assertFalse((root / "b").exists())
            report = next(m for m in s.messages if m["role"] == "tool" and m["tool_name"] == "run_subagent")
            self.assertEqual(report["content"], "Report: a.py sets x = 1")
            self.assertFalse(tools.read_only)

    def test_plan_mode_flow(self):
        with tempfile.TemporaryDirectory() as d:
            ui = agent_ui.BaseUI()
            ui.confirm_plan = lambda: True
            tools = agent_tools.Tools(d, ui=ui)
            client = ScriptedClient([("1. do x\n2. do y", [])])
            s = agent.Session(client, tools, ui, "m", 4096, "system")
            s.command("/plan")
            self.assertTrue(tools.plan_mode)
            s.turn("add a feature")
            self.assertIn("Plan mode is on", client.kwargs[0]["messages"][1]["content"])
            self.assertFalse(tools.plan_mode)  # approved -> implementation turn ran
            self.assertEqual(client.kwargs[1]["messages"][-2]["content"], "The plan is approved. Implement it now.")

    def test_session_save_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            ui = agent_ui.BaseUI()
            tools = agent_tools.Tools(d, ui=ui)
            s = agent.Session(ScriptedClient([]), tools, ui, "m", 4096, "system")
            s.turn("remember this")
            self.assertTrue(s.file.exists())
            self.assertIn("remember this", s.file.with_suffix(".md").read_text())
            s2 = agent.Session(ScriptedClient([]), agent_tools.Tools(d, ui=ui), ui, "m", 4096, "system")
            self.assertTrue(s2.resume())
            self.assertEqual([m["content"] for m in s2.messages[1:]], ["remember this", "Done."])
            self.assertEqual(s2.messages[0]["content"], "system")
            self.assertEqual(s2.file, s.file)
            s2.command("/clear")
            s3 = agent.Session(ScriptedClient([]), agent_tools.Tools(d, ui=ui), ui, "m", 4096, "system")
            self.assertFalse(s3.resume())

    def test_session_commands(self):
        with tempfile.TemporaryDirectory() as d:
            ui = agent_ui.BaseUI()
            tools = agent_tools.Tools(d, ui=ui)
            s = agent.Session(FakeClient(), tools, ui, "m", 4096, "system")
            s.messages.append({"role": "user", "content": "hi"})
            self.assertTrue(s.command("/clear"))
            self.assertEqual(len(s.messages), 1)
            for c in ("/context", "/nope", "/undo", "/allow", "/allow git status", "/allow rm git status", "/jobs", "/mcp",
                      "/plan off"):
                self.assertTrue(s.command(c))
            self.assertFalse(s.command("/exit"))


if __name__ == "__main__":
    unittest.main()
