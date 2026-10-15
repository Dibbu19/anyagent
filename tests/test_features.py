"""Tests for providers, patch tools, git checkpoints, sessions, MCP HTTP, headless mode. No model server needed."""
import http.server
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from rich.console import Console

import anyagent.agent as agent
import anyagent.tools as agent_tools
import anyagent.ui as agent_ui
from anyagent import mcp, providers

_HOME = tempfile.TemporaryDirectory()
agent_tools.CONFIG_DIR = Path(_HOME.name)
agent_tools.CONFIG_PATH = agent_tools.CONFIG_DIR / "config.json"
agent.SESSIONS_DIR = agent_tools.CONFIG_DIR / "sessions"


def yes_tools(root, **kw):
    return agent_tools.Tools(root, approve=lambda t, b, kind="text": (True, ""), ui=agent_ui.BaseUI(), **kw)


class Scripted:
    """Replays (content, [(tool, args)]) steps, then 'Done.'"""
    supports_images = True

    def __init__(self, steps=()):
        self.steps, self.kwargs = list(steps), []

    def chat(self, **kw):
        self.kwargs.append(kw)
        if not kw.get("stream"):
            return NS(message=NS(content="feat: scripted commit\n\n- did things"))
        if not self.steps:
            return iter([NS(message=NS(content="Done.", thinking="", tool_calls=None), prompt_eval_count=50, eval_count=3)])
        content, calls = self.steps.pop(0)
        tcs = [NS(function=NS(name=n, arguments=a)) for n, a in calls]
        return iter([NS(message=NS(content=content, thinking="", tool_calls=tcs or None), prompt_eval_count=50, eval_count=3)])


# ----------------------------------------------------------------- fake HTTP model servers
class _Handler(http.server.BaseHTTPRequestHandler):
    openai_events = []
    anthropic_events = []
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self._json({"data": [{"id": "fake-model", "context_length": 8192}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        events = self.openai_events if self.path.endswith("/chat/completions") else self.anthropic_events
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for ev in events:
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_openai_stream_tool_calls_and_history_conversion(self):
        _Handler.openai_events = [
            {"choices": [{"delta": {"content": "Look"}}]},
            {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "read_file", "arguments": "{\"pa"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "th\": \"a.py\"}"}}]}}]},
            {"choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": 9}},
        ]
        p = providers.OpenAIProvider(host=self.base + "/v1", api_key="k")
        p.show("fake-model")
        self.assertEqual(p.context_length("fake-model"), 8192)
        with self.assertRaises(providers.ProviderError):
            p.show("other")
        history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi", "images": ["AAA="]},
                   {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a"}}}]},
                   {"role": "tool", "tool_name": "read_file", "content": "body"}]
        res = agent_ui.StreamResult()
        agent_ui.BaseUI().consume_stream(p.chat("fake-model", history, tools=agent_tools.TOOLS, stream=True,
                                                options={"temperature": 0.1}), res)
        self.assertEqual(res.content, "Look")
        self.assertEqual(res.thinking, "hmm")
        self.assertEqual((res.calls[0].function.name, res.calls[0].function.arguments), ("read_file", {"path": "a.py"}))
        self.assertEqual((res.last.prompt_eval_count, res.last.eval_count), (42, 9))
        path, headers, body = _Handler.seen[-1]
        self.assertEqual(headers["authorization"], "Bearer k")
        self.assertEqual(body["temperature"], 0.1)
        self.assertEqual(body["messages"][1]["content"][1]["type"], "image_url")
        self.assertEqual(body["messages"][2]["tool_calls"][0]["id"], body["messages"][3]["tool_call_id"])
        self.assertEqual(json.loads(body["messages"][2]["tool_calls"][0]["function"]["arguments"]), {"path": "a"})
        non_stream = p.chat("fake-model", history)
        self.assertEqual(non_stream.message.content, "Look")

    def test_anthropic_stream_and_conversion(self):
        _Handler.anthropic_events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 11}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi "}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "t"}},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "search_text"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"pattern\": "}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"x\"}"}},
            {"type": "message_delta", "usage": {"output_tokens": 5}},
        ]
        p = providers.AnthropicProvider(host=self.base, api_key="sk")
        history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"},
                   {"role": "assistant", "content": "ok", "tool_calls": [{"function": {"name": "a", "arguments": {}}},
                                                                         {"function": {"name": "b", "arguments": {}}}]},
                   {"role": "tool", "tool_name": "a", "content": "ra"}, {"role": "tool", "tool_name": "b", "content": "rb"}]
        res = agent_ui.StreamResult()
        agent_ui.BaseUI().consume_stream(p.chat("claude", history, tools=agent_tools.TOOLS[:1], stream=True, think=True), res)
        self.assertNotIn("thinking", _Handler.seen[-1][2])  # thinking is skipped when tools are present
        res = agent_ui.StreamResult()
        agent_ui.BaseUI().consume_stream(p.chat("claude", history, stream=True, think=True, options={"temperature": 0.2}), res)
        self.assertEqual(res.content, "Hi ")
        self.assertEqual(res.calls[0].function.arguments, {"pattern": "x"})
        self.assertEqual((res.last.prompt_eval_count, res.last.eval_count), (11, 5))
        path, headers, body = _Handler.seen[-1]
        self.assertEqual(headers["x-api-key"], "sk")
        self.assertEqual(body["system"], "sys")
        self.assertEqual(_Handler.seen[-2][2]["tools"][0]["name"], "list_dir")
        self.assertEqual(body["thinking"]["type"], "enabled")
        self.assertNotIn("temperature", body)
        body = _Handler.seen[-2][2]
        tool_results = body["messages"][2]["content"]
        self.assertEqual([b["type"] for b in tool_results], ["tool_result", "tool_result"])
        self.assertEqual(tool_results[0]["tool_use_id"], body["messages"][1]["content"][1]["id"])
        self.assertEqual(providers.make_provider("anthropic").name, "anthropic")

    def test_openai_retries_without_stream_options(self):
        class Strict(_Handler):
            pass
        seen = []
        orig = _Handler.do_POST

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            if "stream_options" in body:
                self.send_response(400); self.end_headers(); self.wfile.write(b'{"error": "unknown field stream_options"}'); return
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            self.wfile.write(b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\ndata: [DONE]\n\n')
        _Handler.do_POST = do_POST
        try:
            p = providers.OpenAIProvider(host=self.base + "/v1")
            self.assertEqual(p.chat("fake-model", [{"role": "user", "content": "hi"}]).message.content, "ok")
        finally:
            _Handler.do_POST = orig
        self.assertEqual([("stream_options" in b) for b in seen], [True, False])

    def test_retries_then_gives_up(self):
        calls = []

        def flaky():
            calls.append(1)
            raise ConnectionError("connection refused")

        providers.RETRY_BACKOFF = 0
        with self.assertRaises(ConnectionError):
            providers.with_retries(flaky)
        self.assertEqual(len(calls), providers.RETRIES)
        with self.assertRaises(ValueError):
            providers.with_retries(lambda: (_ for _ in ()).throw(ValueError("bad json")))


class EditToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        (self.root / "app.py").write_text("import os\n\ndef add(a, b):\n    return a - b\n\ndef sub(a, b):\n    return a - b\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_multi_edit_one_approval(self):
        calls = []
        t = agent_tools.Tools(self.root, approve=lambda ti, b, kind="text": (calls.append(ti), (True, ""))[1], ui=agent_ui.BaseUI())
        r = t.multi_edit("app.py", [{"old_string": "def add(a, b):\n    return a - b", "new_string": "def add(a, b):\n    return a + b"},
                                    {"old_string": "import os", "new_string": "import os\nimport sys"}])
        self.assertIn("Applied 2 edits", r)
        self.assertEqual(calls, ["Edit app.py (2 edits)"])
        self.assertIn("import sys", (self.root / "app.py").read_text())
        self.assertIn("a + b", (self.root / "app.py").read_text())
        self.assertIn("Error: edit 1", t.multi_edit("app.py", [{"old_string": "nope", "new_string": "x"}]))
        self.assertIn("matches 2", t.multi_edit("app.py", [{"old_string": "return a", "new_string": "x"}]))

    def test_apply_patch_fuzzy_offset(self):
        t = yes_tools(self.root)
        patch = ("--- a/app.py\n+++ b/app.py\n@@ -10,3 +10,3 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n \n")
        r = t.apply_patch(patch)
        self.assertIn("Patched app.py (1 hunks)", r)
        self.assertIn("applied at line 3", r)
        self.assertEqual((self.root / "app.py").read_text().count("a + b"), 1)
        self.assertIn("does not match", t.apply_patch("@@ -1 +1 @@\n-zzz\n+yyy\n", path="app.py"))
        self.assertIn("no '+++ path'", t.apply_patch("@@ -1 +1 @@\n-x\n+y\n"))

    def test_insert_at_line(self):
        t = yes_tools(self.root)
        self.assertIn("at line 1", t.insert_at_line("app.py", 1, "#!/usr/bin/env python3"))
        self.assertTrue((self.root / "app.py").read_text().startswith("#!/usr/bin/env python3\nimport os\n"))
        t.insert_at_line("app.py", 0, "# end")
        self.assertTrue((self.root / "app.py").read_text().endswith("    return a - b\n# end\n"))

    def test_deny_list_and_hooks(self):
        t = yes_tools(self.root, hooks={"after_edit": "wc -l {path}", "after_shell": "echo hooked"})
        self.assertIn("refused by the deny list", t.run_shell("sudo rm -rf /"))
        self.assertIn("refused by the deny list", t.run_shell("git push --force origin main"))
        r = t.edit_file("app.py", "import os", "import os\n")
        self.assertIn("[hook after_edit: `wc -l app.py` exit 0]", r)
        self.assertIn("hooked", t.run_shell("true"))

    def test_git_tree_checkpoint_and_undo(self):
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", "add", "-A"], cwd=self.root, check=True)
        subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "init"], cwd=self.root, check=True)
        (self.root / "app.py").write_text("changed before command\n")
        t = yes_tools(self.root)
        self.assertTrue(t.git)
        t.run_shell("echo clobbered > app.py")
        self.assertEqual((self.root / "app.py").read_text(), "clobbered\n")
        self.assertEqual(t.checkpoints[-1][0], "")
        self.assertIn("Restored tracked files", t.undo())
        self.assertEqual((self.root / "app.py").read_text(), "changed before command\n")

    def test_parallel_dispatch_and_search_parsing(self):
        t = yes_tools(self.root)
        out = t.dispatch_parallel([("read_file", {"path": "app.py"}), ("find_files", {"pattern": "*.py"})])
        self.assertIn("import os", out[0])
        self.assertIn("app.py", out[1])
        page = ('<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fx&amp;rut=1">'
                'Example <b>title</b></a> junk <a class="result__snippet" href="x">A snippet</a>')
        self.assertEqual(agent_tools.parse_ddg(page), "Example title\nhttps://example.com/x\nA snippet")
        self.assertEqual(agent_tools.outside_paths("cat $HOME/.ssh/id_rsa", self.root), ["~/.ssh/id_rsa"])
        argv = agent_tools.sandbox_argv("ls", self.root, platform="linux")
        self.assertEqual(argv[0], "bwrap")
        self.assertIn(str(self.root), argv)
        with self.assertRaises(RuntimeError):
            agent_tools.sandbox_argv("ls", self.root, platform="win32")

    def test_project_config_merges_over_global(self):
        agent_tools.save_config({"allow": ["git status"], "hooks": {"after_edit": "a"}})
        (self.root / ".anyagent").mkdir()
        (self.root / ".anyagent" / "config.json").write_text(json.dumps({"allow": ["pytest"], "deny": ["rm -rf"], "hooks": {"after_shell": "b"}}))
        cfg = agent_tools.load_config(self.root)
        self.assertEqual(cfg["allow"], ["git status", "pytest"])
        self.assertEqual(cfg["deny"], ["rm -rf"])
        self.assertEqual(cfg["hooks"], {"after_edit": "a", "after_shell": "b"})
        self.assertIn("sudo ", agent_tools.load_config()["deny"])


class AgentFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        (self.root / "a.py").write_text("x = 1\n")
        self.ui = agent_ui.BaseUI()

    def tearDown(self):
        self.tmp.cleanup()

    def session(self, client, tools=None):
        tools = tools or yes_tools(self.root)
        return agent.Session(client, tools, self.ui, "m", 4096, agent.build_system(self.root, tools))

    def test_harness_and_memory_in_system_prompt(self):
        tools = yes_tools(self.root)
        system = agent.build_system(self.root, tools)
        self.assertIn("Explore first", system)
        self.assertNotIn("Harness policy", system)
        self.assertIn("Tools available: list_dir", system)
        (self.root / ".anyagent").mkdir()
        (self.root / ".anyagent" / "HARNESS.md").write_text("Custom rules here\n<!-- policy -->\nhidden")
        s = self.session(Scripted())
        self.assertTrue(s.messages[0]["content"].startswith("Custom rules here"))
        self.assertNotIn("hidden", s.messages[0]["content"])
        self.assertIn("Saved to .anyagent/memory.md", s.remember("tests live in tests/"))
        self.assertIn("tests live in tests/", s.messages[0]["content"])
        self.assertIn("# Project memory", (self.root / ".anyagent" / "memory.md").read_text())

    def test_token_meter_calibrates(self):
        m = agent.TokenMeter()
        msgs = [{"role": "user", "content": "x" * 4000}]
        self.assertEqual(m.estimate(msgs), 1000)
        self.assertEqual(m.observe(msgs, [], 100), 1000)      # under-report ignored, estimate wins
        self.assertEqual(m.observe(msgs, [], 2000), 2000)     # plausible report: ratio moves toward 2 chars/token
        self.assertLess(m.ratio, 4.0)
        self.assertGreater(m.estimate(msgs), 1000)

    def test_consecutive_errors_stop_turn(self):
        steps = [("", [("read_file", {"path": f"missing{i}.py"})]) for i in range(20)]
        client = Scripted(steps)
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]
        agent.run_turn(client, yes_tools(self.root), messages, "m", 4096)
        tool_msgs = [m["content"] for m in messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), agent.ERRORS_STOP_AT)
        self.assertIn("in a row have failed", tool_msgs[agent.ERRORS_NUDGE_AT - 1])

    def test_parallel_read_only_calls_in_one_step(self):
        client = Scripted([("", [("read_file", {"path": "a.py"}), ("find_files", {"pattern": "*.py"})])])
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]
        agent.run_turn(client, yes_tools(self.root), messages, "m", 4096)
        self.assertEqual([m["tool_name"] for m in messages if m["role"] == "tool"], ["read_file", "find_files"])
        self.assertIn("x = 1", messages[3]["content"])

    def test_parallel_and_write_subagents(self):
        client = Scripted([("", [("run_subagent", {"tasks": ["t1", "t2"]})]),
                           ("r1", []), ("r2", []),
                           ("", [("run_subagent", {"task": "edit it", "write": True})]),
                           ("", [("write_file", {"path": "b.txt", "content": "b"})]), ("wrote", [])])
        s = self.session(client)
        s.turn("go")
        report = [m for m in s.messages if m["role"] == "tool"][0]["content"]
        self.assertIn("## Task: t1", report)
        self.assertIn("## Task: t2", report)
        self.assertTrue((self.root / "b.txt").exists())
        self.assertEqual([m for m in s.messages if m["role"] == "tool"][1]["content"], "wrote")

    def test_compaction_keeps_latest_exchange(self):
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "old"},
                    {"role": "assistant", "content": "old reply"}, {"role": "user", "content": "latest question"},
                    {"role": "assistant", "content": "latest reply"}]
        agent.compact(Scripted(), messages, "m", 4096)
        self.assertEqual([m["content"] for m in messages[3:]], ["latest question", "latest reply"])
        self.assertIn("scripted", messages[1]["content"])

    def test_sessions_list_resume_and_prune(self):
        agent.KEEP_SESSIONS = 2
        for i in range(3):
            s = self.session(Scripted())
            s.started = f"2026010{i}-000000"
            s.turn(f"task {i}")
        sessions = agent.list_sessions(self.root)
        self.assertEqual([x[2] for x in sessions], ["task 2", "task 1"])
        s = self.session(Scripted())
        self.assertTrue(s.command("/resume 2"))
        self.assertEqual(s.messages[1]["content"], "task 1")
        picked = []
        self.ui.pick = lambda options, title: (picked.append(options), 0)[1]
        s2 = self.session(Scripted())
        s2.command("/resume")
        self.assertEqual(s2.messages[1]["content"], "task 2")
        self.assertIn("task 2", picked[0][0])

    def test_custom_commands_and_stats(self):
        (self.root / ".anyagent" / "commands").mkdir(parents=True)
        (self.root / ".anyagent" / "commands" / "review.md").write_text("Review $ARGUMENTS carefully.")
        client = Scripted()
        s = self.session(client)
        self.assertTrue(s.command("/review a.py"))
        self.assertEqual(client.kwargs[0]["messages"][1]["content"], "Review a.py carefully.")
        self.assertEqual(len(s.stats), 1)
        s.pricing = {"m": {"input": 1_000_000, "output": 0}}
        self.assertEqual(s.cost(), 50.0)
        self.assertTrue(s.command("/stats"))
        self.assertTrue(s.command("/think"))
        self.assertTrue(s.think)
        s.turn("again")
        self.assertIs(client.kwargs[-1]["think"], True)

    def test_git_diff_and_commit(self):
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="a@b", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="a@b")
        subprocess.run(["git", "add", "-A"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=self.root, check=True, env=env)
        os.environ.update({k: v for k, v in env.items() if k.startswith("GIT_")})
        s = self.session(Scripted())
        self.assertEqual(s.diff(), "(no changes to tracked files)")
        (self.root / "a.py").write_text("x = 2\n")
        (self.root / "new.txt").write_text("n\n")
        d = s.diff()
        self.assertIn("a.py", d)
        self.assertIn("New untracked files:\nnew.txt", d)
        self.assertIn("scripted commit", s.commit())
        self.assertEqual(subprocess.run(["git", "log", "--oneline"], cwd=self.root, capture_output=True, text=True).stdout.count("\n"), 2)
        self.assertEqual(s.commit(), "Nothing to commit.")

    def test_headless_json_main(self):
        class FakeProvider(providers.Provider):
            name = "fake"
            supports_images = False

            def __init__(self, host=None):
                self.inner = Scripted([("", [("write_file", {"path": "out.txt", "content": "hi"})])])

            def chat(self, **kw):
                return self.inner.chat(**kw)

            def show(self, model):
                pass

        providers.PROVIDERS["fake"] = FakeProvider
        agent.PROVIDERS["fake"] = FakeProvider
        try:
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                rc = agent.main(["--root", str(self.root), "--provider", "fake", "--model", "m", "--json", "make out.txt"])
            finally:
                sys.stdout = old
        finally:
            del providers.PROVIDERS["fake"]
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(data["result"], "Done.")
        self.assertEqual(data["tools"][0]["tool"], "write_file")
        self.assertEqual((self.root / "out.txt").read_text(), "hi")


class _MCPHttpHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "id" not in msg:
            self.send_response(202); self.end_headers(); return
        m = msg["method"]
        if m == "initialize":
            res = {"protocolVersion": "2025-03-26", "capabilities": {"resources": {}}, "serverInfo": {"name": "h"}}
        elif m == "tools/list":
            res = {"tools": [{"name": "add", "inputSchema": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}}]}
        elif m == "resources/list":
            res = {"resources": [{"uri": "mem://greeting", "name": "greeting"}]}
        elif m == "resources/read":
            res = {"contents": [{"uri": msg["params"]["uri"], "text": "hello resource"}]}
        elif m == "tools/call":
            a = msg["params"]["arguments"]
            res = {"content": [{"type": "text", "text": str(a["a"] + a["b"])}]}
        else:
            res = {}
        body = f"event: message\ndata: {json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': res})}\n\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Mcp-Session-Id", "sess-1")
        self.end_headers()
        self.wfile.write(body)


PING_SERVER = r'''
import json, sys
def send(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
sys.stderr.write("booting\n"); sys.stderr.flush()
for line in sys.stdin:
    msg = json.loads(line)
    m, i = msg.get("method"), msg.get("id")
    if m == "initialize":
        send({"jsonrpc": "2.0", "id": 99, "method": "ping"})   # server-initiated request first
        send({"jsonrpc": "2.0", "id": i, "result": {"protocolVersion": "2025-03-26", "capabilities": {}}})
    elif m == "tools/list":
        send({"jsonrpc": "2.0", "id": i, "result": {"tools": []}})
    elif i == 99:
        sys.stderr.write("pong received\n"); sys.stderr.flush()
'''


class MCPTransportTests(unittest.TestCase):
    def test_http_transport_tools_and_resources(self):
        srv = http.server.HTTPServer(("127.0.0.1", 0), _MCPHttpHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            s = mcp.MCPHttpServer("h", f"http://127.0.0.1:{srv.server_port}/mcp", headers={"X-Key": "1"}).start()
            self.assertEqual(s.session_id, "sess-1")
            self.assertEqual([t["name"] for t in s.tools], ["add"])
            self.assertEqual(s.call_tool("add", {"a": 2, "b": 3}), "5")
            self.assertEqual(s.read_resource("mem://greeting"), "hello resource")
            t = yes_tools(Path(tempfile.mkdtemp()))
            names = t.register_mcp(s)
            self.assertEqual(names, ["mcp__h__add", "mcp__h__read_resource"])
            self.assertEqual(t.dispatch("mcp__h__read_resource", {"uri": "mem://greeting"}), "hello resource")
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / ".mcp.json").write_text(json.dumps({"mcpServers": {"h2": {"url": f"http://127.0.0.1:{srv.server_port}/mcp"}}}))
                started = mcp.load_servers({}, d)
                self.assertEqual(started[0][1:], ("h2", ""))
        finally:
            srv.shutdown()

    def test_stdio_server_requests_and_stderr(self):
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "srv.py"
            script.write_text(PING_SERVER)
            s = mcp.MCPServer("p", sys.executable, [str(script)]).start()
            try:
                import time
                time.sleep(0.3)
                self.assertIn("booting", s.log())
                self.assertIn("pong received", s.log())
            finally:
                s.close()
            bad = mcp.load_servers({"mcp": {"nope": {"command": "/definitely/missing/binary"}}}, d)
            self.assertIsNone(bad[0][0])
            self.assertIn("cannot start", bad[0][2])


class UIFeatureTests(unittest.TestCase):
    def test_thinking_stream_is_shown_not_stored(self):
        console = Console(file=io.StringIO(), width=100, record=True, force_terminal=False)
        ui = agent_ui.UI(root=".", console=console)
        chunks = [NS(message=NS(content="", thinking="let me think", tool_calls=None)),
                  NS(message=NS(content="Answer", thinking="", tool_calls=None))]
        res = agent_ui.StreamResult()
        ui.consume_stream(iter(chunks), res)
        self.assertEqual(res.thinking, "let me think")
        self.assertEqual(res.content, "Answer")
        self.assertEqual(agent_ui.tool_label("apply_patch", {"patch": "--- a/x.py\n+++ b/x.py\n@@"}), ("Patch", "x.py"))
        self.assertEqual(agent_ui.tool_label("multi_edit", {"path": "x.py", "edits": [{}, {}]}), ("Update", "x.py (2 edits)"))
        self.assertEqual(agent_ui.summarize_result("web_search", "T\nhttp://a\ns\n\nT2\nhttp://b\ns")[0], "Found 2 results")
        ui.status = {"model": "m", "ctx": 100, "used": 50, "tps": 12.3, "cost": 0.5, "think": True}
        bar = ui._toolbar()
        self.assertIn("12 tok/s", bar)
        self.assertIn("$0.500", bar)
        self.assertIn("think", bar)
        h = agent_ui.HeadlessUI()
        self.assertEqual(h.approve("x", "y"), (True, ""))


if __name__ == "__main__":
    unittest.main()
