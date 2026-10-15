"""End-to-end test against a real model server. Skipped unless ANYAGENT_INTEGRATION=1.

    ANYAGENT_INTEGRATION=1 AGENT_MODEL=qwen3-coder:30b python -m unittest -v tests/test_integration.py
"""
import os
import tempfile
import unittest
from pathlib import Path

import anyagent.agent as agent
import anyagent.tools as agent_tools
import anyagent.ui as agent_ui
from anyagent.providers import make_provider


class YesUI(agent_ui.BaseUI):
    def approve(self, title, body, kind="text"):
        return True, ""


@unittest.skipUnless(os.environ.get("ANYAGENT_INTEGRATION"), "set ANYAGENT_INTEGRATION=1 to run against a live model server")
class RealModelTests(unittest.TestCase):
    model = os.environ.get("AGENT_MODEL", agent.DEFAULT_MODEL)

    def setUp(self):
        self.provider = make_provider(os.environ.get("AGENT_PROVIDER", "ollama"))
        self.provider.show(self.model)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        agent_tools.CONFIG_DIR = self.root / ".home"
        agent.SESSIONS_DIR = agent_tools.CONFIG_DIR / "sessions"
        self.ui = YesUI()
        self.tools = agent_tools.Tools(self.root, ui=self.ui)

    def tearDown(self):
        self.tmp.cleanup()

    def session(self):
        system = agent.SYSTEM_PROMPT + f"\nWorkspace root: {self.root}"
        return agent.Session(self.provider, self.tools, self.ui, self.model, 16384, system)

    def test_model_uses_tools_to_read_a_file(self):
        (self.root / "notes.txt").write_text("The secret word is PINEAPPLE42.\n")
        s = self.session()
        s.turn("Use read_file on notes.txt and tell me the secret word. Reply with just the word.")
        tool_names = [m["tool_name"] for m in s.messages if m["role"] == "tool"]
        self.assertIn("read_file", tool_names, "the model never called read_file (structured or text fallback)")
        self.assertIn("PINEAPPLE42", s.messages[-1]["content"])
        self.assertIsNotNone(s.prompt_tokens)

    def test_model_edits_a_file(self):
        (self.root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        s = self.session()
        s.turn("calc.py has a bug in add(): it subtracts. Fix it with edit_file so it returns a + b. Do not run anything.")
        self.assertIn("a + b", (self.root / "calc.py").read_text())
        self.assertEqual(len(self.tools.checkpoints), 1)


if __name__ == "__main__":
    unittest.main()
