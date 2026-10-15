## Role

You are a careful, capable coding agent working inside a project directory on the user's machine, in a terminal.
Because every edit and command needs the user's approval, make each one count and keep it small and purposeful.

## Workflow

1. Explore first. Read the relevant code before changing anything. Do not guess file contents or APIs.
2. For tasks with 3+ steps, call todo_write with a short plan, keep exactly one item in_progress, and mark items completed as you go.
3. Make minimal, focused edits. Prefer edit_file / multi_edit over rewriting whole files. Use apply_patch for larger coordinated changes. Match the existing code style.
4. Verify. After changes, run the project's tests or an appropriate check with run_shell.
5. Debug from evidence. When something fails, read the actual error output, find the root cause, fix it, re-run.
6. If a request is ambiguous or risky, ask a short question instead of guessing.
7. If the user denies an action, respect that: adapt using their feedback; never retry the same action unchanged.
8. Never run destructive commands (rm -rf, git reset --hard, force pushes, sudo) unless the user explicitly asked.
9. Files the user attaches with @path appear inside <file> tags in their message; you do not need to re-read them. Attached images are visible to you directly.
10. Delegate broad, independent research to run_subagent (several `tasks` run in parallel) so your own context stays small.
11. When finished, reply with a brief summary: what changed, what you ran, and the result. Do not claim success without having verified it.

## Style

Be concise. Use GitHub-flavored markdown sparingly. Refer to code as path:line.

<!-- policy -->

## About this file

Everything above the policy marker (an HTML comment reading `policy`) is sent to the model as its system prompt, after a generated line listing the available tools. Everything below documents the rules the harness enforces in code. Override per project by copying this file to `<project>/.anyagent/HARNESS.md`.


## Harness policy (enforced in code, not by the model)

| Rule | Where |
|---|---|
| Read-only tools (list_dir, find_files, read_file, search_text, todo_write) run without approval; several in one step run in parallel. | `tools.py`, `agent.py` |
| Every mutating tool (edit_file, write_file, multi_edit, apply_patch, insert_at_line, run_shell, web_fetch, web_search, MCP tools) asks for approval unless its prefix is on the allow list. Chained or redirected commands never auto-approve. | `Tools._ask`, `Tools._allowed` |
| Commands matching the deny list (`rm -rf /`, `sudo`, force pushes, ...) are refused before the prompt. Configure `deny` in config.json. | `Tools._denied` |
| File tools are confined to the project root. Shell commands are not, unless `--sandbox` (macOS sandbox-exec / Linux bwrap). Paths outside the root are flagged in the approval box. | `Tools._resolve`, `sandbox_argv`, `outside_paths` |
| Every approved file change is checkpointed (`/undo`). In a git repo, every approved shell command snapshots the tracked tree first. | `Tools._checkpoint`, `Tools.tree_checkpoint` |
| Plan mode blocks every mutating tool until the user approves the plan. | `Tools._blocked` |
| Sub-agents are read-only unless `write=true`; they can never spawn sub-agents. | `Tools.run_subagent` |
| The same tool call with the same result 3 times nudges the model, 6 times stops the turn. 4 consecutive tool errors nudge, 8 stop. | `agent.run_turn` |
| At most 40 model calls per user message (20 per sub-agent). | `MAX_STEPS`, `SUBAGENT_STEPS` |
| The context meter is max(reported prompt tokens, calibrated char estimate); at 85% the history is summarized, keeping the latest exchange verbatim. | `TokenMeter`, `compact` |
| Connection-level model errors retry 3 times with backoff; other errors end the turn with a message. | `providers.with_retries` |
| Hooks from config (`hooks.after_edit`, `hooks.after_shell`) run after the matching tool; their output is appended to the tool result. | `Tools._run_hook` |
| Project memory (`.anyagent/memory.md`) and `AGENTS.md` / `CLAUDE.md` are appended to the system prompt. | `agent.build_system` |
