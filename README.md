# AnyAgent: a Claude Code-style terminal agent for local models

A provider-agnostic Python coding agent with a Claude Code-like terminal experience. It talks to Ollama
(default model `qwen3-coder:30b`), any OpenAI-compatible server (LM Studio, vLLM, llama.cpp, OpenRouter,
OpenAI) and Anthropic. The rules it runs under live in one file: `anyagent/HARNESS.md`.

- Streamed markdown replies, `● Read(app.py)` / `⎿ Read 42 lines` tool lines, a todo list
- Boxed diff previews with line numbers, an arrow-key approval menu, "view full diff", batch approval
- Slash commands, `@file` attachments with Tab completion, input history, multi-line input
- A status bar with model and context usage; automatic compaction before the context fills up
- **Approval before every edit and every shell command.** Opt-in allow list for harmless prefixes
- Checkpoint before every file change (`/undo`), git tree snapshots before commands, `/diff`, `/commit`
- Sessions: `any --continue`, `/resume` picker, markdown transcripts; project memory with `/remember`
- Plan mode, parallel sub-agents, thinking mode, patch/multi-edit tools, web search and fetch, images
- MCP servers (stdio and HTTP), hooks, custom slash commands, per-project config, deny list
- Background jobs, streamed command output, optional sandbox (macOS `sandbox-exec`, Linux `bwrap`)
- Headless mode for scripts and CI: `any --yes --sandbox --json "task"`

## Install (macOS/Linux, Python 3.9+)

```bash
ollama list                        # model server running, model pulled (ollama pull qwen3-coder:30b)

pipx install .                     # from this directory; gives you the `any` (alias `anyagent`) command
# or, for development:
python3 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e .

python -m unittest tests/test_agent.py                 # offline tests
ANYAGENT_INTEGRATION=1 python -m unittest tests/test_integration.py   # against the real model server

cd ~/path/to/your/project
any
```

One-shot: `any "run the tests and fix any failures"`. Resume: `any --continue`. Also `python -m anyagent`.

## Providers: how to start and test each one

Every feature works the same on every provider; only the transport differs. The offline test suite exercises
each provider against a fake server. To prove a real endpoint, run the integration test with the matching
environment, then start the agent with the same variables.

### Ollama (default)

```bash
ollama serve                                   # or the menu-bar app
ollama pull qwen3-coder:30b
any                                            # same as: any --provider ollama --model qwen3-coder:30b
ANYAGENT_INTEGRATION=1 python -m unittest -v tests/test_integration.py
```

`OLLAMA_HOST=http://other-box:11434` points at a remote server. `--think` works on thinking models (qwen3,
deepseek-r1). `web_search` uses DuckDuckGo unless `OLLAMA_API_KEY` is set, then Ollama's hosted search.

### OpenAI-compatible: LM Studio, vLLM, llama.cpp server, OpenRouter, OpenAI

```bash
# LM Studio (Local Server tab, port 1234) / vLLM / llama.cpp --server
any --provider openai --host http://localhost:1234/v1 --model qwen2.5-coder-32b-instruct
ANYAGENT_INTEGRATION=1 AGENT_PROVIDER=openai OPENAI_BASE_URL=http://localhost:1234/v1 \
  AGENT_MODEL=qwen2.5-coder-32b-instruct python -m unittest -v tests/test_integration.py

# OpenRouter
export OPENAI_API_KEY=sk-or-...
any --provider openai --host https://openrouter.ai/api/v1 --model qwen/qwen3-coder
ANYAGENT_INTEGRATION=1 AGENT_PROVIDER=openai OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
  AGENT_MODEL=qwen/qwen3-coder python -m unittest -v tests/test_integration.py

# OpenAI
export OPENAI_API_KEY=sk-...
any --provider openai --host https://api.openai.com/v1 --model gpt-5
```

`--model` must be an id the server lists under `/v1/models` (`/model` with Tab shows them). The server's
`context_length` (OpenRouter reports it) sets the compaction threshold; otherwise 32768 is assumed.
Servers that reject `stream_options` are retried without it automatically.

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
any --provider anthropic --model claude-sonnet-5-5
ANYAGENT_INTEGRATION=1 AGENT_PROVIDER=anthropic AGENT_MODEL=claude-sonnet-5-5 \
  python -m unittest -v tests/test_integration.py
```

`--think` is honored only on turns without tools (the API requires signed thinking blocks to be echoed back
inside tool loops, which this history does not keep). Add `"pricing"` to config to see cost in `/stats`.

### What the integration test checks

1. The model calls `read_file` (structured call or the text fallback) and reports a secret word from a file.
2. The model fixes a bug with `edit_file`, and the edit is checkpointed.

It approves every action itself, runs in a temporary directory, and skips unless `ANYAGENT_INTEGRATION=1`.
Set `AGENT_PROVIDER`, `AGENT_MODEL` and the provider's key or base URL as above; `--host` maps to
`OLLAMA_HOST`, `OPENAI_BASE_URL` or `ANTHROPIC_BASE_URL`.

## Using it

| Input | What it does |
|---|---|
| `Enter` | send the message |
| `\` then `Enter`, or `Esc` `Enter`, or `Ctrl-J` | newline (pasting multi-line text also works) |
| `@src/app.py` | attach a file's contents (Tab completes paths) |
| `/` + Tab | complete slash commands |
| `Ctrl-C` while the agent works | interrupt it, then tell it what to do next |
| `Ctrl-C` twice or `Ctrl-D` at the prompt | exit |

| Command | Purpose |
|---|---|
| `/plan` | plan mode: the agent is read-only until you approve its plan at the end of its turn |
| `/undo` | revert the most recent approved edit/write; in git, also the tracked-tree snapshot taken before the last command |
| `/diff`, `/commit [msg]` | changes since the session started; stage all and commit (message written by the model if omitted) |
| `/remember <fact>` | append to `.anyagent/memory.md`, loaded into every session for this project |
| `/resume [n]` | pick one of the last saved sessions for this directory |
| `/think` | toggle the model's thinking mode (reasoning streams dim, is not kept in history) |
| `/stats` | model calls, tokens, tok/s, and cost when `pricing` is configured |
| `/allow` | show auto-approved command prefixes; `/allow git status` adds, `/allow rm git status` removes |
| `/jobs` | list background commands; `/jobs kill 2` stops one |
| `/mcp` | list tools from connected MCP servers |
| `/compact [focus]`, `/context`, `/model [name]`, `/clear`, `/init`, `/help`, `/exit` | as named |
| `/<name> args` | custom command from `.anyagent/commands/<name>.md` (`$ARGUMENTS` is substituted) |

## Approvals

Every edit shows a diff, every command shows the exact text, directory and any paths outside the project:

```
╭─ Edit app.py ──────────────────────────────╮
│ Added 1 line, removed 1 line               │
│     2 -     return a - b                   │
│     2 +     return a + b                   │
╰────────────────────────────────────────────╯
 Do you want to proceed?
 ❯ 1. Yes
   2. No
   3. No, and tell the agent what to do differently
   4. View full diff                      (long diffs)  /  Yes, and always allow 'git status'  (commands)
```

Use ↑/↓ + Enter, or press `1`-`4` (`y`/`n` also work). `Esc` rejects. Option 3 sends your sentence to the agent.
When the agent proposes several file changes in one step you get one combined prompt: apply all, review one
by one, or reject all. The allow list only ever covers simple commands: anything with `;`, `&&`, `|`,
redirects or `$(...)` still asks.

## Options

| Flag / env | Default | Purpose |
|---|---|---|
| `--root` | current dir | Directory the file tools can touch and where commands run |
| `--provider` / `AGENT_PROVIDER` | `ollama` | `ollama`, `openai` (any /v1 server) or `anthropic` |
| `--model` / `AGENT_MODEL` | `qwen3-coder:30b` | Model (`/model` switches at runtime, Tab completes) |
| `--num-ctx` / `AGENT_NUM_CTX` | min(model's, `32768`) | Context window. Lower to `16384` if memory is tight |
| `--think` | off | Thinking mode (`/think` toggles) |
| `--temperature` / `AGENT_TEMPERATURE` | model default | Sampling temperature. Try `0.2` vs the default on your own tasks |
| `--host` / `OLLAMA_HOST`, `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL` | per provider | Model server URL |
| `--continue` | off | Resume the last session for this directory |
| `--sandbox` | off | Run commands under `sandbox-exec` (macOS) or `bwrap` (Linux); writes limited to the project and temp dirs |
| `--yes` | off | Approve everything without asking. Pair with `--sandbox` |
| `--json` | off | One-shot headless run; prints `{result, usage, tools}` as JSON (implies `--yes`) |
| `--shell-timeout` / `AGENT_SHELL_TIMEOUT` | `120` | Seconds before a command is killed (the model can pass a per-call `timeout` or `background=true`) |
| `--allow PREFIX` | none | Auto-approve commands starting with PREFIX (repeatable; `/allow` persists to config) |

State lives in `~/.anyagent/` (override with `AGENT_HOME`): `config.json`, `sessions/<project>/` (JSON +
markdown transcript per session, last 20 kept), `jobs/` (background logs), `history`. A project can add
`.anyagent/config.json` (merged over the global one), `.anyagent/HARNESS.md`, `.anyagent/memory.md` and
`.anyagent/commands/*.md`.

```json
{
  "allow": ["git status", "pytest"],
  "deny": ["rm -rf /", "sudo ", "git push --force"],
  "hooks": {"after_edit": "ruff check {path}", "after_shell": ""},
  "pricing": {"claude-sonnet-5-5": {"input": 3, "output": 15}},
  "mcp": {"remote": {"url": "http://localhost:8000/mcp", "headers": {"Authorization": "Bearer x"}}}
}
```

Hook output is appended to the tool result the model sees. Prices are dollars per million tokens.

## MCP servers

Add stdio servers to `<project>/.mcp.json` or `~/.anyagent/config.json` under `"mcp"`:

```json
{"mcpServers": {"fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}}}
```

HTTP servers use `{"url": ..., "headers": {...}}`. Their tools appear to the model as `mcp__<server>__<tool>`
(plus `mcp__<server>__read_resource` when the server exposes resources) and ask for approval like any other
mutating tool. `/mcp` shows each server's tools and recent stderr.

## Files and encodings

Edits keep the file's encoding (UTF-8, UTF-8 BOM, UTF-16, cp1252/latin-1) and line endings (CRLF stays
CRLF). `edit_file` first matches exactly, then ignoring trailing whitespace, then ignoring indentation
(re-indenting the replacement); on a miss it reports the closest lines so the model can correct itself.
`multi_edit` applies several replacements to one file under one approval, `apply_patch` applies a unified
diff with fuzzy hunk placement, `insert_at_line` inserts without needing a match. `@image.png` attaches an
image when the provider supports vision.

## Memory notes

The 30B model at 4-bit is roughly 19 GB, plus KV cache that grows with `num_ctx`. On a 32 GB Mac 32768 is
usually fine; on less, use `--num-ctx 16384`. The context meter uses the larger of Ollama's reported prompt
size and a character-based estimate, because Ollama can under-report cached prompt tokens; at 85% the agent
summarizes the conversation automatically (`/compact` does it on demand).

## Safety notes

- Without `--sandbox`, approved shell commands run with your user permissions and are **not confined** to the
  project. Paths outside the project are flagged in the approval box; read each command before approving.
- Commands stream their output, get no stdin, and are killed (including child processes) on timeout or Ctrl-C.
- `/undo` only covers the agent's own edits/writes, not what a shell command changed. Commit to git before a
  session so `git diff` / `git checkout .` can undo anything.
- If the same tool call fails the same way three times the model is told to change approach; six times stops the
  turn. Four consecutive tool errors of any kind nudge, eight stop.
- The deny list refuses `sudo`, `rm -rf /`, force pushes and the like before the approval prompt, even with `--yes`.
- `--yes` removes the only guard between the model and your shell. Use it with `--sandbox`, in a container, or in CI.

## Layout

| File | Role |
|---|---|
| `anyagent/HARNESS.md` | the rules: model-facing system prompt above the marker, code-enforced policy table below |
| `anyagent/agent.py` | entry point (`any`), model/tool loop, text tool-call fallback parser, loop and error detection, token meter, slash commands, plan mode, sub-agents, compaction, sessions, git, memory, headless mode |
| `anyagent/tools.py` | tools and their schemas; encoding/newline-safe edits, fuzzy matching and patches, checkpoints, allow/deny lists, hooks, shell (streamed, timeout, background, sandbox), web search/fetch, MCP dispatch |
| `anyagent/ui.py` | terminal UI (rich + prompt_toolkit): approvals, batch approvals, live output, thinking stream, headless UI |
| `anyagent/providers.py` | provider interface; Ollama, OpenAI-compatible and Anthropic backends; retries |
| `anyagent/mcp.py` | MCP client: stdio and streamable HTTP, tools and resources |
| `tests/test_agent.py`, `tests/test_features.py` | offline tests (fake model servers, fake MCP servers) |
| `tests/test_integration.py` | real model server test, opt-in with `ANYAGENT_INTEGRATION=1` |
| `.github/workflows/` | CI on push (Ubuntu + macOS, Python 3.9 and 3.12); PyPI publish on `v*` tags via trusted publishing |
| `packaging/homebrew/anyagent.rb` | Homebrew formula template for a tap |
