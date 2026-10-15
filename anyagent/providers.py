"""Model providers. A provider exposes an Ollama-shaped streaming `chat()` so the agent loop
and UI never depend on a specific backend. Add a backend by subclassing Provider and
registering it in PROVIDERS.

Stream chunks have `.message.content`, `.message.thinking`, `.message.tool_calls` (each with
`.function.name` / `.function.arguments`) and, on the last chunk, `.prompt_eval_count` and
`.eval_count`.

History format (shared by every provider) is Ollama's:
    {"role": "user"|"assistant"|"system"|"tool", "content": str, "images"?: [base64],
     "tool_calls"?: [{"function": {"name", "arguments"}}], "tool_name"?: str}
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from types import SimpleNamespace as NS

RETRIES = 3            # connection-level retries per model call
RETRY_BACKOFF = 1.0    # seconds, doubled each retry


def chunk(content="", tool_calls=None, thinking="", prompt_tokens=None, eval_tokens=None):
    return NS(message=NS(content=content, thinking=thinking, tool_calls=tool_calls or None),
              prompt_eval_count=prompt_tokens, eval_count=eval_tokens)


def tool_call(name, args):
    return NS(function=NS(name=name, arguments=args))


class ProviderError(Exception):
    pass


class Provider:
    name = "base"
    supports_images = False

    def chat(self, model, messages, tools=None, stream=False, options=None, keep_alive=None, think=None):
        raise NotImplementedError

    def show(self, model):          # raise if the model is unavailable
        raise NotImplementedError

    def list_models(self) -> list:
        return []

    def context_length(self, model):
        """The model's maximum context, or None when unknown."""
        return None

    def version(self) -> str:
        return ""

    def check(self) -> str:
        """Startup warning text, or '' when everything looks fine."""
        return ""


def with_retries(fn, ui=None):
    """Call fn(); on connection errors retry with backoff. Never swallows KeyboardInterrupt."""
    delay = RETRY_BACKOFF
    for attempt in range(RETRIES):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 - backend-specific exception types
            if attempt == RETRIES - 1 or not _retryable(e):
                raise
            if ui:
                ui.warn(f"Model server error ({e}); retrying in {delay:.0f}s…")
            time.sleep(delay)
            delay *= 2


def _retryable(e: Exception) -> bool:
    text = f"{type(e).__name__}: {e}".lower()
    return any(k in text for k in ("connect", "timeout", "timed out", "reset", "503", "502", "429", "overloaded",
                                   "temporarily", "unavailable"))


# ------------------------------------------------------------------ Ollama
class OllamaProvider(Provider):
    name = "ollama"
    supports_images = True
    MIN_VERSION = (0, 11, 0)  # qwen3-coder's native tool-call parser landed around here

    def __init__(self, host=None):
        import ollama
        self.client = ollama.Client(host=host) if host else ollama.Client()

    def chat(self, model, messages, tools=None, stream=False, options=None, keep_alive="30m", think=None):
        kw = {}
        if think is not None:
            kw["think"] = think
        return self.client.chat(model=model, messages=messages, tools=tools or None, stream=stream,
                                options=options or {}, keep_alive=keep_alive, **kw)

    def show(self, model):
        return self.client.show(model)

    def list_models(self):
        return [m.model for m in self.client.list().models]

    def context_length(self, model):
        try:
            info = self.client.show(model).modelinfo or {}
        except Exception:
            return None
        for k, v in info.items():
            if k.endswith(".context_length"):
                return int(v)
        return None

    def version(self) -> str:
        try:
            return str(self.client._client.get("/api/version").json().get("version", ""))
        except Exception:
            return ""

    def check(self) -> str:
        v = self.version()
        if not v:
            return ""
        try:
            parts = tuple(int(x) for x in v.split("-")[0].split(".")[:3])
        except ValueError:
            return ""
        if parts < self.MIN_VERSION:
            return (f"Ollama {v} is older than {'.'.join(map(str, self.MIN_VERSION))}: some models return tool "
                    "calls as plain text. The agent parses those, but upgrading Ollama is more reliable.")
        return ""

    def web_search(self, query: str, max_results: int = 5):
        """Ollama's hosted search (needs OLLAMA_API_KEY). Returns [(title, url, snippet)] or raises."""
        r = self.client.web_search(query, max_results=max_results)
        return [(x.title, x.url, x.content) for x in r.results]


# ------------------------------------------------------------- HTTP helpers
def _post_sse(url, payload, headers, timeout=600):
    """POST JSON; yield parsed `data:` JSON objects from an SSE response (or the whole JSON body)."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:500]
        raise ProviderError(f"HTTP {e.code} from {url}: {body}")
    except urllib.error.URLError as e:
        raise ProviderError(f"cannot connect to {url}: {e.reason}")
    ctype = resp.headers.get("Content-Type", "")
    if "text/event-stream" not in ctype:
        yield json.loads(resp.read().decode("utf-8", "replace"))
        return
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except ValueError:
                continue


def _get_json(url, headers, timeout=30):
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        raise ProviderError(f"HTTP {e.code} from {url}: {e.read().decode('utf-8', 'replace')[:300]}")
    except urllib.error.URLError as e:
        raise ProviderError(f"cannot connect to {url}: {e.reason}")


def _call_ids(messages):
    """Positional tool-call ids for providers that need them (our history stores none)."""
    ids, pending = {}, []
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            pending = [f"call_{i}_{j}" for j in range(len(m["tool_calls"]))]
            ids[i] = list(pending)
        elif m.get("role") == "tool":
            ids[i] = pending.pop(0) if pending else f"call_{i}"
    return ids


# ------------------------------------------------------- OpenAI-compatible
class OpenAIProvider(Provider):
    """Any /v1/chat/completions server: LM Studio, vLLM, llama.cpp, OpenRouter, OpenAI itself."""
    name = "openai"
    supports_images = True

    def __init__(self, host=None, api_key=None):
        self.base = (host or os.environ.get("OPENAI_BASE_URL") or "http://localhost:1234/v1").rstrip("/")
        self.key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def _headers(self):
        return {"Authorization": f"Bearer {self.key}"} if self.key else {}

    def _convert(self, messages):
        ids = _call_ids(messages)
        out = []
        for i, m in enumerate(messages):
            role = m.get("role")
            if role == "tool":
                out.append({"role": "tool", "tool_call_id": ids[i], "content": str(m.get("content", ""))})
            elif role == "assistant" and m.get("tool_calls"):
                out.append({"role": "assistant", "content": m.get("content") or None, "tool_calls": [
                    {"id": cid, "type": "function", "function": {
                        "name": tc["function"]["name"], "arguments": json.dumps(tc["function"]["arguments"])}}
                    for cid, tc in zip(ids[i], m["tool_calls"])]})
            elif role == "user" and m.get("images"):
                out.append({"role": "user", "content": [{"type": "text", "text": m.get("content", "")}] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}} for b in m["images"]]})
            else:
                out.append({"role": role, "content": m.get("content", "")})
        return out

    def chat(self, model, messages, tools=None, stream=False, options=None, keep_alive=None, think=None):
        options = options or {}
        payload = {"model": model, "messages": self._convert(messages), "stream": True,
                   "stream_options": {"include_usage": True}}
        if tools:
            payload["tools"] = tools
        if "temperature" in options:
            payload["temperature"] = options["temperature"]
        if think is not None and os.environ.get("OPENAI_REASONING_EFFORT"):
            payload["reasoning_effort"] = os.environ["OPENAI_REASONING_EFFORT"]
        gen = self._stream(payload)
        if stream:
            return gen
        content, calls, pt, ct = "", [], None, None
        for c in gen:
            content += c.message.content or ""
            calls += c.message.tool_calls or []
            pt, ct = c.prompt_eval_count or pt, c.eval_count or ct
        return chunk(content, calls, prompt_tokens=pt, eval_tokens=ct)

    def _stream(self, payload):
        pending: dict = {}  # index -> {name, args}
        usage = None
        events = _post_sse(f"{self.base}/chat/completions", payload, self._headers())
        try:
            first = next(events)
        except StopIteration:
            first = None
        except ProviderError as e:
            if "stream_options" not in str(e) or "stream_options" not in payload:
                raise
            payload = {k: v for k, v in payload.items() if k != "stream_options"}  # server predates the field
            events = _post_sse(f"{self.base}/chat/completions", payload, self._headers())
            first = next(events, None)
        for ev in ([first] if first is not None else []) + list(events):
            if ev.get("usage"):
                usage = ev["usage"]
            for choice in ev.get("choices", []):
                delta = choice.get("delta") or choice.get("message") or {}
                if delta.get("content"):
                    yield chunk(delta["content"])
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    yield chunk(thinking=delta.get("reasoning_content") or delta.get("reasoning"))
                for tc in delta.get("tool_calls") or []:
                    slot = pending.setdefault(tc.get("index", len(pending)), {"name": "", "args": ""})
                    fn = tc.get("function") or {}
                    slot["name"] = fn.get("name") or slot["name"]
                    slot["args"] += fn.get("arguments") or ""
        calls = []
        for _, slot in sorted(pending.items()):
            try:
                args = json.loads(slot["args"] or "{}")
            except ValueError:
                args = {"_raw": slot["args"]}
            calls.append(tool_call(slot["name"], args if isinstance(args, dict) else {}))
        yield chunk("", calls, prompt_tokens=(usage or {}).get("prompt_tokens"),
                    eval_tokens=(usage or {}).get("completion_tokens"))

    def show(self, model):
        models = self.list_models()
        if models and model not in models:
            raise ProviderError(f"model {model!r} not served by {self.base}; available: {', '.join(models[:10])}")

    def list_models(self):
        try:
            return [m["id"] for m in _get_json(f"{self.base}/models", self._headers()).get("data", [])]
        except ProviderError:
            return []

    def context_length(self, model):
        try:
            for m in _get_json(f"{self.base}/models", self._headers()).get("data", []):
                if m.get("id") == model:
                    return m.get("context_length") or m.get("max_model_len") or m.get("context_window")
        except ProviderError:
            pass
        return None


# --------------------------------------------------------------- Anthropic
class AnthropicProvider(Provider):
    name = "anthropic"
    supports_images = True
    VERSION = "2023-06-01"

    def __init__(self, host=None, api_key=None):
        self.base = (host or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def _headers(self):
        return {"x-api-key": self.key, "anthropic-version": self.VERSION}

    def _convert(self, messages):
        ids = _call_ids(messages)
        system, out = "", []
        for i, m in enumerate(messages):
            role = m.get("role")
            if role == "system":
                system += m.get("content", "")
                continue
            if role == "tool":
                block = {"type": "tool_result", "tool_use_id": ids[i], "content": str(m.get("content", ""))}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                        and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
                continue
            if role == "assistant":
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                for cid, tc in zip(ids.get(i, []), m.get("tool_calls") or []):
                    blocks.append({"type": "tool_use", "id": cid, "name": tc["function"]["name"],
                                   "input": tc["function"]["arguments"]})
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(empty)"}]})
                continue
            blocks = [{"type": "text", "text": m.get("content") or "(empty)"}]
            for b in m.get("images") or []:
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b}})
            out.append({"role": "user", "content": blocks})
        return system, out

    def chat(self, model, messages, tools=None, stream=False, options=None, keep_alive=None, think=None):
        options = options or {}
        system, converted = self._convert(messages)
        payload = {"model": model, "messages": converted, "stream": True,
                   "max_tokens": int(os.environ.get("ANTHROPIC_MAX_TOKENS", "8192"))}
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [{"name": t["function"]["name"], "description": t["function"]["description"],
                                 "input_schema": t["function"]["parameters"]} for t in tools]
        if "temperature" in options:
            payload["temperature"] = options["temperature"]
        if think and not tools:
            # shortcut: with tools, Anthropic requires the signed thinking blocks back in every assistant
            # turn of the tool loop, which this history does not keep; upgrade when history stores raw blocks
            payload["thinking"] = {"type": "enabled", "budget_tokens": 4096}
            payload.pop("temperature", None)  # thinking forbids temperature overrides
        gen = self._stream(payload)
        if stream:
            return gen
        content, calls, pt, ct = "", [], None, None
        for c in gen:
            content += c.message.content or ""
            calls += c.message.tool_calls or []
            pt, ct = c.prompt_eval_count or pt, c.eval_count or ct
        return chunk(content, calls, prompt_tokens=pt, eval_tokens=ct)

    def _stream(self, payload):
        blocks: dict = {}
        pt = ct = None
        for ev in _post_sse(f"{self.base}/v1/messages", payload, self._headers()):
            t = ev.get("type")
            if t == "message_start":
                pt = (ev.get("message", {}).get("usage") or {}).get("input_tokens")
            elif t == "content_block_start":
                cb = ev.get("content_block", {})
                if cb.get("type") == "tool_use":
                    blocks[ev["index"]] = {"name": cb.get("name"), "json": ""}
            elif t == "content_block_delta":
                d = ev.get("delta", {})
                if d.get("type") == "text_delta":
                    yield chunk(d.get("text", ""))
                elif d.get("type") == "thinking_delta":
                    yield chunk(thinking=d.get("thinking", ""))
                elif d.get("type") == "input_json_delta" and ev["index"] in blocks:
                    blocks[ev["index"]]["json"] += d.get("partial_json", "")
            elif t == "message_delta":
                ct = (ev.get("usage") or {}).get("output_tokens", ct)
            elif t == "error":
                raise ProviderError(str(ev.get("error")))
        calls = []
        for _, b in sorted(blocks.items()):
            try:
                args = json.loads(b["json"] or "{}")
            except ValueError:
                args = {}
            calls.append(tool_call(b["name"], args if isinstance(args, dict) else {}))
        yield chunk("", calls, prompt_tokens=pt, eval_tokens=ct)

    def show(self, model):
        if not self.key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")

    def list_models(self):
        try:
            return [m["id"] for m in _get_json(f"{self.base}/v1/models", self._headers()).get("data", [])]
        except ProviderError:
            return []

    def context_length(self, model):
        return 200_000


PROVIDERS = {"ollama": OllamaProvider, "openai": OpenAIProvider, "anthropic": AnthropicProvider}


def make_provider(name: str, host=None) -> Provider:
    try:
        cls = PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unknown provider {name!r}; known: {', '.join(PROVIDERS)}")
    if name == "ollama":
        return cls(host=host or os.environ.get("OLLAMA_HOST"))
    return cls(host=host)


__all__ = ["Provider", "OllamaProvider", "OpenAIProvider", "AnthropicProvider", "PROVIDERS", "make_provider",
           "ProviderError", "with_retries", "chunk", "tool_call", "NS"]
