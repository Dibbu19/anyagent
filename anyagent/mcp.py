"""Minimal MCP (Model Context Protocol) client: stdio (JSON-RPC, newline-delimited) and
streamable HTTP (POST JSON-RPC; JSON or SSE replies).

Configure servers in ~/.anyagent/config.json ("mcp") or <root>/.mcp.json ("mcpServers"):
    {"mcpServers": {"fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
                    "remote": {"url": "http://localhost:8000/mcp", "headers": {"Authorization": "Bearer x"}}}}
"""
from __future__ import annotations

import collections
import json
import os
import queue
import subprocess
import threading
import urllib.error
import urllib.request

PROTOCOL_VERSION = "2025-03-26"


class MCPError(Exception):
    pass


def _content_text(result: dict) -> str:
    parts = []
    for c in result.get("content", []) or result.get("contents", []):
        if c.get("type") == "text" or "text" in c:
            parts.append(c.get("text", ""))
        elif "blob" in c:
            parts.append(f"[binary {c.get('mimeType', '')} content omitted]")
        else:
            parts.append(f"[{c.get('type')} content omitted]")
    text = "\n".join(parts) or json.dumps(result)
    return f"Error: {text}" if result.get("isError") else text


class _Base:
    name: str
    timeout: float
    tools: list
    resources: list
    stderr: collections.deque

    def request(self, method, params=None):
        raise NotImplementedError

    def notify(self, method, params=None):
        raise NotImplementedError

    def start(self):
        init = self.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                           "clientInfo": {"name": "anyagent", "version": "0.3"}})
        self.notify("notifications/initialized")
        caps = (init or {}).get("capabilities", {})
        self.tools = list((self.request("tools/list") or {}).get("tools", []))
        self.resources = []
        if "resources" in caps:
            try:
                self.resources = list((self.request("resources/list") or {}).get("resources", []))
            except MCPError:
                pass
        return self

    def call_tool(self, name: str, arguments: dict) -> str:
        try:
            return _content_text(self.request("tools/call", {"name": name, "arguments": arguments or {}}) or {})
        except MCPError as e:
            return f"Error: {e}"

    def read_resource(self, uri: str) -> str:
        try:
            return _content_text(self.request("resources/read", {"uri": uri}) or {})
        except MCPError as e:
            return f"Error: {e}"

    def log(self) -> str:
        return "\n".join(self.stderr)

    def _handle_server_request(self, msg: dict):
        """Reply to server-initiated requests: ping gets {}, anything else 'not supported'."""
        reply = {"jsonrpc": "2.0", "id": msg["id"]}
        if msg.get("method") == "ping":
            reply["result"] = {}
        else:
            reply["error"] = {"code": -32601, "message": f"{msg.get('method')} not supported by anyagent"}
        try:
            self._send(reply)
        except MCPError:
            pass

    def close(self) -> None:
        pass


class MCPServer(_Base):
    """stdio transport."""

    def __init__(self, name: str, command: str, args=(), env=None, cwd=None, timeout: float = 60):
        self.name, self.timeout = name, timeout
        self.tools, self.resources = [], []
        self.stderr = collections.deque(maxlen=50)
        self._id = 0
        self._queue: queue.Queue = queue.Queue()
        try:
            self.proc = subprocess.Popen([command, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, bufsize=1, cwd=cwd,
                                         env=dict(os.environ, **(env or {})))
        except OSError as e:
            raise MCPError(f"cannot start {command}: {e}")
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._err_reader, daemon=True).start()

    def _reader(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "method" in msg and "id" in msg:
                self._handle_server_request(msg)
            else:
                self._queue.put(msg)
        self._queue.put(None)

    def _err_reader(self):
        for line in self.proc.stderr:
            self.stderr.append(line.rstrip())

    def _send(self, msg: dict) -> None:
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise MCPError(f"server {self.name} is not running: {e}")

    def request(self, method: str, params=None):
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        while True:
            try:
                msg = self._queue.get(timeout=self.timeout)
            except queue.Empty:
                raise MCPError(f"server {self.name}: timed out waiting for {method}")
            if msg is None:
                err = self.log()
                raise MCPError(f"server {self.name} exited" + (f": {err[-300:]}" if err else ""))
            if msg.get("id") != rid:
                continue  # notifications are ignored
            if "error" in msg:
                raise MCPError(msg["error"].get("message", str(msg["error"])))
            return msg.get("result")

    def notify(self, method: str, params=None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(3)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


class MCPHttpServer(_Base):
    """Streamable HTTP transport: every request is a POST; replies are JSON or an SSE stream."""

    def __init__(self, name: str, url: str, headers=None, timeout: float = 60):
        self.name, self.url, self.timeout = name, url, timeout
        self.headers = dict(headers or {})
        self.tools, self.resources = [], []
        self.stderr = collections.deque(maxlen=50)
        self.session_id = None
        self._id = 0

    def _post(self, msg: dict):
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   **self.headers}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        req = urllib.request.Request(self.url, data=json.dumps(msg).encode(), headers=headers, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raise MCPError(f"HTTP {e.code} from {self.url}: {e.read().decode('utf-8', 'replace')[:300]}")
        except urllib.error.URLError as e:
            raise MCPError(f"cannot reach {self.url}: {e.reason}")
        sid = resp.headers.get("Mcp-Session-Id")
        if sid:
            self.session_id = sid
        with resp:
            if "text/event-stream" in resp.headers.get("Content-Type", ""):
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if line.startswith("data:"):
                        try:
                            yield json.loads(line[5:].strip())
                        except ValueError:
                            continue
            else:
                body = resp.read().decode("utf-8", "replace").strip()
                if body:
                    yield json.loads(body)

    def _send(self, msg: dict) -> None:
        for _ in self._post(msg):
            pass

    def request(self, method: str, params=None):
        self._id += 1
        rid = self._id
        for msg in self._post({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}):
            if msg.get("id") != rid:
                if "method" in msg and "id" in msg:
                    self._handle_server_request(msg)
                continue
            if "error" in msg:
                raise MCPError(msg["error"].get("message", str(msg["error"])))
            return msg.get("result")
        raise MCPError(f"server {self.name}: no response to {method}")

    def notify(self, method: str, params=None) -> None:
        try:
            self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})
        except MCPError as e:
            self.stderr.append(str(e))


def load_servers(config: dict, root) -> list:
    """Start every configured server. Returns [(server | None, name, error)]."""
    specs = dict(config.get("mcp") or {})
    try:
        with open(os.path.join(root, ".mcp.json")) as f:
            specs.update((json.load(f) or {}).get("mcpServers") or {})
    except (OSError, ValueError):
        pass
    out = []
    for name, spec in specs.items():
        try:
            if spec.get("url"):
                srv = MCPHttpServer(name, spec["url"], spec.get("headers")).start()
            else:
                srv = MCPServer(name, spec["command"], spec.get("args", []), spec.get("env"), cwd=str(root)).start()
            out.append((srv, name, ""))
        except Exception as e:
            out.append((None, name, str(e)))
    return out
