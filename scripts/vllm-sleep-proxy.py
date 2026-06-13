#!/usr/bin/env python3
"""
vllm-sleep-proxy.py — Reverse proxy for vLLM with auto sleep/wake.

Listens on a port, forwards to vLLM. Puts vLLM to sleep after N seconds
of no requests. Wakes vLLM before forwarding if sleeping.

Usage:
    python3 vllm-sleep-proxy.py --port 8078 --upstream http://localhost:8079 --idle-seconds 300

Requires: vLLM running with --enable-sleep-mode and VLLM_SERVER_DEV_MODE=1
"""

import argparse
import json
import threading
import time
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler

# Defaults
LISTEN_PORT = 8078
UPSTREAM = "http://localhost:8079"
IDLE_TIMEOUT = 300  # 5 minutes

# State
last_activity = time.time()
sleeping = False


def touch():
    """Mark activity."""
    global last_activity
    last_activity = time.time()


def is_sleeping():
    global sleeping
    return sleeping


def sleep_vllm():
    """Put vLLM to sleep (level 1: offload weights to CPU)."""
    global sleeping
    try:
        req = urllib.request.Request(
            f"{UPSTREAM}/dev/sleep",
            data=json.dumps({"level": 1}).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as res:
            res.read()
        sleeping = True
        print(f"[sleep-proxy] vLLM asleep (VRAM freed)")
    except Exception as e:
        print(f"[sleep-proxy] sleep failed: {e}")


def wake_vllm():
    """Wake vLLM from sleep."""
    global sleeping
    try:
        req = urllib.request.Request(
            f"{UPSTREAM}/dev/wake_up",
            data=b"",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as res:
            res.read()
        sleeping = False
        print(f"[sleep-proxy] vLLM awake")
    except Exception as e:
        print(f"[sleep-proxy] wake failed: {e}")


def check_vllm_sleeping():
    """Check vLLM /is_sleeping endpoint."""
    try:
        req = urllib.request.Request(f"{UPSTREAM}/dev/is_sleeping", method="GET")
        with urllib.request.urlopen(req, timeout=5) as res:
            return json.loads(res.read())
    except Exception:
        return None


def idle_watcher():
    """Background thread: sleep vLLM after IDLE_TIMEOUT seconds of no requests."""
    global last_activity, sleeping
    while True:
        time.sleep(5)
        idle = time.time() - last_activity
        if idle > IDLE_TIMEOUT and not sleeping:
            print(f"[sleep-proxy] idle {idle:.0f}s, sleeping vLLM...")
            sleep_vllm()


class ProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        touch()
        self._proxy()

    def do_POST(self):
        touch()
        self._proxy()

    def do_PUT(self):
        touch()
        self._proxy()

    def do_DELETE(self):
        touch()
        self._proxy()

    def do_PATCH(self):
        touch()
        self._proxy()

    def _proxy(self):
        # If vLLM is sleeping, wake it first
        if is_sleeping():
            print(f"[sleep-proxy] request arrived while sleeping, waking...")
            wake_vllm()

        # Read body
        body = b""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            body = self.rfile.read(content_length)

        # Build upstream request
        upstream_url = f"{UPSTREAM}{self.path}"
        req = urllib.request.Request(
            upstream_url,
            data=body if self.command in ("POST", "PUT", "PATCH") else None,
            method=self.command,
        )

        # Forward headers (skip hop-by-hop)
        skip_headers = {"host", "content-length", "connection", "transfer-encoding"}
        for key, val in self.headers.items():
            if key.lower() not in skip_headers:
                req.add_header(key, val)

        try:
            with urllib.request.urlopen(req, timeout=300) as res:
                self.send_response(res.status)
                for key, val in res.getheaders():
                    if key.lower() not in {"transfer-encoding", "connection"}:
                        self.send_header(key, val)
                self.end_headers()
                while True:
                    chunk = res.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except urllib.error.HTTPError as e:
            # Forward error status from vLLM
            self.send_response(e.code)
            for key, val in e.headers.items():
                self.send_header(key, val)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            print(f"[sleep-proxy] proxy error: {e}")
            self.send_response(502)
            self.end_headers()
            self.wfile.write(b"Bad Gateway")

    def log_message(self, format, *args):
        # Suppress default logging — too noisy for streaming
        pass


def main():
    parser = argparse.ArgumentParser(description="vLLM sleep proxy")
    parser.add_argument("--port", type=int, default=LISTEN_PORT, help="Listen port")
    parser.add_argument("--upstream", default=UPSTREAM, help="vLLM upstream URL")
    parser.add_argument("--idle-seconds", type=int, default=IDLE_TIMEOUT, help="Idle timeout before sleep")
    args = parser.parse_args()

    global UPSTREAM, LISTEN_PORT, IDLE_TIMEOUT
    UPSTREAM = args.upstream
    LISTEN_PORT = args.port
    IDLE_TIMEOUT = args.idle_seconds

    print(f"[sleep-proxy] listening on :{LISTEN_PORT}, upstream {UPSTREAM}, idle {IDLE_TIMEOUT}s")

    # Start idle watcher
    watcher = threading.Thread(target=idle_watcher, daemon=True)
    watcher.start()

    server = HTTPServer(("0.0.0.0", LISTEN_PORT), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[sleep-proxy] shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
