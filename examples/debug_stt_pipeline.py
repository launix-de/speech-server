#!/usr/bin/env python3
"""Debug: inject TTS audio into conference, check if STT picks it up.

Uses admin token for everything to avoid auth issues.
Tests the exact pipeline chain: TTS → conference mixer → STT → webhook.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import requests

RESULTS = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        RESULTS.append(body)
        print(f"  >>> WEBHOOK: {body}")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, *a):
        pass


def api(base, token, method, path, body=None):
    r = requests.request(method, f"{base}{path}",
                         json=body,
                         headers={"Authorization": f"Bearer {token}"},
                         timeout=10)
    try:
        return r.status_code, r.json() if r.content else {}
    except Exception:
        return r.status_code, {"raw": r.text}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tts-piper", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--port", type=int, default=19876)
    args = p.parse_args()

    B, T, P = args.tts_piper.rstrip("/"), args.token, args.port
    AT = "dbg-acct-tok"

    # Webhook receiver
    srv = HTTPServer(("0.0.0.0", P), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    print("1. Account + Subscriber")
    print("  ", api(B, T, "PUT", "/api/accounts/dbg",
                    {"token": AT, "features": ["tts", "stt"]}))
    print("  ", api(B, AT, "PUT", "/api/subscribe/dbg-sub",
                    {"base_url": f"http://127.0.0.1:{P}",
                     "bearer_token": "x", "inbound_dids": [], "events": {}}))

    print("\n2. Create conference")
    code, data = api(B, AT, "POST", "/api/calls",
                     {"subscriber_id": "dbg-sub"})
    call_id = data.get("call_id", "")
    print(f"   {code} call_id={call_id}")
    if not call_id:
        print(f"   FAILED: {data}")
        return

    print("\n3. STT pipeline: conference → stt → webhook")
    dsl = f"conference:{call_id} | stt:de | webhook:http://127.0.0.1:{P}/stt"
    print(f"   DSL: {dsl}")
    code, data = api(B, T, "POST", "/api/pipelines", {"dsl": dsl})
    stt_pid = data.get("id", "")
    print(f"   {code} pid={stt_pid}")
    if not stt_pid:
        print(f"   FAILED: {data}")
        return

    print("\n4. TTS pipeline: text_input → tts → conference")
    dsl2 = f"text_input | tts:de_DE-thorsten-medium | conference:{call_id}"
    code, data = api(B, T, "POST", "/api/pipelines", {"dsl": dsl2})
    tts_pid = data.get("id", "")
    print(f"   {code} pid={tts_pid}")

    print("\n5. Feed text")
    api(B, T, "POST", f"/api/pipelines/{tts_pid}/input",
        {"text": "Hallo Welt. Das ist ein Test."})

    print("   Waiting 15s...")
    for i in range(15):
        time.sleep(1)
        if RESULTS:
            break

    print(f"\n6. Result: {len(RESULTS)} webhook(s)")
    for r in RESULTS:
        print(f"   text={r.get('text')}")
    if not RESULTS:
        print("   NOTHING — STT produced no output")

    print("\n7. Cleanup")
    api(B, T, "DELETE", f"/api/pipelines/{stt_pid}")
    api(B, T, "DELETE", f"/api/pipelines/{tts_pid}")
    api(B, AT, "DELETE", f"/api/calls/{call_id}")
    srv.shutdown()


if __name__ == "__main__":
    main()
