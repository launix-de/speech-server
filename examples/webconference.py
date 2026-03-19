#!/usr/bin/env python3
"""Web conference example.

Starts a small HTTP server.  Every visitor sees:
- An iframe to join the conference
- A live transcript log (all STT output)

When someone joins:
- A TTS bot says "Herzlich willkommen" into the conference
- The TTS bot auto-removes after speaking

A Whisper STT participant listens to the conference and posts
transcription segments back to this server via webhook.

Usage:
    python examples/webconference.py [--tts-piper http://localhost:5000] [--token test-secret-123] [--port 8080]
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
_LOG = logging.getLogger("webconference")

# Global state
PIPER_URL = "http://localhost:5000"
ADMIN_TOKEN = "test-secret-123"
MY_PORT = 8080
MY_HOST = "0.0.0.0"

ACCOUNT_ID = "demo"
ACCOUNT_TOKEN = "demo-token"
SUBSCRIBER_ID = "demo-sub"
CALL_ID = None  # set after conference creation
TRANSCRIPT: list[dict] = []  # shared transcript log


def piper(method: str, path: str, body: dict = None, token: str = None) -> dict:
    """Call tts-piper API."""
    url = f"{PIPER_URL}{path}"
    headers = {"Authorization": f"Bearer {token or ADMIN_TOKEN}"}
    resp = requests.request(method, url, json=body, headers=headers, timeout=10)
    _LOG.info("%s %s → %s", method, path, resp.status_code)
    if resp.content:
        try:
            return resp.json()
        except Exception:
            return {"raw": resp.text}
    return {}


def setup_conference() -> str:
    """Provision account, subscriber, and create the conference."""
    # 1. Create account
    piper("PUT", f"/api/accounts/{ACCOUNT_ID}", {
        "token": ACCOUNT_TOKEN,
        "features": ["tts", "stt", "webclient"],
    })

    # 2. Register subscriber (ourselves)
    my_url = f"http://127.0.0.1:{MY_PORT}"
    piper("PUT", f"/api/subscribe/{SUBSCRIBER_ID}", {
        "base_url": my_url,
        "bearer_token": "demo-bearer",
        "inbound_dids": [],
        "events": {},
    }, token=ACCOUNT_TOKEN)

    # 3. Create conference (= call)
    result = piper("POST", "/api/calls", {
        "subscriber_id": SUBSCRIBER_ID,
    }, token=ACCOUNT_TOKEN)
    call_id = result.get("call_id", "")
    _LOG.info("Conference created: %s", call_id)

    # 4. Start STT on conference
    piper("POST", f"/api/calls/{call_id}/commands", {
        "commands": [{
            "action": "stt_start",
            "language": "de",
            "model": "medium",
            "callback": "/stt-webhook",
        }],
    }, token=ACCOUNT_TOKEN)
    _LOG.info("STT started on conference")

    return call_id


JOINED_COUNT = 0
HOLD_MUSIC_PID = None


def _update_hold_music(call_id: str) -> None:
    """Start/stop hold music based on participant count.

    0 participants → silence
    1 participant  → hold music loop
    ≥2 participants → silence (people can talk)
    """
    if JOINED_COUNT == 1:
        _start_hold_music(call_id)
    else:
        _stop_hold_music(call_id)


def on_user_joined(call_id: str) -> None:
    """Called when any user joins the conference."""
    global JOINED_COUNT
    JOINED_COUNT += 1

    if JOINED_COUNT == 1:
        # First user: welcome + hold music
        piper("POST", f"/api/calls/{call_id}/commands", {
            "commands": [{
                "action": "tts",
                "text": "Herzlich willkommen in der Konferenz! Bitte warten Sie auf weitere Teilnehmer.",
                "voice": "de_DE-thorsten-medium",
            }],
        }, token=ACCOUNT_TOKEN)
    else:
        # Second+ user: announce
        piper("POST", f"/api/calls/{call_id}/commands", {
            "commands": [{
                "action": "tts",
                "text": "Ein weiterer Teilnehmer ist beigetreten.",
                "voice": "de_DE-thorsten-medium",
            }],
        }, token=ACCOUNT_TOKEN)
    _update_hold_music(call_id)


def on_user_left(call_id: str) -> None:
    """Called when a user leaves the conference."""
    global JOINED_COUNT
    JOINED_COUNT = max(0, JOINED_COUNT - 1)
    _update_hold_music(call_id)


def _start_hold_music(call_id: str) -> None:
    global HOLD_MUSIC_PID
    if HOLD_MUSIC_PID:
        return  # already playing
    piper("POST", f"/api/calls/{call_id}/commands", {
        "commands": [{
            "action": "play",
            "url": f"{PIPER_URL}/examples/queue.mp3",
            "loop": True,
            "volume": 30,
        }],
    }, token=ACCOUNT_TOKEN)
    time.sleep(0.5)
    try:
        call_data = piper("GET", f"/api/calls/{call_id}", token=ACCOUNT_TOKEN)
        for p in call_data.get("participants", []):
            if p.get("type") == "play":
                HOLD_MUSIC_PID = p["id"]
                _LOG.info("Hold music started: %s", HOLD_MUSIC_PID)
    except Exception:
        pass



def _stop_hold_music(call_id: str) -> None:
    global HOLD_MUSIC_PID
    if HOLD_MUSIC_PID:
        _LOG.info("Stopping hold music: %s", HOLD_MUSIC_PID)
        piper("POST", f"/api/calls/{call_id}/commands", {
            "commands": [{"action": "stop_play", "pid": HOLD_MUSIC_PID}],
        }, token=ACCOUNT_TOKEN)
        HOLD_MUSIC_PID = None


PENDING_IFRAME_URLS: list = []  # filled by webclient-callback


def create_webclient_slot(call_id: str) -> str:
    """Create a webclient slot and return the iframe URL."""
    PENDING_IFRAME_URLS.clear()
    piper("POST", f"/api/calls/{call_id}/commands", {
        "commands": [{
            "action": "webclient",
            "user": f"user-{int(time.time()) % 10000}",
            "base_url": PIPER_URL,
            "callback": "/webclient-callback",
        }],
    }, token=ACCOUNT_TOKEN)
    # Wait for the callback to deliver the iframe_url
    for _ in range(20):
        if PENDING_IFRAME_URLS:
            return PENDING_IFRAME_URLS.pop(0)
        time.sleep(0.25)
    return ""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

PAGE_HTML = """\
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Web Conference</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; padding: 20px; max-width: 800px; margin: 0 auto; }
h1 { margin-bottom: 16px; }
.phone-frame { width: 300px; height: 200px; border: 2px solid #ccc;
               border-radius: 12px; overflow: hidden; margin-bottom: 20px; }
.phone-frame iframe { width: 100%%; height: 100%%; border: none; }
.transcript { background: #f5f5f5; border-radius: 8px; padding: 16px;
              max-height: 400px; overflow-y: auto; font-size: 14px; }
.transcript .entry { padding: 4px 0; border-bottom: 1px solid #e0e0e0; }
.transcript .entry .time { color: #999; font-size: 12px; margin-right: 8px; }
.loading { color: #999; font-style: italic; }
#join-btn { padding: 12px 24px; font-size: 16px; background: #4caf50;
            color: white; border: none; border-radius: 8px; cursor: pointer;
            margin-bottom: 16px; }
#join-btn:hover { background: #43a047; }
</style>
</head>
<body>
<h1>Web Conference</h1>
<button id="join-btn" onclick="joinConference()">Join Conference</button>
<div id="phone-container"></div>
<h2 style="margin: 16px 0 8px;">Live Transcript</h2>
<div class="transcript" id="transcript">
  <div class="loading">Waiting for speech...</div>
</div>
<script>
function joinConference() {
  document.getElementById('join-btn').textContent = 'Joining...';
  fetch('/join').then(r => r.json()).then(data => {
    if (data.iframe_url) {
      window.open(data.iframe_url, '_blank', 'width=400,height=300');
      document.getElementById('join-btn').textContent = 'Join Conference';
    }
  });
}

function refreshTranscript() {
  fetch('/transcript').then(r => r.json()).then(entries => {
    const el = document.getElementById('transcript');
    if (entries.length === 0) {
      el.innerHTML = '<div class="loading">Waiting for speech...</div>';
      return;
    }
    el.innerHTML = entries.map(e =>
      '<div class="entry"><span class="time">' +
      new Date(e.ts * 1000).toLocaleTimeString() +
      '</span>' + e.text + '</div>'
    ).join('');
    el.scrollTop = el.scrollHeight;
  });
}

setInterval(refreshTranscript, 2000);
refreshTranscript();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            self._html(PAGE_HTML)

        elif path == "/join":
            # TTS is triggered by the "joined" callback (see do_POST)
            iframe_url = create_webclient_slot(CALL_ID)
            self._json({"iframe_url": iframe_url})

        elif path == "/transcript":
            self._json(TRANSCRIPT[-100:])

        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._read_body()

        if path == "/stt-webhook":
            # STT segment from tts-piper
            if body and body.get("text"):
                TRANSCRIPT.append({
                    "text": body["text"],
                    "ts": time.time(),
                    "start": body.get("start"),
                    "end": body.get("end"),
                })
                _LOG.info("STT: %s", body["text"])
            self._json({"ok": True})

        elif path == "/webclient-callback":
            _LOG.info("WebClient callback: %s", body)
            if body.get("result") == "ready" and body.get("iframe_url"):
                PENDING_IFRAME_URLS.append(body["iframe_url"])
            if body.get("result") == "joined":
                threading.Thread(target=on_user_joined, args=(CALL_ID,),
                                 daemon=True).start()
            if body.get("result") == "left":
                threading.Thread(target=on_user_left, args=(CALL_ID,),
                                 daemon=True).start()
            self._json({"ok": True})

        else:
            self.send_error(404)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length:
            try:
                return json.loads(self.rfile.read(length))
            except Exception:
                pass
        return {}

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _html(self, html):
        data = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass  # suppress default access log


def main():
    global PIPER_URL, ADMIN_TOKEN, MY_PORT, CALL_ID

    parser = argparse.ArgumentParser(description="Web conference example")
    parser.add_argument("--tts-piper", required=True,
                        help="tts-piper base URL (e.g. https://tts.example.com)")
    parser.add_argument("--token", required=True,
                        help="tts-piper admin token")
    parser.add_argument("--port", type=int, default=8080,
                        help="HTTP port for this example server")
    args = parser.parse_args()

    PIPER_URL = args.tts_piper.rstrip("/")
    ADMIN_TOKEN = args.token
    MY_PORT = args.port

    _LOG.info("Setting up conference on %s...", PIPER_URL)
    CALL_ID = setup_conference()
    _LOG.info("Conference ready: %s", CALL_ID)

    from socketserver import ThreadingMixIn
    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
    server = ThreadedHTTPServer((MY_HOST, MY_PORT), Handler)
    _LOG.info("Web conference running at http://localhost:%d", MY_PORT)
    _LOG.info("Open in browser to join!")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _LOG.info("Shutting down...")
        # Cleanup: end conference
        try:
            piper("DELETE", f"/api/calls/{CALL_ID}", token=ACCOUNT_TOKEN)
        except Exception:
            pass
        server.server_close()


if __name__ == "__main__":
    main()
