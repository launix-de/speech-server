#!/usr/bin/env python3
"""AI voice assistant example — per-user rooms, pure DSL/API.

Each visitor gets their own conference and chat log. All audio
pipelines are built via DSL and controlled through the pipeline API.
No custom telephony commands — only standard library building blocks.

Pipelines per room:
  1. Webclient (user-controlled DSL with tee for per-user STT):
     codec:{session} | tee:stt | conference:{call} | codec:{session}
     mix:stt | stt:de | webhook:http://localhost:PORT/stt
  2. Streaming TTS (created via pipeline API):
     text_input | tts{"voice":"VOICE"} | conference:CALL_ID
     Text fed via POST /api/pipelines/<pid>/input

Usage:
    python examples/ai.py \\
        --tts-piper http://localhost:5000 \\
        --token SECRET \\
        --ollama http://localhost:11434 \\
        --port 8090
"""
from __future__ import annotations

import argparse
import json
import logging
import secrets
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
_LOG = logging.getLogger("ai")

# ---------------------------------------------------------------------------
# Config (set from CLI args)
# ---------------------------------------------------------------------------
PIPER_URL = "http://localhost:5000"
ADMIN_TOKEN = ""
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:12b"
MY_PORT = 8090
MY_HOST = "0.0.0.0"
VOICE = "de_DE-thorsten-medium"

ACCOUNT_ID = "ai-demo"
ACCOUNT_TOKEN = "ai-demo-token"
SUBSCRIBER_ID = "ai-sub"


# ---------------------------------------------------------------------------
# Room — per-user state
# ---------------------------------------------------------------------------

class Room:
    def __init__(self, room_id: str, call_id: str):
        self.room_id = room_id
        self.call_id = call_id
        self.chat_log: list[dict] = []
        self.tts_pipeline_id: str | None = None
        self.pending_iframe_url: str | None = None
        self.iframe_ready = threading.Event()
        self.llm_lock = threading.Lock()
        self.ai_speaking = False
        self.stt_buffer: list[str] = []
        self.stt_timer: threading.Timer | None = None

    def add_message(self, role: str, text: str) -> int:
        self.chat_log.append({"role": role, "text": text, "ts": time.time()})
        return len(self.chat_log) - 1

    def cleanup(self):
        if self.tts_pipeline_id:
            try:
                piper("DELETE", f"/api/pipelines/{self.tts_pipeline_id}")
            except Exception:
                pass
        try:
            piper("DELETE", f"/api/calls/{self.call_id}", token=ACCOUNT_TOKEN)
        except Exception:
            pass


_rooms: dict[str, Room] = {}
_call_to_room: dict[str, str] = {}


def get_room(room_id: str) -> Room | None:
    return _rooms.get(room_id)


def get_room_by_call(call_id: str) -> Room | None:
    rid = _call_to_room.get(call_id)
    return _rooms.get(rid) if rid else None


# ---------------------------------------------------------------------------
# tts-piper API helpers
# ---------------------------------------------------------------------------

def piper(method: str, path: str, body: dict = None,
          token: str = None) -> dict:
    url = f"{PIPER_URL}{path}"
    headers = {"Authorization": f"Bearer {token or ADMIN_TOKEN}"}
    resp = requests.request(method, url, json=body, headers=headers, timeout=10)
    _LOG.info("%s %s -> %s", method, path, resp.status_code)
    if resp.content:
        try:
            return resp.json()
        except Exception:
            return {"raw": resp.text}
    return {}


def piper_call_commands(call_id: str, commands: list) -> dict:
    return piper("POST", f"/api/calls/{call_id}/commands",
                 {"commands": commands}, token=ACCOUNT_TOKEN)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_account_provisioned = False


def ensure_account():
    global _account_provisioned
    if _account_provisioned:
        return
    piper("PUT", f"/api/accounts/{ACCOUNT_ID}", {
        "token": ACCOUNT_TOKEN,
        "features": ["tts", "stt", "webclient"],
    })
    my_url = f"http://127.0.0.1:{MY_PORT}"
    piper("PUT", f"/api/subscribe/{SUBSCRIBER_ID}", {
        "base_url": my_url,
        "bearer_token": "ai-bearer",
        "inbound_dids": [],
        "events": {},
    }, token=ACCOUNT_TOKEN)
    _account_provisioned = True


def create_room() -> Room:
    ensure_account()
    result = piper("POST", "/api/calls", {
        "subscriber_id": SUBSCRIBER_ID,
    }, token=ACCOUNT_TOKEN)
    call_id = result.get("call_id", "")
    room_id = secrets.token_urlsafe(8)
    room = Room(room_id, call_id)
    _rooms[room_id] = room
    _call_to_room[call_id] = room_id
    _LOG.info("Room %s created (call %s)", room_id, call_id)
    return room


def create_webclient_slot(room: Room) -> str:
    """Create a webclient slot with per-user STT via DSL pipes."""
    room.pending_iframe_url = None
    room.iframe_ready.clear()

    from urllib.parse import urlencode
    stt_webhook = f"http://127.0.0.1:{MY_PORT}/stt?call_id={room.call_id}"

    piper_call_commands(room.call_id, [{
        "action": "webclient",
        "user": f"user-{room.room_id}",
        "base_url": PIPER_URL,
        "callback": "/webclient-callback",
        "pipes": [
            "codec:{session_id} | tee:stt | conference:{call_id} | codec:{session_id}",
            f"mix:stt | stt:de | webhook:{stt_webhook}",
        ],
    }])

    room.iframe_ready.wait(timeout=5)
    return room.pending_iframe_url or ""


def create_tts_pipeline(room: Room) -> str | None:
    """Create a streaming TTS pipeline via the pipeline API."""
    dsl = (
        f'text_input | tts{{"voice":"{VOICE}"}} | conference:{room.call_id}'
    )
    result = piper("POST", "/api/pipelines", {"dsl": dsl})
    pid = result.get("id")
    if pid:
        room.tts_pipeline_id = pid
        _LOG.info("Room %s: TTS pipeline %s", room.room_id, pid)
    return pid


def feed_tts(room: Room, text: str) -> None:
    """Feed text into the room's streaming TTS pipeline."""
    if not room.tts_pipeline_id:
        return
    piper("POST", f"/api/pipelines/{room.tts_pipeline_id}/input",
          {"text": text})


# ---------------------------------------------------------------------------
# LLM (streaming)
# ---------------------------------------------------------------------------

def stream_ollama(messages: list[dict]):
    ollama_messages = [
        {"role": "system", "content": (
            "Du bist ein hilfreicher Sprachassistent in einem Telefonat. "
            "Wenn ein Nutzer beitritt, begruesse ihn freundlich. "
            "Antworte praezise und hilfreich. Keine Emojis. "
            "Bevorzuge kurze Antworten, aber wenn der Nutzer ausfuehrliche "
            "Antworten oder Code verlangt, liefere sie."
        )},
    ]
    for m in messages:
        ollama_messages.append({"role": m["role"], "content": m["text"]})

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": ollama_messages,
                  "stream": True},
            timeout=120, stream=True,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            obj = json.loads(line)
            content = obj.get("message", {}).get("content", "")
            if content:
                yield content
            if obj.get("done"):
                break
    except Exception as e:
        _LOG.warning("Ollama error: %s", e)
        yield "Entschuldigung, ich habe gerade technische Schwierigkeiten."


# ---------------------------------------------------------------------------
# Event handlers (all room-scoped)
# ---------------------------------------------------------------------------

def on_user_joined(room: Room) -> None:
    """User connected — create TTS pipeline, let LLM greet."""
    create_tts_pipeline(room)
    room.add_message("system", "Ein Nutzer ist beigetreten.")

    def _greet():
        with room.llm_lock:
            snapshot = list(room.chat_log)
        stream_and_speak(room, snapshot)

    threading.Thread(target=_greet, daemon=True).start()


def stream_and_speak(room: Room, messages: list[dict]) -> None:
    """Stream LLM response, feed sentences to TTS pipeline."""
    if not room.tts_pipeline_id:
        _LOG.warning("Room %s: no TTS pipeline", room.room_id)
        return

    room.ai_speaking = True
    accumulated = ""
    full_response = ""

    try:
        for chunk in stream_ollama(messages):
            accumulated += chunk
            full_response += chunk

            while _has_sentence_end(accumulated):
                sentence, accumulated = _split_sentence(accumulated)
                if sentence.strip():
                    feed_tts(room, sentence.strip())

        if accumulated.strip():
            feed_tts(room, accumulated.strip())

        if full_response:
            room.add_message("assistant", full_response)
            _LOG.info("Room %s AI: %s", room.room_id, full_response[:80])
    finally:
        time.sleep(2)
        room.ai_speaking = False


# ---------------------------------------------------------------------------
# STT filtering and debounce
# ---------------------------------------------------------------------------

_WHISPER_HALLUCINATIONS = {
    "du", "sie", "der hier", "der hier und der hier",
    "untertitelung von wikipedia", "untertitel von",
    "untertitel der amara.org-community",
    "vielen dank", "danke", "tschüss", "bis dann",
    "copyright", "www", "thank you", "thanks for watching",
    ".", "..", "...", "!", "?",
}

_STT_DEBOUNCE = 1.0


def _is_hallucination(text: str) -> bool:
    t = text.lower().strip(" .!?,")
    if len(t) < 3:
        return True
    if t in _WHISPER_HALLUCINATIONS:
        return True
    return False


def _flush_stt_buffer(room: Room) -> None:
    room.stt_timer = None
    if not room.stt_buffer:
        return
    text = " ".join(room.stt_buffer)
    room.stt_buffer.clear()

    room.add_message("user", text)
    _LOG.info("Room %s user: %s", room.room_id, text)

    def _respond():
        with room.llm_lock:
            snapshot = list(room.chat_log)
        stream_and_speak(room, snapshot)

    threading.Thread(target=_respond, daemon=True).start()


def on_stt_segment(room: Room, data: dict) -> None:
    text = data.get("text", "").strip()
    if not text:
        return
    if _is_hallucination(text):
        _LOG.debug("Room %s filtered: %s", room.room_id, text)
        return
    if room.ai_speaking:
        _LOG.debug("Room %s ignored (AI speaking): %s", room.room_id, text)
        return

    room.stt_buffer.append(text)
    _LOG.info("Room %s STT: %s (buf=%d)", room.room_id, text, len(room.stt_buffer))

    if text.rstrip().endswith((".", "!", "?")):
        if room.stt_timer:
            room.stt_timer.cancel()
        _flush_stt_buffer(room)
        return

    if room.stt_timer:
        room.stt_timer.cancel()
    room.stt_timer = threading.Timer(_STT_DEBOUNCE, _flush_stt_buffer, args=(room,))
    room.stt_timer.start()


def _has_sentence_end(text: str) -> bool:
    for ch in ".!?":
        idx = text.find(ch)
        if idx >= 0 and idx < len(text) - 1:
            return True
    return False


def _split_sentence(text: str) -> tuple[str, str]:
    best = len(text)
    for ch in ".!?":
        idx = text.find(ch)
        if 0 <= idx < best:
            best = idx
    return text[:best + 1], text[best + 1:]


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

PAGE_HTML = """\
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Voice Assistant</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee;
       display: flex; flex-direction: column; align-items: center;
       min-height: 100vh; padding: 24px; }
h1 { margin-bottom: 20px; font-size: 22px; color: #e0e0ff; }

#join-btn { padding: 14px 32px; font-size: 16px; background: #4caf50;
            color: white; border: none; border-radius: 24px; cursor: pointer;
            margin-bottom: 24px; transition: background 0.2s; }
#join-btn:hover { background: #43a047; }
#join-btn:disabled { background: #555; cursor: default; }

.chat { width: 100%%; max-width: 600px; flex: 1; overflow-y: auto;
        display: flex; flex-direction: column; gap: 12px;
        padding-bottom: 20px; }

.msg { max-width: 80%%; padding: 10px 14px; border-radius: 16px;
       font-size: 15px; line-height: 1.5; word-wrap: break-word;
       white-space: pre-wrap; }
.msg.user { align-self: flex-end; background: #2d5a27; border-bottom-right-radius: 4px; }
.msg.assistant { align-self: flex-start; background: #2a2a4a;
                 border-bottom-left-radius: 4px; }
.msg .role { font-size: 11px; color: #999; margin-bottom: 2px; }

.empty { color: #666; font-style: italic; text-align: center; margin-top: 40px; }
</style>
</head>
<body>
<h1>AI Voice Assistant</h1>
<button id="join-btn" onclick="joinConference()">Beitreten</button>
<div class="chat" id="chat">
  <div class="empty">Klicke auf "Beitreten" und sprich los.</div>
</div>
<script>
var roomId = ROOM_ID_PLACEHOLDER;

function joinConference() {
  var btn = document.getElementById('join-btn');
  btn.textContent = 'Verbinde...';
  btn.disabled = true;
  fetch('/room/' + roomId + '/join').then(r => r.json()).then(data => {
    if (data.iframe_url) {
      window.open(data.iframe_url, '_blank', 'width=400,height=300');
      btn.textContent = 'Verbunden';
    } else {
      btn.textContent = 'Fehler';
      btn.disabled = false;
    }
  }).catch(() => { btn.textContent = 'Fehler'; btn.disabled = false; });
}

var lastLen = 0;
function refreshChat() {
  fetch('/room/' + roomId + '/chat').then(r => r.json()).then(msgs => {
    if (msgs.length === lastLen) return;
    lastLen = msgs.length;
    var el = document.getElementById('chat');
    el.innerHTML = msgs.map(m =>
      '<div class="msg ' + m.role + '">' +
      '<div class="role">' + (m.role === 'user' ? 'Du' : 'Assistent') + '</div>' +
      escapeHtml(m.text) + '</div>'
    ).join('');
    el.scrollTop = el.scrollHeight;
  });
}

function escapeHtml(s) {
  var d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

setInterval(refreshChat, 1000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            room = create_room()
            self.send_response(302)
            self.send_header("Location", f"/room/{room.room_id}")
            self.end_headers()

        elif path.startswith("/room/"):
            parts = path.split("/")
            if len(parts) < 3:
                self.send_error(404)
                return
            room_id = parts[2]
            sub = parts[3] if len(parts) > 3 else ""
            room = get_room(room_id)
            if not room:
                self.send_error(404)
                return

            if sub == "":
                html = PAGE_HTML.replace("ROOM_ID_PLACEHOLDER",
                                         json.dumps(room_id))
                self._html(html)
            elif sub == "join":
                iframe_url = create_webclient_slot(room)
                self._json({"iframe_url": iframe_url})
            elif sub == "chat":
                self._json([{"role": m["role"], "text": m["text"]}
                            for m in room.chat_log if m["role"] != "system"])
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._read_body()

        call_id = body.get("callId", "")
        if not call_id:
            qs = parse_qs(urlparse(self.path).query)
            call_id = qs.get("call_id", [""])[0]
        room = get_room_by_call(call_id) if call_id else None

        if path.startswith("/stt"):
            if room:
                on_stt_segment(room, body)
            self._json({"ok": True})

        elif path == "/webclient-callback":
            _LOG.info("WebClient callback: %s", body)
            if body.get("result") == "ready" and body.get("iframe_url"):
                if room:
                    room.pending_iframe_url = body["iframe_url"]
                    room.iframe_ready.set()
                self._json({"ok": True})
            elif body.get("result") == "joined":
                if room:
                    threading.Thread(target=on_user_joined, args=(room,),
                                     daemon=True).start()
                self._json({"ok": True})
            elif body.get("result") == "left":
                _LOG.info("User left")
                self._json({"ok": True})
            else:
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
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global PIPER_URL, ADMIN_TOKEN, OLLAMA_URL, OLLAMA_MODEL
    global MY_PORT, VOICE

    parser = argparse.ArgumentParser(description="AI voice assistant example")
    parser.add_argument("--tts-piper", required=True,
                        help="tts-piper base URL")
    parser.add_argument("--token", required=True,
                        help="tts-piper admin token")
    parser.add_argument("--ollama", default="http://localhost:11434",
                        help="Ollama API base URL")
    parser.add_argument("--model", default="gemma3:12b",
                        help="Ollama model name")
    parser.add_argument("--voice", default="de_DE-thorsten-medium",
                        help="TTS voice")
    parser.add_argument("--port", type=int, default=8090,
                        help="HTTP port for this server")
    args = parser.parse_args()

    PIPER_URL = args.tts_piper.rstrip("/")
    ADMIN_TOKEN = args.token
    OLLAMA_URL = args.ollama.rstrip("/")
    OLLAMA_MODEL = args.model
    VOICE = args.voice
    MY_PORT = args.port

    from socketserver import ThreadingMixIn

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer((MY_HOST, MY_PORT), Handler)
    _LOG.info("AI assistant running at http://localhost:%d", MY_PORT)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _LOG.info("Shutting down...")
        for room in list(_rooms.values()):
            room.cleanup()
        server.server_close()


if __name__ == "__main__":
    main()
