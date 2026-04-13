"""WebClient: browser joins a conference via existing pipe + codec endpoints.

No custom WebSocket handler. The browser uses:
1. ``/ws/pipe`` — sends DSL config, receives nothing (or NDJSON if STT)
2. ``/ws/socket/{sessionId}`` — Fourier codec audio transport

The ``webclient`` command creates a session ID and returns the
URLs + DSL config for the browser. The phone UI (``/phone/{nonce}``)
is an iframe that does exactly what ``stt.html`` does: open two
WebSockets, send DSL, connect codec.

The conference mixer is injected into PipelineBuilder by registering
a ``/ws/pipe`` pre-hook that injects ``call.mixer`` for known session IDs.
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from typing import Optional

import requests as http_requests
from flask import Blueprint, Response, jsonify, request

from . import auth, call_state, subscriber

_LOGGER = logging.getLogger("telephony.webclient")

bp = Blueprint("telephony_webclient", __name__)

# session_id -> {call_id, src_id (from add_source), nonce, user,
#                created_at, last_seen, ...}
_sessions: dict = {}
_sessions_lock = threading.Lock()

# Sessions that never get a connected codec socket are reaped after this long.
SESSION_CONNECT_TIMEOUT_SECONDS = 300.0     # 5 min to connect

# How often the reaper wakes up.
_REAPER_INTERVAL_SECONDS = 30.0

_reaper_started = False
_reaper_lock = threading.Lock()


def _ensure_reaper_started() -> None:
    """Launch the reaper thread on first session creation."""
    global _reaper_started
    with _reaper_lock:
        if _reaper_started:
            return
        _reaper_started = True
    t = threading.Thread(target=_reaper_loop, daemon=True, name="webclient-reaper")
    t.start()


def _reaper_loop() -> None:
    while True:
        try:
            time.sleep(_REAPER_INTERVAL_SECONDS)
            _reap_stale_sessions()
        except Exception as e:
            _LOGGER.warning("webclient reaper error: %s", e)


def _reap_stale_sessions() -> None:
    """Drop sessions whose browser never connected within the TTL."""
    from speech_pipeline.CodecSocketSession import get_session as _get_codec
    now = time.time()
    stale: list[str] = []
    with _sessions_lock:
        for sid, entry in _sessions.items():
            created = entry.get("created_at", now)
            age = now - created
            codec_session = _get_codec(sid)
            # Already connected → let the WS-disconnect path handle cleanup.
            if codec_session is not None and codec_session.connected.is_set():
                continue
            if age > SESSION_CONNECT_TIMEOUT_SECONDS:
                stale.append(sid)
    for sid in stale:
        _LOGGER.info(
            "WebClient session %s never connected within %ds — reaping",
            sid, int(SESSION_CONNECT_TIMEOUT_SECONDS),
        )
        close_webclient_session(sid)


def register_webclient(call: call_state.Call, user: str,
                        nonce: str, dsl: str = None,
                        pipes: list = None,
                        session_id: str | None = None) -> dict:
    """Register a webclient session. Returns session info with DSL.

    If *dsl* or *pipes* is given, ``{session_id}`` and ``{call_id}``
    placeholders are substituted.  Otherwise the default bidirectional
    codec-conference pipeline is used.
    """
    # 16 bytes = 128 bit random — not guessable within the reap window.
    session_id = session_id or ("wc-" + secrets.token_urlsafe(16))

    if pipes:
        resolved = [p.replace("{session_id}", session_id)
                      .replace("{call_id}", call.call_id) for p in pipes]
        dsl = json.dumps({"pipes": resolved})
    elif dsl:
        dsl = (dsl.replace("{session_id}", session_id)
                  .replace("{call_id}", call.call_id))
    else:
        dsl = f"codec:{session_id} | conference:{call.call_id} | codec:{session_id}"

    entry = {
        "session_id": session_id,
        "call_id": call.call_id,
        "nonce": nonce,
        "user": user,
        "dsl": dsl,
        "created_at": time.time(),
    }
    with _sessions_lock:
        _sessions[session_id] = entry
    _ensure_reaper_started()
    _LOGGER.info("WebClient session %s for call %s (user=%s)",
                 session_id, call.call_id, user)
    return entry


def create_webclient_leg(call: call_state.Call, user: str,
                         leg_id: str,
                         base_url: str,
                         ready_callback: str,
                         leg_callbacks: dict,
                         number: str) -> dict:
    nonce_entry = auth.create_nonce(
        account_id=call.account_id,
        subscriber_id=call.subscriber_id,
        user=user)
    nonce = nonce_entry["nonce"]

    sess = register_webclient(call, user, nonce, session_id=leg_id)
    sess["subscriber_id"] = call.subscriber_id
    sess["leg_id"] = leg_id
    sess["number"] = number
    sess["ready_callback"] = ready_callback
    sess["leg_callbacks"] = dict(leg_callbacks or {})

    from urllib.parse import urlencode
    query = urlencode({
        "base": base_url,
        "session": sess["session_id"],
        "dsl": sess["dsl"],
    })
    iframe_url = f"{base_url}/phone/{nonce}?{query}"

    _send_callback(
        sess["subscriber_id"],
        ready_callback,
        {
            "callId": call.call_id,
            "command": "webclient",
            "participantId": leg_id,
            "result": "ready",
            "iframe_url": iframe_url,
            "nonce": nonce,
            "user": user,
            "session_id": sess["session_id"],
            "leg_id": leg_id,
        },
    )
    return {
        "leg_id": leg_id,
        "session_id": sess["session_id"],
        "iframe_url": iframe_url,
    }


def emit_leg_event(session_id: str, event: str) -> None:
    sess = _sessions.get(session_id)
    if not sess:
        return
    callback = (sess.get("leg_callbacks") or {}).get(event)
    if not callback:
        return
    _send_callback(
        sess.get("subscriber_id", ""),
        callback,
        {
            "leg_id": sess.get("leg_id", session_id),
            "event": event,
            "number": sess.get("number", ""),
            "direction": "outbound",
            "call_id": sess.get("call_id"),
        },
    )


def _send_callback(subscriber_id: str, callback_path: str, payload: dict) -> None:
    if not subscriber_id or not callback_path:
        return
    sub = subscriber.get(subscriber_id)
    if not sub:
        return
    from . import _shared
    url = _shared.subscriber_url(sub, callback_path)

    _shared.post_webhook(url, payload, sub["bearer_token"])


def get_webclient_session(session_id: str) -> Optional[dict]:
    return _sessions.get(session_id)


def remove_webclient_session(session_id: str) -> None:
    with _sessions_lock:
        _sessions.pop(session_id, None)


def close_webclient_session(session_id: str) -> None:
    """Idempotent teardown: drop from registry, revoke nonce, close codec,
    unregister participant from the call, detach any ConferenceLeg."""
    with _sessions_lock:
        entry = _sessions.pop(session_id, None)
    if not entry:
        return

    nonce = entry.get("nonce")
    if nonce:
        try:
            auth.revoke_nonce(nonce)
        except Exception:
            pass

    try:
        from speech_pipeline.CodecSocketSession import get_session as get_codec_session
        codec_session = get_codec_session(session_id)
        if codec_session:
            codec_session.close()
    except Exception:
        pass

    # Detach from the call: unregister participant so the mixer can idle out.
    call_id = entry.get("call_id")
    if call_id:
        try:
            call = call_state.get_call(call_id)
            if call:
                call.unregister_participant(session_id)
        except Exception:
            pass

    _LOGGER.info("WebClient session %s closed", session_id)


def close_call_sessions(call_id: str) -> None:
    with _sessions_lock:
        to_close = [sid for sid, entry in _sessions.items()
                    if entry.get("call_id") == call_id]
    for session_id in to_close:
        close_webclient_session(session_id)


def get_mixer_for_session(session_id: str):
    """Called by pipe pre-hook to inject the conference mixer."""
    with _sessions_lock:
        entry = _sessions.get(session_id)
    if not entry:
        return None, None
    call = call_state.get_call(entry["call_id"])
    if not call:
        return None, None
    return entry["call_id"], call.mixer


# ---------------------------------------------------------------------------
# Phone UI
# ---------------------------------------------------------------------------

_PHONE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Conference Phone</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; display: flex;
       align-items: center; justify-content: center;
       height: 100vh; background: #fafafa; }
.phone { text-align: center; }
.status { font-size: 14px; color: #666; margin-bottom: 12px; min-height: 20px; }
.btn { width: 80px; height: 80px; border-radius: 50%; border: none;
       cursor: pointer; font-size: 32px; color: #fff;
       display: flex; align-items: center; justify-content: center;
       margin: 0 auto; transition: transform 0.1s; }
.btn:active { transform: scale(0.9); }
.btn-join { background: #4caf50; }
.btn-leave { background: #f44336; }
.btn-mute { background: #4caf50; }
.btn-muted { background: #f44336; }
.row { display: flex; gap: .5rem; justify-content: center; margin-bottom: 12px; font-size: 13px; color: #888; }
.controls { display: flex; gap: 12px; justify-content: center; align-items: center; }
</style>
</head>
<body>
<div class="phone">
  <div class="status" id="status">Loading codec…</div>
  <div class="row" id="profile-sel">
    <label><input type="radio" name="profile" value="low"> low</label>
    <label><input type="radio" name="profile" value="medium" checked> mid</label>
    <label><input type="radio" name="profile" value="high"> high</label>
  </div>
  <div class="controls">
    <button class="btn btn-join" id="btn-rec" style="display:none">&#x260E;</button>
    <button class="btn btn-mute" id="btn-mute" style="display:none">&#x1F50A;</button>
  </div>
</div>
<!-- codec.js loaded from tts-piper server, same way as sts.html -->
<script>
// Read params
var params = new URLSearchParams(location.search);
var sessionId = params.get('session');
var dsl = params.get('dsl');
var appBase = location.pathname.replace(/\/phone\/[^/?#]+$/, '');
var wsBase = location.origin.replace(/^http/, 'ws') + appBase;
// The nonce is the last path segment; /ws/pipe requires auth and
// accepts nonces as short-lived tokens for the webclient session.
var nonce = location.pathname.match(/\/phone\/([^/?#]+)/);
nonce = nonce ? nonce[1] : '';

// Load codec.js relative to the current /tts app prefix.
var s = document.createElement('script');
s.src = appBase + '/examples/codec.js';
s.onload = initPhone;
s.onerror = function() { document.getElementById('status').textContent = 'Failed to load codec.js'; };
document.head.appendChild(s);

function initPhone() {
  document.getElementById('status').textContent = 'Ready';
  document.getElementById('btn-rec').style.display = 'flex';
}
</script>
<!-- Exact same pattern as sts.html — proven to work -->
<script>
var Codec;     // set after codec.js loads
var recording = false;
var muted = false;
var pipeWs = null;
var codecWs = null;
var mic = null;
var speaker = null;

var btn = document.getElementById('btn-rec');
var muteBtn = document.getElementById('btn-mute');
var statusEl = document.getElementById('status');
var eventUrl = location.pathname + '/event';

function notifyEvent(eventName) {
  if (!sessionId) return;
  try {
    fetch(eventUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session: sessionId, event: eventName }),
      keepalive: true
    }).catch(function () {});
  } catch (_) {}
}

function getProfile() {
  var sel = document.querySelector('input[name="profile"]:checked');
  return sel ? sel.value : 'medium';
}

function start() {
  Codec = AudioCodec;
  statusEl.textContent = 'Requesting microphone…';

  // 1. Get mic permission FIRST — before opening any WebSockets
  navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
  }).then(function(stream) {
    // Mic granted — now open pipeline + codec
    statusEl.textContent = 'Connecting…';
    openPipeline(stream);
  }).catch(function(e) {
    statusEl.textContent = 'Microphone denied: ' + e.message;
  });
}

// 2. Open pipe WS + codec WS (only after mic granted)
function openPipeline(micStream) {
  pipeWs = new WebSocket(wsBase + '/ws/pipe?token=' + encodeURIComponent(nonce));
  pipeWs.binaryType = 'arraybuffer';

  pipeWs.onopen = function () {
    var config = (dsl.charAt(0) === '{') ? JSON.parse(dsl) : { pipe: dsl };
    pipeWs.send(JSON.stringify(config));
    statusEl.textContent = 'Pipeline started…';
    connectCodec(micStream);
  };

  pipeWs.onerror = function () { statusEl.textContent = 'Pipeline WS error'; };
  pipeWs.onclose = function () {
    if (recording) stop();
    statusEl.textContent = 'Disconnected';
  };
}

// 3. Open codec WS
function connectCodec(micStream) {
  var profile = getProfile();

  codecWs = new WebSocket(wsBase + '/ws/socket/' + sessionId);
  codecWs.binaryType = 'arraybuffer';

  codecWs.onopen = function () {
    codecWs.send(JSON.stringify({ type: 'hello', profiles: [profile] }));
  };

  codecWs.onmessage = function (evt) {
    if (typeof evt.data !== 'string') return;
    try {
      var obj = JSON.parse(evt.data);
      if (obj.type === 'hello') {
        statusEl.textContent = 'Connected (profile: ' + obj.profile + ')';
        openChannels(micStream);
      }
      if (obj.error) statusEl.textContent = 'Error: ' + obj.error;
    } catch (_) {}
  };

  codecWs.onerror = function () { statusEl.textContent = 'Codec WS error'; };
  codecWs.onclose = function () {
    if (recording) stop();
  };
}

// 4. Start mic encoder + speaker decoder (mic already granted)
function openChannels(micStream) {
  speaker = Codec.openSpeaker(codecWs);

  // Use the already-granted mic stream directly
  var audioCtx = new AudioContext({ sampleRate: Codec.SAMPLE_RATE });
  var source = audioCtx.createMediaStreamSource(micStream);
  var processor = audioCtx.createScriptProcessor(Codec.FRAME_SAMPLES, 1, 1);
  processor.onaudioprocess = function(e) {
    if (!recording || !codecWs || codecWs.readyState !== WebSocket.OPEN) return;
    var samples = e.inputBuffer.getChannelData(0);
    var payload = samples;
    if (muted) payload = new Float32Array(samples.length);
    var encoded = Codec.encodeFrame(payload, getProfile());
    codecWs.send(encoded);
  };
  source.connect(processor);
  processor.connect(audioCtx.destination);

  mic = {
    stream: micStream,
    audioCtx: audioCtx,
    processor: processor,
    close: function() {
      processor.disconnect();
      source.disconnect();
      micStream.getTracks().forEach(function(t) { t.stop(); });
      audioCtx.close().catch(function(){});
    }
  };

  recording = true;
  muted = false;
  btn.className = 'btn btn-leave';
  btn.innerHTML = '&#x2716;';
  btn.style.display = 'flex';
  muteBtn.style.display = 'inline-flex';
  syncMuteButton();
  notifyEvent('answered');
}

// Profile switch mid-stream: close mic, reopen with new profile
// (server detects new profile from frame header automatically)
document.querySelectorAll('input[name="profile"]').forEach(function(radio) {
  radio.addEventListener('change', function() {
    if (!recording || !codecWs) return;
    if (mic) { mic.close(); mic = null; }
    Codec.openMic(codecWs, getProfile()).then(function(h) {
      mic = h;
      statusEl.textContent = 'Connected (profile: ' + getProfile() + ')';
    });
  });
});

function stop() {
  var wasRecording = recording;
  recording = false;
  muted = false;
  btn.className = 'btn btn-join';
  btn.innerHTML = '&#x260E;';

  if (mic) { mic.close(); mic = null; }
  if (speaker) { speaker.close(); speaker = null; }
  if (codecWs && codecWs.readyState === WebSocket.OPEN) {
    codecWs.send('__END__');
    codecWs.close();
  }
  codecWs = null;
  // Nonce spent — hide button
  btn.style.display = 'none';
  muteBtn.style.display = 'none';
  syncMuteButton();
  statusEl.textContent = 'Disconnected';
  if (wasRecording) notifyEvent('completed');
}

function syncMuteButton() {
  muteBtn.innerHTML = muted ? '&#x1F507;' : '&#x1F50A;';
  muteBtn.className = 'btn btn-mute' + (muted ? ' btn-muted' : '');
}

btn.addEventListener('click', function () {
  if (recording) stop(); else start();
});

muteBtn.addEventListener('click', function () {
  if (!recording) return;
  muted = !muted;
  syncMuteButton();
  statusEl.textContent = muted ? 'Connected (muted)' : 'Connected (live)';
});

window.addEventListener('beforeunload', function () {
  if (recording) notifyEvent('completed');
});
</script>
</body>
</html>
"""


@bp.route("/phone/<nonce>")
def phone_ui(nonce: str):
    """Serve the phone iframe UI.

    Non-consuming nonce check — the iframe may be loaded more than
    once (browser reload, navigation) before the user actually joins
    the call.  The nonce is burned later at the WS handshake.
    """
    entry = auth.check_nonce(nonce)
    if not entry:
        return ("Invalid or expired nonce\n", 403)
    # Find webclient session by nonce
    for sid, sess in _sessions.items():
        if sess.get("nonce") == nonce:
            resp = Response(_PHONE_HTML, mimetype="text/html")
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Permissions-Policy"] = "microphone=(*)"
            return resp
    return ("Session not found\n", 404)


@bp.route("/phone/<nonce>/event", methods=["POST"])
def phone_event(nonce: str):
    """Post lifecycle event (answered/completed) for a phone nonce.

    Non-consuming — a single call may fire both ``answered`` and
    ``completed`` through this endpoint.
    """
    entry = auth.check_nonce(nonce)
    if not entry:
        return ("Invalid or expired nonce\n", 403)

    body = request.get_json(force=True, silent=True) or {}
    session_id = body.get("session", "")
    event = body.get("event", "")
    sess = _sessions.get(session_id)
    if not sess or sess.get("nonce") != nonce:
        return ("Session not found\n", 404)
    if event not in ("answered", "completed"):
        return ("Unsupported event\n", 400)

    emit_leg_event(session_id, event)
    return jsonify({"ok": True})
