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
from typing import Optional

from flask import Blueprint, Response

from . import auth, call_state

_LOGGER = logging.getLogger("telephony.webclient")

bp = Blueprint("telephony_webclient", __name__)

# session_id -> {call_id, src_id (from add_source), nonce, user, ...}
_sessions: dict = {}


def register_webclient(call: call_state.Call, user: str,
                        nonce: str, dsl: str = None,
                        pipes: list = None) -> dict:
    """Register a webclient session. Returns session info with DSL.

    If *dsl* or *pipes* is given, ``{session_id}`` and ``{call_id}``
    placeholders are substituted.  Otherwise the default bidirectional
    codec-conference pipeline is used.
    """
    session_id = "wc-" + secrets.token_urlsafe(8)

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
    }
    _sessions[session_id] = entry
    _LOGGER.info("WebClient session %s for call %s (user=%s)",
                 session_id, call.call_id, user)
    return entry


def get_webclient_session(session_id: str) -> Optional[dict]:
    return _sessions.get(session_id)


def remove_webclient_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def close_webclient_session(session_id: str) -> None:
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

    _LOGGER.info("WebClient session %s closed", session_id)


def close_call_sessions(call_id: str) -> None:
    for session_id, entry in list(_sessions.items()):
        if entry.get("call_id") == call_id:
            close_webclient_session(session_id)


def get_mixer_for_session(session_id: str):
    """Called by pipe pre-hook to inject the conference mixer."""
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
.row { display: flex; gap: .5rem; justify-content: center; margin-bottom: 12px; font-size: 13px; color: #888; }
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
  <button class="btn btn-join" id="btn-rec" style="display:none">&#x260E;</button>
</div>
<!-- codec.js loaded from tts-piper server, same way as sts.html -->
<script>
// Read params
var params = new URLSearchParams(location.search);
var baseUrl = params.get('base') || location.origin;
var sessionId = params.get('session');
var dsl = params.get('dsl');
var wsBase = baseUrl.replace(/^http/, 'ws');

// Dynamic codec.js load (same origin as tts-piper)
var s = document.createElement('script');
s.src = baseUrl + '/examples/codec.js';
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
var pipeWs = null;
var codecWs = null;
var mic = null;
var speaker = null;

var btn = document.getElementById('btn-rec');
var statusEl = document.getElementById('status');

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
  pipeWs = new WebSocket(wsBase + '/ws/pipe');
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
    var encoded = Codec.encodeFrame(samples, getProfile());
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
  btn.className = 'btn btn-leave';
  btn.innerHTML = '&#x2716;';
  btn.style.display = 'flex';
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
  recording = false;
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
  statusEl.textContent = 'Disconnected';
}

btn.addEventListener('click', function () {
  if (recording) stop(); else start();
});
</script>
</body>
</html>
"""


@bp.route("/phone/<nonce>")
def phone_ui(nonce: str):
    """Serve the phone iframe UI."""
    entry = auth.validate_nonce(nonce)
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
