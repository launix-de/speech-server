"""Command executor — all audio uses ConferenceMixer.

Producers (TTS, Play): conf.add_source(stage)
Consumers (STT): conf.add_sink(stage)
Participants (WebClient): conf.add_source + conf.add_sink(mute_source=id)
"""
from __future__ import annotations

import logging
import secrets
import threading
import time
from typing import Optional

import requests as http_requests

from . import call_state, subscriber

_LOGGER = logging.getLogger("telephony.commands")


def execute_commands(call: call_state.Call, commands: list) -> None:
    """Execute a list of commands sequentially in a background thread."""
    def _run():
        _LOGGER.info("execute_commands thread start: %d commands for %s",
                     len(commands), call.call_id)
        for cmd in commands:
            if call.status == "completed":
                _LOGGER.info("Call %s completed, stopping commands", call.call_id)
                break
            action = cmd.get("action", "")
            _LOGGER.info("Executing command: %s for call %s", action, call.call_id)
            handler = _HANDLERS.get(action)
            if not handler:
                _LOGGER.warning("Unknown command: %s", action)
                continue
            try:
                handler(call, cmd)
            except Exception as e:
                _LOGGER.warning("Command %s failed for call %s: %s",
                                action, call.call_id, e)
    threading.Thread(target=_run, daemon=True,
                     name=f"cmd-{call.call_id}").start()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _cmd_originate(call: call_state.Call, cmd: dict) -> None:
    """Originate an outbound SIP leg and bridge into conference."""
    from . import leg as leg_mod, pbx as pbx_reg

    to = cmd.get("to", "")
    callbacks = cmd.get("callbacks", {})
    pbx_entry = pbx_reg.get(call.pbx_id)
    if not pbx_entry:
        _LOGGER.warning("No PBX %s for call %s", call.pbx_id, call.call_id)
        return

    leg = leg_mod.create_leg("outbound", to, call.pbx_id,
                              call.subscriber_id)
    leg.callbacks = callbacks

    # Originate blocks until answered (or fails)
    leg_mod.originate_and_bridge(leg, call, pbx_entry)


def _cmd_add_leg(call: call_state.Call, cmd: dict) -> None:
    """Add an existing leg (by ID) into this conference."""
    from . import leg as leg_mod

    leg_id = cmd.get("leg_id", "")
    leg = leg_mod.get_leg(leg_id)
    if not leg:
        _LOGGER.warning("Leg %s not found", leg_id)
        return
    if not leg.voip_call:
        _LOGGER.warning("Leg %s has no SIP call to bridge", leg_id)
        return

    callbacks = cmd.get("callbacks", {})
    leg.callbacks.update(callbacks)
    leg_mod.bridge_to_call(leg, call)


def _cmd_tts(call: call_state.Call, cmd: dict) -> None:
    """TTS: fire-and-forget — runs in background, auto-cleanup."""
    from speech_pipeline.TTSProducer import TTSProducer
    from . import _shared, auth
    if not auth.check_feature(call.account_id, "tts"):
        _LOGGER.warning("TTS denied: account %s lacks tts feature", call.account_id)
        return

    text = cmd.get("text", "")
    voice = cmd.get("voice", "de_DE-thorsten-medium")
    callback = cmd.get("callback")
    callback_data = cmd.get("callback_data", {})
    pid = "tts-" + secrets.token_urlsafe(8)

    registry = _shared.tts_registry
    if not registry or not text:
        _LOGGER.warning("TTS failed: registry=%s text=%s", registry, text[:20] if text else None)
        _send_callback(call, callback, pid, "tts", "failed",
                       error="no text or TTS unavailable", **callback_data)
        return

    call.register_participant(pid, type="tts", text=text[:80])

    def _run():
        try:
            voice_obj = registry.ensure_loaded(voice)
            synth_config = registry.create_synthesis_config(voice_obj, {})
            producer = TTSProducer(voice_obj, synth_config, text, sentence_silence=0.0)
            src_id = call.mixer.add_source(producer)
            call.mixer.wait_source(src_id)
            call.mixer.remove_source(src_id)
        except Exception as e:
            _LOGGER.warning("TTS failed for call %s: %s", call.call_id, e)
            _send_callback(call, callback, pid, "tts", "failed",
                           error=str(e), **callback_data)
            return
        finally:
            call.unregister_participant(pid)

        _send_callback(call, callback, pid, "tts", "finished", **callback_data)

    threading.Thread(target=_run, daemon=True, name=f"tts-{pid}").start()


def _cmd_play(call: call_state.Call, cmd: dict) -> None:
    """Play audio: fire-and-forget — runs in background, auto-cleanup.

    Supports ``loop: true`` or ``loop: N`` for looping playback.
    Returns a handle (``pid``) that can be used with ``stop_play``
    to cancel the loop immediately.
    """
    from speech_pipeline.AudioReader import AudioReader

    url = cmd.get("url", "")
    callback = cmd.get("callback")
    loop = cmd.get("loop", False)
    volume = cmd.get("volume", 100)  # 0-100, default 100%
    pid = "play-" + secrets.token_urlsafe(8)

    if not url:
        _send_callback(call, callback, pid, "play", "failed", error="no url")
        return

    # Store stop event so stop_play can cancel the loop
    stop_event = threading.Event()
    call.register_participant(pid, type="play", url=url, _stop=stop_event)

    def _run():
        try:
            iterations = 0
            max_iter = int(loop) if isinstance(loop, (int, float)) and loop > 1 else (999999 if loop else 1)

            while iterations < max_iter and not stop_event.is_set() and not call.mixer.cancelled:
                reader = AudioReader(url, chunk_seconds=0.5)
                source_stage = reader
                if volume != 100:
                    from speech_pipeline.GainStage import GainStage
                    gain = GainStage(reader.output_format.sample_rate,
                                     float(volume) / 100.0)
                    reader.pipe(gain)
                    source_stage = gain

                src_id = call.mixer.add_source(source_stage)
                # Store current src_id so stop_play can kill it
                p = call.get_participant(pid)
                if p:
                    p["_current_src"] = src_id

                call.mixer.wait_source(src_id)

                if stop_event.is_set():
                    call.mixer.kill_source(src_id)
                else:
                    call.mixer.remove_source(src_id)
                iterations += 1
        except Exception as e:
            _LOGGER.warning("Play failed for call %s: %s", call.call_id, e)
            _send_callback(call, callback, pid, "play", "failed", error=str(e))
            return
        finally:
            call.unregister_participant(pid)

        _send_callback(call, callback, pid, "play", "finished")

    threading.Thread(target=_run, daemon=True, name=f"play-{pid}").start()
    return pid


def _cmd_stop_play(call: call_state.Call, cmd: dict) -> None:
    """Stop a looping play by participant ID — immediate silence."""
    pid = cmd.get("participant_id") or cmd.get("pid")
    if not pid:
        return
    p = call.get_participant(pid)
    if not p:
        return
    # Signal the loop to stop
    if p.get("_stop"):
        p["_stop"].set()
    # Kill the current source immediately (clears buffer → instant silence)
    if p.get("_current_src"):
        call.mixer.kill_source(p["_current_src"])


def _cmd_hangup(call: call_state.Call, cmd: dict) -> None:
    """Hang up entire call or a specific leg."""
    from . import leg as leg_mod

    leg_id = cmd.get("leg_id") or cmd.get("participant_id")
    if leg_id:
        leg = leg_mod.get_leg(leg_id)
        if leg:
            leg.hangup()
            leg_mod.delete_leg(leg_id)
        call.unregister_participant(leg_id)
    else:
        call.end()
        call_state.delete_call(call.call_id)


def _cmd_transfer(call: call_state.Call, cmd: dict) -> None:
    """Transfer a SIP leg or webclient to another conference.

    Removes the participant from the current conference and adds it
    to the target conference.  Works for both SIP legs and webclient
    sessions.

    Params:
    - participant_id or leg_id: which participant to move
    - target_call_id: the conference to move to
    """
    from . import leg as leg_mod

    pid = cmd.get("participant_id") or cmd.get("leg_id")
    target_call_id = cmd.get("target_call_id")
    if not pid or not target_call_id:
        _LOGGER.warning("transfer: missing participant_id or target_call_id")
        return

    target_call = call_state.get_call(target_call_id)
    if not target_call:
        _LOGGER.warning("transfer: target call %s not found", target_call_id)
        return

    # For SIP legs: kill source/sink on current mixer, re-bridge to target
    leg = leg_mod.get_leg(pid)
    if leg and leg._src_id:
        call.mixer.kill_source(leg._src_id)
        if leg._sink_id:
            call.mixer.remove_sink(leg._sink_id)
        call.unregister_participant(pid)
        leg_mod.bridge_to_call(leg, target_call)
        _LOGGER.info("Transferred leg %s from %s to %s",
                     pid, call.call_id, target_call_id)
        return

    # TODO: webclient transfer (needs DSL pipeline rebuild)
    _LOGGER.warning("transfer not yet implemented for webclient participants")


def _cmd_dtmf(call: call_state.Call, cmd: dict) -> None:
    """Send DTMF tones into a SIP leg as inband audio.

    Generates DTMF audio tones and feeds them into the conference
    mixer as a temporary source.  The receiving SIP endpoint should
    decode the tones.

    Params:
    - digits: string of digits (0-9, *, #)
    - duration_ms: per-digit duration (default 200ms)
    """
    import struct, math

    digits_str = str(cmd.get("digits", ""))
    duration_ms = cmd.get("duration_ms", 200)
    if not digits_str:
        return

    # DTMF frequency pairs (ITU-T Q.23)
    DTMF_FREQS = {
        "1": (697, 1209), "2": (697, 1336), "3": (697, 1477),
        "4": (770, 1209), "5": (770, 1336), "6": (770, 1477),
        "7": (852, 1209), "8": (852, 1336), "9": (852, 1477),
        "*": (941, 1209), "0": (941, 1336), "#": (941, 1477),
    }

    rate = call_state.MIXER_SAMPLE_RATE
    samples_per_digit = int(rate * duration_ms / 1000)
    gap_samples = int(rate * 50 / 1000)  # 50ms gap between digits

    pcm = bytearray()
    for digit in digits_str:
        freqs = DTMF_FREQS.get(digit)
        if not freqs:
            continue
        f1, f2 = freqs
        for i in range(samples_per_digit):
            t = i / rate
            sample = 0.3 * (math.sin(2 * math.pi * f1 * t) +
                            math.sin(2 * math.pi * f2 * t))
            pcm.extend(struct.pack("<h", max(-32768, min(32767, int(sample * 32767)))))
        # Gap
        pcm.extend(b"\x00\x00" * gap_samples)

    if not pcm:
        return

    # Feed as temporary source via add_source (proper Stage pipeline)
    from speech_pipeline.QueueSource import QueueSource
    import queue as _queue

    q = _queue.Queue()
    q.put(bytes(pcm))
    q.put(None)  # EOF
    src = QueueSource(q, rate, "s16le")
    src_id = call.mixer.add_source(src)
    call.mixer.wait_source(src_id)
    call.mixer.remove_source(src_id)

    _LOGGER.info("DTMF sent for call %s: %s", call.call_id, digits_str)


def _cmd_stt_start(call: call_state.Call, cmd: dict) -> None:
    """STT as conference sink: mix → resample → Whisper → WebhookSink."""
    from speech_pipeline.WhisperSTT import WhisperTranscriber
    from speech_pipeline.WebhookSink import WebhookSink
    from . import auth
    if not auth.check_feature(call.account_id, "stt"):
        _LOGGER.warning("STT denied: account %s lacks stt feature", call.account_id)
        return

    callback = cmd.get("callback")
    language = cmd.get("language", "de")
    if not callback:
        return
    if call.stt_pipeline_id:
        return

    sub = subscriber.get(call.subscriber_id)
    bearer = sub["bearer_token"] if sub else ""
    webhook_url = (sub["base_url"].rstrip("/") + "/" + callback.lstrip("/")
                   if sub else callback)

    pid = "stt-" + secrets.token_urlsafe(8)
    call.stt_pipeline_id = pid
    call.stt_callback = callback
    call.register_participant(pid, type="stt")

    # STT as a conference sink: conf.add_sink(transcriber → webhook_sink)
    # ConferenceMixer pipes via QueueSource → auto-resample → transcriber
    transcriber = WhisperTranscriber(
        model_size=cmd.get("model", "base"),
        language=language)
    webhook_sink = WebhookSink(
        url=webhook_url, bearer_token=bearer,
        extra_fields={"callId": call.call_id})
    transcriber.pipe(webhook_sink)

    call.mixer.add_sink(transcriber)
    _LOGGER.info("STT started for call %s", call.call_id)


def _cmd_stt_stop(call: call_state.Call, cmd: dict) -> None:
    pid = call.stt_pipeline_id
    call.stt_pipeline_id = None
    call.stt_callback = None
    if pid:
        call.unregister_participant(pid)
    # The sidechain thread stops when the tee sends EOF or is cancelled


def _cmd_webclient(call: call_state.Call, cmd: dict) -> None:
    """Create a webclient slot — returns iframe_url via callback."""
    from . import auth, webclient as wc

    if not auth.check_feature(call.account_id, "webclient"):
        _LOGGER.warning("WebClient denied: account %s lacks webclient feature", call.account_id)
        return

    user = cmd.get("user", "anonymous")
    callback = cmd.get("callback")
    base_url = cmd.get("base_url", "").rstrip("/")

    nonce_entry = auth.create_nonce(
        account_id=call.account_id,
        subscriber_id=call.subscriber_id,
        user=user)
    nonce = nonce_entry["nonce"]

    # Register webclient session (creates DSL, session_id etc.)
    sess = wc.register_webclient(call, user, nonce,
                                  dsl=cmd.get("dsl"),
                                  pipes=cmd.get("pipes"))
    session_id = sess["session_id"]

    call.register_participant(session_id, type="webclient", user=user,
                              nonce=nonce, callback=callback)

    # Build iframe URL with all params the phone UI needs
    from urllib.parse import urlencode
    query = urlencode({
        "base": base_url,
        "session": session_id,
        "dsl": sess["dsl"],
    })
    iframe_url = f"{base_url}/phone/{nonce}?{query}"
    _LOGGER.info("WebClient slot for call %s: %s", call.call_id, iframe_url)

    _send_callback(call, callback, session_id, "webclient", "ready",
                   iframe_url=iframe_url, nonce=nonce, user=user,
                   session_id=session_id)


# ---------------------------------------------------------------------------
# Callback helper
# ---------------------------------------------------------------------------

def _send_callback(call: call_state.Call, callback_path: Optional[str],
                   participant_id: str, command: str, result: str,
                   **extra) -> list:
    if not callback_path:
        return []
    sub = subscriber.get(call.subscriber_id)
    if not sub:
        return []

    url = sub["base_url"].rstrip("/") + "/" + callback_path.lstrip("/")
    payload = {
        "callId": call.call_id,
        "command": command,
        "participantId": participant_id,
        "result": result,
        **extra,
    }
    import threading

    def _send():
        try:
            resp = http_requests.post(url, json=payload, headers={
                "Authorization": f"Bearer {sub['bearer_token']}",
            }, timeout=10)
            if resp.status_code == 200 and resp.content:
                body = resp.json()
                cmds = body.get("commands", [])
                if cmds:
                    execute_commands(call, cmds)
        except Exception as e:
            _LOGGER.warning("Callback to %s failed: %s", url, e)

    threading.Thread(target=_send, daemon=True).start()
    return []


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS = {
    "originate": _cmd_originate,
    "add_leg": _cmd_add_leg,
    "tts": _cmd_tts,
    "play": _cmd_play,
    "stop_play": _cmd_stop_play,
    "hangup": _cmd_hangup,
    "transfer": _cmd_transfer,
    "dtmf": _cmd_dtmf,
    "stt_start": _cmd_stt_start,
    "stt_stop": _cmd_stt_stop,
    "webclient": _cmd_webclient,
}
