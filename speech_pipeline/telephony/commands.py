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
        import sys
        print(f"[CMD] execute_commands thread start: {len(commands)} commands for {call.call_id}", file=sys.stderr, flush=True)
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
    from . import _shared

    text = cmd.get("text", "")
    voice = cmd.get("voice", "de_DE-thorsten-medium")
    callback = cmd.get("callback")
    pid = "tts-" + secrets.token_urlsafe(8)

    registry = _shared.tts_registry
    if not registry or not text:
        _LOGGER.warning("TTS failed: registry=%s text=%s", registry, text[:20] if text else None)
        _send_callback(call, callback, pid, "tts", "failed",
                       error="no text or TTS unavailable")
        return

    call.register_participant(pid, type="tts", text=text[:80])

    def _run():
        try:
            voice_obj = registry.ensure_loaded(voice)
            synth_config = registry.create_synthesis_config(voice_obj, {})
            producer = TTSProducer(voice_obj, synth_config, text, sentence_silence=0.0)
            src_id = call.mixer.add_source(producer)

            # add_source starts a pump thread — wait for it to finish.
            # The pump thread pushes all TTS frames into the mixer queue
            # then sends EOF (None). The mixer reads them at real-time
            # pace. We wait for the source to be marked finished by the
            # mixer (all frames consumed), not just for the pump to end.
            src = call.mixer._sources.get(src_id)
            if src and src.thread:
                src.thread.join()
            # Wait for mixer to consume all buffered frames
            if src:
                import sys
                print(f"[TTS] waiting: finished={src.finished} buffer={len(src.buffer)} frame_bytes={call.mixer.frame_bytes}", file=sys.stderr, flush=True)
                while not call.mixer.cancelled:
                    if src.finished and len(src.buffer) < call.mixer.frame_bytes:
                        break
                    time.sleep(0.05)
                print(f"[TTS] done waiting: finished={src.finished} buffer={len(src.buffer)}", file=sys.stderr, flush=True)
            call.mixer.remove_source(src_id)
        except Exception as e:
            _LOGGER.warning("TTS failed for call %s: %s", call.call_id, e)
            _send_callback(call, callback, pid, "tts", "failed", error=str(e))
            return
        finally:
            call.unregister_participant(pid)

        _send_callback(call, callback, pid, "tts", "finished")

    threading.Thread(target=_run, daemon=True, name=f"tts-{pid}").start()


def _cmd_play(call: call_state.Call, cmd: dict) -> None:
    """Play audio: fire-and-forget — runs in background, auto-cleanup."""
    from speech_pipeline.AudioReader import AudioReader

    url = cmd.get("url", "")
    callback = cmd.get("callback")
    pid = "play-" + secrets.token_urlsafe(8)

    if not url:
        _send_callback(call, callback, pid, "play", "failed", error="no url")
        return

    call.register_participant(pid, type="play", url=url)

    def _run():
        try:
            reader = AudioReader(url)
            src_id = call.mixer.add_source(reader)

            src = call.mixer._sources.get(src_id)
            if src and src.thread:
                src.thread.join()
            if src:
                while not call.mixer.cancelled:
                    if src.finished and len(src.buffer) < call.mixer.frame_bytes:
                        break
                    time.sleep(0.05)
            call.mixer.remove_source(src_id)
        except Exception as e:
            _LOGGER.warning("Play failed for call %s: %s", call.call_id, e)
            _send_callback(call, callback, pid, "play", "failed", error=str(e))
            return
        finally:
            call.unregister_participant(pid)

        _send_callback(call, callback, pid, "play", "finished")

    threading.Thread(target=_run, daemon=True, name=f"play-{pid}").start()


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


def _cmd_hold(call: call_state.Call, cmd: dict) -> None:
    """Mute a leg by cancelling its RX pipeline.  Optionally play hold music."""
    from . import leg as leg_mod

    leg_id = cmd.get("leg_id") or cmd.get("participant_id")
    if leg_id:
        leg = leg_mod.get_leg(leg_id)
        if leg and leg._rx_pipeline:
            leg._rx_pipeline.cancel()  # stops SIP→mixer flow
    music = cmd.get("music")
    if music:
        _cmd_play(call, {"url": music, "action": "play"})


def _cmd_unhold(call: call_state.Call, cmd: dict) -> None:
    """Unmute a leg — rebuild the RX pipeline."""
    # TODO: rebuild SIPSource → tee → QueueSink(mixer) pipeline for this leg
    # For now, log a warning
    _LOGGER.warning("unhold not yet implemented for leg rebuild")


def _cmd_dtmf(call: call_state.Call, cmd: dict) -> None:
    _LOGGER.info("DTMF for call %s: %s", call.call_id, cmd.get("digits"))


def _cmd_stt_start(call: call_state.Call, cmd: dict) -> None:
    """STT as tee sidechain: tee → (auto resample) → Whisper → WebhookSink."""
    from speech_pipeline.WhisperSTT import WhisperTranscriber
    from speech_pipeline.WebhookSink import WebhookSink

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

    user = cmd.get("user", "anonymous")
    callback = cmd.get("callback")
    base_url = cmd.get("base_url", "").rstrip("/")

    nonce_entry = auth.create_nonce(
        account_id=call.account_id,
        subscriber_id=call.subscriber_id,
        user=user)
    nonce = nonce_entry["nonce"]

    # Register webclient session (creates DSL, session_id etc.)
    sess = wc.register_webclient(call, user, nonce)
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
    try:
        resp = http_requests.post(url, json=payload, headers={
            "Authorization": f"Bearer {sub['bearer_token']}",
        }, timeout=10)
        if resp.status_code == 200 and resp.content:
            body = resp.json()
            cmds = body.get("commands", [])
            if cmds:
                execute_commands(call, cmds)
            return cmds
    except Exception as e:
        _LOGGER.warning("Callback to %s failed: %s", url, e)
    return []


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS = {
    "originate": _cmd_originate,
    "add_leg": _cmd_add_leg,
    "tts": _cmd_tts,
    "play": _cmd_play,
    "hangup": _cmd_hangup,
    "hold": _cmd_hold,
    "unhold": _cmd_unhold,
    "dtmf": _cmd_dtmf,
    "stt_start": _cmd_stt_start,
    "stt_stop": _cmd_stt_stop,
    "webclient": _cmd_webclient,
}
