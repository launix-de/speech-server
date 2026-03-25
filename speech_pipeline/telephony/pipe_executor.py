"""DSL pipe executor for telephony calls.

All wiring is done directly via mixer.add_source/add_sink — no
delegation to legacy klumpen-functions (bridge_to_call, _cmd_play).

DSL syntax::

    sip:leg-abc{"completed":"/hook"} -> call:call-xyz -> sip:leg-abc
    play:hold_music{"url":"https://example.com/music.mp3","loop":true} -> call:call-xyz
    tts:welcome{"text":"Willkommen"} -> call:call-xyz

Element format: ``type:id{json_params}``
Separator: ``->`` or ``|``
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from typing import Any, Dict, List, Tuple

_LOGGER = logging.getLogger("telephony.pipe-executor")


# ---------------------------------------------------------------------------
# DSL Parser
# ---------------------------------------------------------------------------

def parse_dsl(dsl: str) -> List[Tuple[str, str, dict]]:
    """Parse ``'type:id{json} -> type:id'`` into [(type, id, params), ...]."""
    elements = []
    pos = 0
    s = dsl.strip()
    while pos < len(s):
        while pos < len(s) and s[pos] in ' \t':
            pos += 1
        if pos >= len(s):
            break
        if s[pos:pos+2] == '->':
            pos += 2
            continue
        if s[pos] == '|':
            pos += 1
            continue
        start = pos
        while pos < len(s) and s[pos] not in ':{| \t':
            if s[pos] == '-' and pos+1 < len(s) and s[pos+1] == '>':
                break
            pos += 1
        typ = s[start:pos].strip()
        elem_id = ""
        if pos < len(s) and s[pos] == ':':
            pos += 1
            start = pos
            while pos < len(s) and s[pos] not in '{| \t':
                if s[pos] == '-' and pos+1 < len(s) and s[pos+1] == '>':
                    break
                pos += 1
            elem_id = s[start:pos].strip()
        params = {}
        if pos < len(s) and s[pos] == '{':
            json_start = pos
            depth, in_str, esc = 0, False, False
            while pos < len(s):
                ch = s[pos]
                if esc:
                    esc = False
                elif ch == '\\' and in_str:
                    esc = True
                elif ch == '"' and not esc:
                    in_str = not in_str
                elif not in_str:
                    if ch == '{': depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            pos += 1
                            break
                pos += 1
            try:
                params = json.loads(s[json_start:pos])
            except json.JSONDecodeError as e:
                _LOGGER.warning("JSON parse error in DSL: %s", e)
        elements.append((typ, elem_id, params))
    return elements


# ---------------------------------------------------------------------------
# Pipe Executor
# ---------------------------------------------------------------------------

class CallPipeExecutor:
    """Executes DSL pipes for a Call. All wiring via mixer directly."""

    def __init__(self, call, tts_registry=None, subscriber=None):
        self.call = call
        self._tts_registry = tts_registry
        self._subscriber = subscriber
        # stage_id -> handle with _stop event and _current_src
        self._stages: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # -- Public API --------------------------------------------------------

    def add_pipes(self, pipes: List[str]) -> List[dict]:
        results = []
        for pipe_str in pipes:
            try:
                elements = parse_dsl(pipe_str)
                self._execute_pipe(elements)
                results.append({"pipe": pipe_str, "ok": True})
            except Exception as e:
                _LOGGER.warning("Pipe failed: %s — %s", pipe_str, e, exc_info=True)
                results.append({"pipe": pipe_str, "ok": False, "error": str(e)})
        return results

    def kill_stage(self, stage_id: str) -> bool:
        with self._lock:
            handle = self._stages.pop(stage_id, None)
        if not handle:
            return False
        stop = getattr(handle, '_stop', None)
        if stop:
            stop.set()
        src_id = getattr(handle, '_current_src', None)
        if src_id:
            try:
                self.call.mixer.kill_source(src_id)
            except Exception:
                pass
        _LOGGER.info("Killed stage %s", stage_id)
        return True

    def kill_all_play(self) -> int:
        with self._lock:
            keys = [k for k in self._stages if k.startswith("play:")]
        killed = 0
        for k in keys:
            if self.kill_stage(k):
                killed += 1
        return killed

    def list_stages(self) -> List[dict]:
        with self._lock:
            return [{"id": k} for k in self._stages]

    # -- Pipe execution ----------------------------------------------------

    def _execute_pipe(self, elements):
        if not elements:
            return
        types = [e[0] for e in elements]

        if types == ["sip", "call", "sip"]:
            self._pipe_sip_bridge(elements)
        elif types == ["play", "call"]:
            self._pipe_play(elements)
        elif types == ["tts", "call"]:
            self._pipe_tts(elements)
        elif len(types) >= 2 and types[0] == "call" and "stt" in types:
            self._pipe_stt(elements)
        else:
            _LOGGER.warning("Unknown pipe pattern: %s", " -> ".join(types))

    # -- sip -> call -> sip ------------------------------------------------
    # Direct add_source + add_sink (same wiring as e2e97ca bridge_to_call)

    def _pipe_sip_bridge(self, elements):
        from . import leg as leg_mod
        from speech_pipeline.SIPSource import SIPSource
        from speech_pipeline.SIPSink import SIPSink

        sip_typ, leg_id, sip_params = elements[0]
        call_typ, call_id, call_params = elements[1]
        mixer = self.call.mixer

        leg = leg_mod.get_leg(leg_id)
        if not leg:
            raise ValueError(f"Leg {leg_id} not found")
        if not leg.voip_call:
            raise ValueError(f"Leg {leg_id} has no SIP call")

        # Attach webhook callbacks from DSL params
        if sip_params:
            leg.callbacks.update(sip_params)

        # Mark as bridged so sip_listener._wait_for_bridge returns
        leg.call_id = self.call.call_id
        leg.status = "in-progress"
        leg.answered_at = leg.answered_at or time.time()
        self.call.register_participant(leg.leg_id, type="sip",
                                       direction=leg.direction,
                                       number=leg.number)

        # Create session adapter (same as e2e97ca _CallSession — simple)
        session = leg_mod._CallSession(leg.voip_call)

        # RX: SIPSource → mixer (auto-converts 8k→48k via pipe)
        rx = SIPSource(session)
        leg._src_id = mixer.add_source(rx)

        # TX: mixer → SIPSink (auto-converts 48k→8k via pipe, mix-minus)
        tx = SIPSink(session)
        leg._sink_id = mixer.add_sink(tx, mute_source=leg._src_id)

        # DTMF monitor
        def _dtmf():
            while leg.status == "in-progress":
                try:
                    d = leg.voip_call.get_dtmf(length=1)
                    if d:
                        leg_mod._fire_callback(leg, "dtmf", digit=d,
                                               call_id=self.call.call_id)
                except Exception:
                    time.sleep(0.1)
        threading.Thread(target=_dtmf, daemon=True,
                         name=f"dtmf-{leg_id}").start()

        # Hangup monitor
        def _monitor():
            while leg.status == "in-progress":
                ended = False
                if leg.voip_call and hasattr(leg.voip_call, 'state'):
                    try:
                        from pyVoIP.VoIP.VoIP import CallState
                        if leg.voip_call.state == CallState.ENDED:
                            ended = True
                    except Exception:
                        pass
                if hasattr(leg, '_sip_call') and leg._sip_call:
                    if leg._sip_call.state == "ended":
                        ended = True
                if session.hungup.is_set():
                    ended = True
                if ended:
                    break
                time.sleep(0.5)

            leg.status = "completed"
            dur = time.time() - leg.answered_at if leg.answered_at else 0
            try:
                mixer.kill_source(leg._src_id)
                if leg._sink_id:
                    mixer.remove_sink(leg._sink_id)
            except Exception:
                pass
            self.call.unregister_participant(leg.leg_id)
            leg_mod._fire_callback(leg, "completed", duration=dur)
            _LOGGER.info("Leg %s ended (duration=%.1fs)", leg_id, dur)
            leg_mod.delete_leg(leg_id)

        threading.Thread(target=_monitor, daemon=True,
                         name=f"mon-{leg_id}").start()

        _LOGGER.info("SIP bridge: %s -> call:%s (src=%s sink=%s)",
                     leg_id, self.call.call_id, leg._src_id, leg._sink_id)

    # -- play -> call ------------------------------------------------------
    # Direct AudioReader + loop + add_source (same logic as _cmd_play)

    def _pipe_play(self, elements):
        from speech_pipeline.AudioReader import AudioReader

        play_typ, play_id, params = elements[0]
        call_typ, call_id, call_params = elements[1]

        url = params.get("url", play_id)  # URL from params or from id
        if not url:
            raise ValueError("play requires url (in params or as id)")
        loop = params.get("loop", False)
        volume = params.get("volume", 100)
        stage_id = f"play:{play_id or secrets.token_urlsafe(6)}"

        stop_event = threading.Event()
        handle = type('_H', (), {'_stop': stop_event, '_current_src': None})()
        with self._lock:
            self._stages[stage_id] = handle

        mixer = self.call.mixer

        def _run():
            try:
                iters = 0
                max_i = (int(loop) if isinstance(loop, (int, float)) and loop > 1
                         else (999999 if loop else 1))
                while iters < max_i and not stop_event.is_set() and not mixer.cancelled:
                    reader = AudioReader(url, chunk_seconds=0.5)
                    stage = reader
                    if volume != 100:
                        from speech_pipeline.GainStage import GainStage
                        g = GainStage(reader.output_format.sample_rate,
                                      float(volume) / 100.0)
                        reader.pipe(g)
                        stage = g
                    src_id = mixer.add_source(stage)
                    handle._current_src = src_id
                    mixer.wait_source(src_id)
                    if stop_event.is_set():
                        mixer.kill_source(src_id)
                    else:
                        mixer.remove_source(src_id)
                    iters += 1
            except Exception as e:
                _LOGGER.warning("Play %s failed: %s", stage_id, e)
            finally:
                with self._lock:
                    self._stages.pop(stage_id, None)

        threading.Thread(target=_run, daemon=True,
                         name=f"play-{secrets.token_urlsafe(4)}").start()
        _LOGGER.info("Play started: %s (url=%s loop=%s vol=%s)",
                     stage_id, url, loop, volume)

    # -- tts -> call -------------------------------------------------------

    def _pipe_tts(self, elements):
        tts_typ, voice_name, params = elements[0]
        voice_name = voice_name or "de_DE-thorsten-medium"
        text = params.get("text", "")
        if not text:
            raise ValueError("tts requires text param")
        if not self._tts_registry:
            raise ValueError("TTS not available")

        from speech_pipeline.TTSProducer import TTSProducer
        voice = self._tts_registry.ensure_loaded(voice_name)
        syn = self._tts_registry.create_synthesis_config(voice, {})
        producer = TTSProducer(voice, syn, text, sentence_silence=0.0)
        mixer = self.call.mixer

        def _run():
            try:
                src_id = mixer.add_source(producer)
                mixer.wait_source(src_id)
                mixer.remove_source(src_id)
            except Exception as e:
                _LOGGER.warning("TTS failed: %s", e)

        threading.Thread(target=_run, daemon=True,
                         name=f"tts-{secrets.token_urlsafe(4)}").start()
        _LOGGER.info("TTS: %s (voice=%s)", text[:40], voice_name)

    # -- call -> stt -------------------------------------------------------

    def _pipe_stt(self, elements):
        call_typ, call_id, call_params = elements[0]
        stt_elem = webhook_elem = None
        for typ, eid, params in elements[1:]:
            if typ == "stt":
                stt_elem = (typ, eid, params)
            elif typ == "webhook":
                webhook_elem = (typ, eid, params)

        if not stt_elem:
            raise ValueError("No stt element")

        lang = stt_elem[1] or "de"
        stt_params = stt_elem[2]

        from speech_pipeline.WhisperSTT import WhisperTranscriber
        stt = WhisperTranscriber(
            model_name=stt_params.get("model", "base"),
            language=lang)

        url = (webhook_elem[1] if webhook_elem
               else stt_params.get("on_receive", ""))
        if url:
            from speech_pipeline.WebhookSink import WebhookSink
            bearer = self._subscriber.get("bearer_token", "") if self._subscriber else ""
            sink = WebhookSink(url, bearer_token=bearer)
            stt.pipe(sink)
            self.call.mixer.add_sink(sink)
        else:
            self.call.mixer.add_sink(stt)

        _LOGGER.info("STT started: lang=%s", lang)
