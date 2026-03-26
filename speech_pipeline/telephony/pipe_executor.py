"""Generic DSL pipe executor for telephony calls.

Left-to-right chain. Normal stages are connected via pipe().
Special substitutions:
- call:ID → ConferenceLeg (becomes normal stage)
- tee:ID as first element + already exists → add_sidechain mode
- sip/codec without ID → reuse first same-type predecessor's session

Element syntax: ``type:id{json_params}``
Separator: ``->`` or ``|``
"""
from __future__ import annotations

import json
import logging
import re
import secrets
import threading
import time
from typing import Any, Dict, List, Tuple

_LOGGER = logging.getLogger("telephony.pipe-executor")

# ---------------------------------------------------------------------------
# DSL Parser
# ---------------------------------------------------------------------------

_ARROW = re.compile(r'\s*(->|\|)\s*')
_ELEMENT = re.compile(r'\s*([a-z_]+)(?::([^{\s|>]+))?')

def _consume_json(s, pos):
    if pos >= len(s) or s[pos] != '{':
        return {}, pos
    depth, in_str, esc, start = 0, False, False, pos
    while pos < len(s):
        ch = s[pos]
        if esc: esc = False
        elif ch == '\\' and in_str: esc = True
        elif ch == '"' and not esc: in_str = not in_str
        elif not in_str:
            if ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    pos += 1
                    try: return json.loads(s[start:pos]), pos
                    except: return {}, pos
        pos += 1
    return {}, pos

def parse_dsl(dsl: str) -> List[Tuple[str, str, dict]]:
    elements = []
    pos = 0
    s = dsl.strip()
    while pos < len(s):
        m = _ARROW.match(s, pos)
        if m: pos = m.end(); continue
        m = _ELEMENT.match(s, pos)
        if not m: pos += 1; continue
        typ, elem_id = m.group(1), m.group(2) or ""
        pos = m.end()
        params, pos = _consume_json(s, pos)
        elements.append((typ, elem_id, params))
    return elements

# ---------------------------------------------------------------------------
# Pipe Executor
# ---------------------------------------------------------------------------

class CallPipeExecutor:
    def __init__(self, call, tts_registry=None, subscriber=None):
        self.call = call
        self._tts_registry = tts_registry
        self._subscriber = subscriber
        self._stages: Dict[str, Any] = {}   # play handles for kill
        self._tees: Dict[str, Any] = {}     # tee_id -> AudioTee
        self._sessions: Dict[str, Any] = {} # typ -> first session (for ID reuse)
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
        # Signal stop event (for legacy play loops)
        stop = getattr(handle, '_stop', None)
        if stop: stop.set()
        # Cancel the stage itself (stops AudioReader/ConferenceLeg pipeline)
        stage = getattr(handle, '_stage', None)
        if stage and hasattr(stage, 'cancel'):
            try: stage.cancel()
            except: pass
        # Kill mixer source directly
        src_id = getattr(handle, '_current_src', None)
        if src_id:
            try: self.call.mixer.kill_source(src_id)
            except: pass
        _LOGGER.info("Killed stage %s", stage_id)
        return True

    def kill_all_play(self) -> int:
        with self._lock:
            keys = [k for k in self._stages if k.startswith("play:")]
        return sum(1 for k in keys if self.kill_stage(k))

    def list_stages(self) -> List[dict]:
        with self._lock:
            return [{"id": k} for k in self._stages]

    # -- Pipe execution ----------------------------------------------------

    def _execute_pipe(self, elements):
        if not elements:
            return

        # Phase 1: fill in missing IDs (sip without ID = first sip predecessor)
        self._fill_ids(elements)

        # Phase 2: check for sidechain mode (tee:ID as first, already exists)
        first_typ, first_id, first_params = elements[0]
        if first_typ == "tee" and first_id in self._tees:
            self._add_sidechain(elements)
            return

        # Phase 3: resolve all elements to Stage objects
        resolved = []
        for typ, elem_id, params in elements:
            stage = self._create_stage(typ, elem_id, params)
            resolved.append((typ, elem_id, params, stage))

        # Phase 4: wrap SIP sessions with SIPSource/SIPSink
        resolved = self._wrap_sip(resolved)

        # Phase 5: link play handles to their ConferenceLeg (for kill_stage)
        for i in range(len(resolved) - 1):
            l_typ, _, _, l_stage = resolved[i]
            r_typ, _, _, r_stage = resolved[i + 1]
            play_handle = getattr(l_stage, '_play_handle', None)
            from speech_pipeline.ConferenceLeg import ConferenceLeg
            if play_handle and isinstance(r_stage, ConferenceLeg):
                play_handle._stage = r_stage

        # Phase 6: chain via pipe()
        for i in range(len(resolved) - 1):
            _, _, _, l_stage = resolved[i]
            _, _, _, r_stage = resolved[i + 1]
            l_stage.pipe(r_stage)

        # Phase 6: start terminal + monitors
        self._start_all(resolved)

    # -- Fill missing IDs --------------------------------------------------

    def _fill_ids(self, elements):
        """Fill in missing IDs: sip/codec without ID = reuse first predecessor."""
        seen = {}  # typ -> first ID
        for i, (typ, elem_id, params) in enumerate(elements):
            if elem_id:
                if typ not in seen:
                    seen[typ] = elem_id
            else:
                if typ in seen:
                    elements[i] = (typ, seen[typ], params)

    # -- Sidechain mode ----------------------------------------------------

    def _add_sidechain(self, elements):
        """tee:ID (existing) -> ... : add remaining chain as sidechain."""
        tee_id = elements[0][1]
        tee = self._tees[tee_id]

        # Build the sidechain: resolve remaining elements, pipe them
        chain_stages = []
        for typ, elem_id, params in elements[1:]:
            stage = self._create_stage(typ, elem_id, params)
            chain_stages.append(stage)

        for i in range(len(chain_stages) - 1):
            chain_stages[i].pipe(chain_stages[i + 1])

        if chain_stages:
            tee.add_sidechain(chain_stages[0])
            _LOGGER.info("Tee %s: added sidechain (%d stages)", tee_id, len(chain_stages))

    # -- Stage creation ----------------------------------------------------

    def _create_stage(self, typ, elem_id, params):
        # -- call: substitute with ConferenceLeg --
        if typ == "call":
            from speech_pipeline.ConferenceLeg import ConferenceLeg
            conf_leg = ConferenceLeg(sample_rate=self.call.mixer.sample_rate)
            conf_leg.attach(self.call.mixer)
            return conf_leg

        # -- tee: get or create --
        if typ == "tee":
            with self._lock:
                if elem_id in self._tees:
                    return self._tees[elem_id]
            from speech_pipeline.AudioTee import AudioTee
            tee = AudioTee(48000, "s16le")
            with self._lock:
                self._tees[elem_id] = tee
            return tee

        # -- sip: resolve leg session, create Source or Sink --
        if typ == "sip":
            return self._create_sip(elem_id, params)

        # -- play: AudioReader with loop handle --
        if typ == "play":
            return self._create_play(elem_id, params)

        # -- tts: TTSProducer --
        if typ == "tts":
            return self._create_tts(elem_id, params)

        # -- stt: WhisperTranscriber (+ webhook if on_receive) --
        if typ == "stt":
            return self._create_stt(elem_id, params)

        # -- webhook: WebhookSink --
        if typ == "webhook":
            from speech_pipeline.WebhookSink import WebhookSink
            bearer = self._subscriber.get("bearer_token", "") if self._subscriber else ""
            return WebhookSink(elem_id, bearer_token=bearer)

        raise ValueError(f"Unknown element type: {typ}")

    def _create_sip(self, leg_id, params):
        """Resolve SIP leg. Returns session (SIPSource/SIPSink wrap via pipe)."""
        from . import leg as leg_mod

        # Check if we already have a session for this leg
        session_key = f"sip:{leg_id}"
        if session_key in self._sessions:
            return self._sessions[session_key]

        leg = leg_mod.get_leg(leg_id)
        if not leg:
            raise ValueError(f"Leg {leg_id} not found")
        if not leg.voip_call:
            raise ValueError(f"Leg {leg_id} has no SIP call")

        if params:
            leg.callbacks.update(params)

        leg.call_id = self.call.call_id
        leg.status = "in-progress"
        leg.answered_at = leg.answered_at or time.time()
        self.call.register_participant(leg.leg_id, type="sip",
                                       direction=leg.direction,
                                       number=leg.number)

        session = getattr(leg, '_sip_session', None) or leg_mod._CallSession(leg.voip_call)
        self._sessions[session_key] = session
        self._sessions[f"_leg:{leg_id}"] = leg
        return session

    def _create_play(self, play_id, params):
        from speech_pipeline.AudioReader import AudioReader
        url = params.get("url", play_id)
        if not url:
            raise ValueError("play requires url")
        volume = params.get("volume", 100)
        stage_id = f"play:{play_id or secrets.token_urlsafe(6)}"

        stop_event = threading.Event()
        handle = type('_H', (), {
            '_stop': stop_event, '_current_src': None,
            '_url': url, '_loop': params.get("loop", False),
            '_volume': volume, '_stage_id': stage_id,
        })()
        with self._lock:
            self._stages[stage_id] = handle

        reader = AudioReader(url, chunk_seconds=0.5)
        source = reader
        if volume != 100:
            from speech_pipeline.GainStage import GainStage
            g = GainStage(reader.output_format.sample_rate, float(volume) / 100.0)
            reader.pipe(g)
            source = g
        source._play_handle = handle
        return source

    def _create_tts(self, voice_name, params):
        voice_name = voice_name or "de_DE-thorsten-medium"
        text = params.get("text", "")
        if not text:
            raise ValueError("tts requires text")
        if not self._tts_registry:
            raise ValueError("TTS not available")
        from speech_pipeline.TTSProducer import TTSProducer
        voice = self._tts_registry.ensure_loaded(voice_name)
        syn = self._tts_registry.create_synthesis_config(voice, {})
        return TTSProducer(voice, syn, text, sentence_silence=0.0)

    def _create_stt(self, lang, params):
        lang = lang or "de"
        from speech_pipeline.WhisperSTT import WhisperTranscriber
        stt = WhisperTranscriber(model_size=params.get("model", "base"),
                                  language=lang)
        cb = params.get("on_receive", "")
        if cb:
            from speech_pipeline.WebhookSink import WebhookSink
            bearer = self._subscriber.get("bearer_token", "") if self._subscriber else ""
            stt.pipe(WebhookSink(cb, bearer_token=bearer))
        return stt

    # -- SIP wrapping ------------------------------------------------------

    def _wrap_sip(self, resolved):
        """Replace sip sessions with SIPSource (first) / SIPSink (last)."""
        from speech_pipeline.SIPSource import SIPSource
        from speech_pipeline.SIPSink import SIPSink

        result = []
        sip_seen = {}  # leg_id -> "source" created

        for i, (typ, elem_id, params, stage) in enumerate(resolved):
            if typ != "sip":
                result.append((typ, elem_id, params, stage))
                continue

            if elem_id not in sip_seen:
                # First occurrence: SIPSource
                src = SIPSource(stage)
                result.append(("sip_source", elem_id, params, src))
                sip_seen[elem_id] = True
            else:
                # Second occurrence: SIPSink
                sink = SIPSink(stage)
                result.append(("sip_sink", elem_id, params, sink))

        return result

    def _start_all(self, resolved):
        """Start terminal sink + SIP monitors + play loops."""
        from . import leg as leg_mod

        # Find terminal sink (last stage with run())
        terminal = None
        for typ, elem_id, params, stage in reversed(resolved):
            if hasattr(stage, 'run'):
                terminal = (typ, elem_id, params, stage)
                break

        if terminal:
            typ, elem_id, params, stage = terminal
            def _run():
                try:
                    stage.run()
                except Exception as e:
                    _LOGGER.warning("Terminal error: %s", e)
            threading.Thread(target=_run, daemon=True,
                             name=f"terminal-{typ}-{elem_id}").start()
        else:
            # No terminal sink — last stage might be ConferenceLeg (source-only).
            # Drain its output to activate the pump thread.
            last_typ, last_id, last_params, last_stage = resolved[-1]
            from speech_pipeline.ConferenceLeg import ConferenceLeg
            if isinstance(last_stage, ConferenceLeg):
                def _drain():
                    try:
                        for _ in last_stage.stream_pcm24k():
                            if last_stage.cancelled:
                                break
                    except Exception:
                        pass
                threading.Thread(target=_drain, daemon=True,
                                 name=f"drain-confleg").start()

        # Start SIP monitors for all sip legs
        for typ, elem_id, params, stage in resolved:
            if typ == "sip_source":
                leg = self._sessions.get(f"_leg:{elem_id}")
                session = self._sessions.get(f"sip:{elem_id}")
                if leg and session:
                    self._start_sip_monitors(leg, session)

        # Start play loops
        for typ, elem_id, params, stage in resolved:
            if typ in ("play",):
                handle = getattr(stage, '_play_handle', None)
                if handle and handle._current_src:
                    self._start_play_loop(handle, handle._current_src)

    # -- SIP monitors ------------------------------------------------------

    def _start_sip_monitors(self, leg, session):
        from . import leg as leg_mod
        leg_id = leg.leg_id

        def _dtmf():
            while leg.status == "in-progress":
                try:
                    d = leg.voip_call.get_dtmf(length=1)
                    if d:
                        leg_mod._fire_callback(leg, "dtmf", digit=d,
                                               call_id=self.call.call_id)
                except Exception:
                    time.sleep(0.1)
        threading.Thread(target=_dtmf, daemon=True, name=f"dtmf-{leg_id}").start()

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
            if hasattr(leg, '_conf_leg') and leg._conf_leg:
                try: leg._conf_leg.cancel()
                except: pass
            self.call.unregister_participant(leg.leg_id)
            leg_mod._fire_callback(leg, "completed", duration=dur)
            _LOGGER.info("Leg %s ended (%.1fs)", leg_id, dur)
            leg_mod.delete_leg(leg_id)
        threading.Thread(target=_monitor, daemon=True, name=f"mon-{leg_id}").start()

        _LOGGER.info("SIP leg %s wired", leg_id)

    # -- Play loop ---------------------------------------------------------

    def _start_play_loop(self, handle, first_src_id):
        mixer = self.call.mixer
        stop = handle._stop
        url, loop, volume = handle._url, handle._loop, handle._volume
        stage_id = handle._stage_id

        def _run():
            from speech_pipeline.AudioReader import AudioReader
            try:
                mixer.wait_source(first_src_id)
                if stop.is_set():
                    mixer.kill_source(first_src_id)
                    return
                mixer.remove_source(first_src_id)
                iters = 1
                max_i = (int(loop) if isinstance(loop, (int, float)) and loop > 1
                         else (999999 if loop else 1))
                while iters < max_i and not stop.is_set() and not mixer.cancelled:
                    reader = AudioReader(url, chunk_seconds=0.5)
                    stage = reader
                    if volume != 100:
                        from speech_pipeline.GainStage import GainStage
                        g = GainStage(reader.output_format.sample_rate,
                                      float(volume) / 100.0)
                        reader.pipe(g)
                        stage = g
                    sid = mixer.add_source(stage)
                    handle._current_src = sid
                    mixer.wait_source(sid)
                    if stop.is_set():
                        mixer.kill_source(sid)
                    else:
                        mixer.remove_source(sid)
                    iters += 1
            except Exception as e:
                _LOGGER.warning("Play %s failed: %s", stage_id, e)
            finally:
                with self._lock:
                    self._stages.pop(stage_id, None)
        threading.Thread(target=_run, daemon=True,
                         name=f"play-{secrets.token_urlsafe(4)}").start()
        _LOGGER.info("Play started: %s (loop=%s)", stage_id, loop)
