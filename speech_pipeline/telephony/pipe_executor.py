"""Generic DSL pipe executor for telephony calls.

Left-to-right parser consumes elements and arrows. Each element is
resolved to a stage/mixer/tee. Wiring follows 3x3 rules based on
output cardinality of left and input cardinality of right:

  Output side: call/tee = multi-output, else single-output
  Input side:  call = multi-input, else single-input

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
from typing import Any, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger("telephony.pipe-executor")


# ---------------------------------------------------------------------------
# DSL Parser — left-to-right regex consumer
# ---------------------------------------------------------------------------

_ARROW = re.compile(r'\s*(->|\|)\s*')
_ELEMENT = re.compile(r'\s*([a-z_]+)(?::([^{\s|>]+))?')
# JSON block: { ... } with balanced braces
def _consume_json(s: str, pos: int) -> Tuple[dict, int]:
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
    """Parse DSL string into [(type, id, params), ...]."""
    elements = []
    s = dsl.strip()
    pos = 0
    while pos < len(s):
        # skip arrows
        m = _ARROW.match(s, pos)
        if m:
            pos = m.end()
            continue
        # match element
        m = _ELEMENT.match(s, pos)
        if not m:
            pos += 1
            continue
        typ, elem_id = m.group(1), m.group(2) or ""
        pos = m.end()
        params, pos = _consume_json(s, pos)
        elements.append((typ, elem_id, params))
    return elements


# ---------------------------------------------------------------------------
# Stage factories
# ---------------------------------------------------------------------------

def _is_multi_output(typ: str) -> bool:
    """call and tee can have multiple outputs."""
    return typ in ("call", "tee")

def _is_multi_input(typ: str) -> bool:
    """call (conference mixer) can have multiple inputs."""
    return typ == "call"


class CallPipeExecutor:
    """Generic DSL pipe executor. Left-to-right wiring via pipe()/ConferenceLeg/AudioTee."""

    def __init__(self, call, tts_registry=None, subscriber=None):
        self.call = call
        self._tts_registry = tts_registry
        self._subscriber = subscriber
        self._stages: Dict[str, Any] = {}  # stage_id -> handle
        self._tees: Dict[str, Any] = {}    # tee_id -> AudioTee instance
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
        if stop: stop.set()
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

    # -- Generic pipe execution --------------------------------------------

    def _execute_pipe(self, elements: List[Tuple[str, str, dict]]):
        """Walk elements left-to-right, wire each pair."""
        if not elements:
            return

        # Resolve first element
        prev_typ, prev_id, prev_params = elements[0]
        prev_node = self._resolve(prev_typ, prev_id, prev_params)

        # For source-only elements (first in chain), create the source stage
        source_stage = self._make_source(prev_typ, prev_id, prev_params, prev_node)

        for i in range(1, len(elements)):
            cur_typ, cur_id, cur_params = elements[i]
            cur_node = self._resolve(cur_typ, cur_id, cur_params)
            is_last = (i == len(elements) - 1)

            # Wire prev → cur based on cardinality
            source_stage = self._wire(
                prev_typ, prev_id, prev_params, prev_node, source_stage,
                cur_typ, cur_id, cur_params, cur_node, is_last)

            prev_typ, prev_id, prev_params, prev_node = cur_typ, cur_id, cur_params, cur_node

    # -- Resolution --------------------------------------------------------

    def _resolve(self, typ: str, elem_id: str, params: dict):
        """Resolve element to its underlying object (mixer, tee, leg, etc.)."""
        if typ == "call":
            return self.call.mixer
        elif typ == "tee":
            return self._get_or_create_tee(elem_id)
        elif typ == "sip":
            return self._resolve_sip(elem_id, params)
        elif typ == "play":
            return None  # handled in _make_source
        elif typ == "tts":
            return None  # handled in _make_source
        elif typ == "stt":
            return self._make_stt(elem_id, params)
        elif typ == "webhook":
            return self._make_webhook(elem_id, params)
        else:
            raise ValueError(f"Unknown element type: {typ}")

    def _get_or_create_tee(self, tee_id: str):
        with self._lock:
            if tee_id in self._tees:
                return self._tees[tee_id]
        from speech_pipeline.AudioTee import AudioTee
        tee = AudioTee(48000, "s16le")
        with self._lock:
            self._tees[tee_id] = tee
        return tee

    def _resolve_sip(self, leg_id: str, params: dict):
        """Resolve SIP leg — returns session object."""
        from . import leg as leg_mod
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
        session = getattr(leg, '_sip_session', None)
        if not session:
            session = leg_mod._CallSession(leg.voip_call)
        return {"session": session, "leg": leg, "leg_id": leg_id}

    def _make_stt(self, lang: str, params: dict):
        lang = lang or "de"
        from speech_pipeline.WhisperSTT import WhisperTranscriber
        return WhisperTranscriber(
            model_name=params.get("model", "base"),
            language=lang)

    def _make_webhook(self, url: str, params: dict):
        from speech_pipeline.WebhookSink import WebhookSink
        bearer = self._subscriber.get("bearer_token", "") if self._subscriber else ""
        return WebhookSink(url, bearer_token=bearer)

    # -- Source creation ---------------------------------------------------

    def _make_source(self, typ, elem_id, params, node):
        """Create a source Stage for the first element in a pipe."""
        if typ == "sip":
            from speech_pipeline.SIPSource import SIPSource
            return SIPSource(node["session"])
        elif typ == "tee":
            # tee as first element = existing tee, next elements are sidechains
            return None  # handled in _wire
        elif typ == "call":
            return None  # call as source = add_output, handled in _wire
        elif typ == "play":
            return self._start_play(elem_id, params)
        elif typ == "tts":
            return self._start_tts(elem_id, params)
        else:
            return None

    # -- Wiring (the 3x3 matrix) ------------------------------------------

    def _wire(self, l_typ, l_id, l_params, l_node, source_stage,
              r_typ, r_id, r_params, r_node, is_last):
        """Wire left → right. Returns the new source_stage for the next pair."""

        # -- Left is tee (multi-output): add sidechain --
        if l_typ == "tee":
            tee = l_node
            if source_stage is None:
                # tee as first element: add sidechain from existing tee
                sink = self._make_sink(r_typ, r_id, r_params, r_node, is_last)
                if sink:
                    tee.add_sidechain(sink)
                    _LOGGER.info("Tee %s: added sidechain %s", l_id, r_typ)
                return None
            else:
                # tee in middle of chain: pipe source into tee, continue
                source_stage.pipe(tee)
                return tee  # tee passes through to next element

        # -- Right is call (multi-input via ConferenceLeg) --
        if r_typ == "call":
            mixer = r_node
            if is_last and source_stage:
                # source -> call (last): add_source to mixer
                src_id = mixer.add_source(source_stage)
                # Start play loop if this is a play source
                play_handle = getattr(source_stage, '_play_handle', None)
                if play_handle:
                    play_handle._current_src = src_id
                    self._start_play_loop(play_handle, src_id)
                return None
            elif source_stage:
                # source -> call -> ... : ConferenceLeg (bidirectional)
                from speech_pipeline.ConferenceLeg import ConferenceLeg
                conf_leg = ConferenceLeg(sample_rate=mixer.sample_rate)
                conf_leg.attach(mixer)
                source_stage.pipe(conf_leg)
                if l_typ == "sip":
                    l_node["leg"]._conf_leg = conf_leg
                return conf_leg
            else:
                # call as source (no left): not supported in this context
                return None

        # -- Right is sip (sink if last) --
        if r_typ == "sip" and is_last:
            from speech_pipeline.SIPSink import SIPSink
            sink = SIPSink(r_node["session"])
            if source_stage:
                source_stage.pipe(sink)
            # Start pipeline + monitors
            self._start_sip_pipeline(sink, r_node, l_typ)
            return None

        # -- Right is stt/webhook (processor/sink) --
        if r_typ in ("stt", "webhook"):
            stage = r_node
            if source_stage:
                source_stage.pipe(stage)
            if is_last and hasattr(stage, 'run'):
                # Terminal sink — start in thread
                threading.Thread(target=stage.run, daemon=True,
                                 name=f"sink-{r_typ}").start()
            return stage

        # -- Right is tee (single input) --
        if r_typ == "tee":
            tee = r_node
            if source_stage:
                source_stage.pipe(tee)
            return tee

        _LOGGER.warning("Unhandled wire: %s -> %s", l_typ, r_typ)
        return source_stage

    # -- Sink creation -----------------------------------------------------

    def _make_sink(self, typ, elem_id, params, node, is_last):
        """Create a sink Stage for sidechain attachment."""
        if typ == "stt":
            cb_url = params.get("on_receive", "")
            if cb_url:
                from speech_pipeline.WebhookSink import WebhookSink
                bearer = self._subscriber.get("bearer_token", "") if self._subscriber else ""
                webhook = WebhookSink(cb_url, bearer_token=bearer)
                node.pipe(webhook)
                return node  # stt piped to webhook, return stt as sidechain root
            return node
        elif typ == "webhook":
            return node
        return node

    # -- SIP pipeline + monitors -------------------------------------------

    def _start_sip_pipeline(self, sink, sip_info, prev_typ):
        """Start SIPSink in background + DTMF/hangup monitors."""
        from . import leg as leg_mod
        leg = sip_info["leg"]
        leg_id = sip_info["leg_id"]
        session = sip_info["session"]
        mixer = self.call.mixer

        def _run():
            try:
                sink.run()
            except Exception as e:
                _LOGGER.warning("Leg %s pipeline error: %s", leg_id, e)
        threading.Thread(target=_run, daemon=True, name=f"leg-{leg_id}").start()

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
        threading.Thread(target=_dtmf, daemon=True, name=f"dtmf-{leg_id}").start()

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
            if hasattr(leg, '_conf_leg') and leg._conf_leg:
                try: leg._conf_leg.cancel()
                except: pass
            self.call.unregister_participant(leg.leg_id)
            leg_mod._fire_callback(leg, "completed", duration=dur)
            _LOGGER.info("Leg %s ended (%.1fs)", leg_id, dur)
            leg_mod.delete_leg(leg_id)
        threading.Thread(target=_monitor, daemon=True, name=f"mon-{leg_id}").start()

        _LOGGER.info("SIP leg %s wired into call %s", leg_id, self.call.call_id)

    # -- Play --------------------------------------------------------------

    def _start_play(self, play_id, params):
        """Create first AudioReader and register play handle.
        Returns the Stage for the first iteration. Loop continues in _wire."""
        from speech_pipeline.AudioReader import AudioReader
        url = params.get("url", play_id)
        if not url:
            raise ValueError("play requires url")
        volume = params.get("volume", 100)
        stage_id = f"play:{play_id or secrets.token_urlsafe(6)}"

        reader = AudioReader(url, chunk_seconds=0.5)
        source_stage = reader
        if volume != 100:
            from speech_pipeline.GainStage import GainStage
            g = GainStage(reader.output_format.sample_rate, float(volume) / 100.0)
            reader.pipe(g)
            source_stage = g

        # Store handle for kill_stage + loop
        stop_event = threading.Event()
        handle = type('_H', (), {
            '_stop': stop_event, '_current_src': None,
            '_url': url, '_loop': params.get("loop", False),
            '_volume': volume, '_stage_id': stage_id,
        })()
        with self._lock:
            self._stages[stage_id] = handle

        # Tag the source so _wire can find the handle
        source_stage._play_handle = handle
        return source_stage

    def _start_play_loop(self, handle, first_src_id):
        """Wait for first iteration to finish, then loop if needed."""
        mixer = self.call.mixer
        stop = handle._stop
        url, loop, volume = handle._url, handle._loop, handle._volume
        stage_id = handle._stage_id

        def _run():
            from speech_pipeline.AudioReader import AudioReader
            try:
                # Wait for first iteration (already added to mixer)
                mixer.wait_source(first_src_id)
                if stop.is_set():
                    mixer.kill_source(first_src_id)
                    return
                mixer.remove_source(first_src_id)

                # Loop remaining
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

    # -- TTS ---------------------------------------------------------------

    def _start_tts(self, voice_name, params):
        """Create TTSProducer stage."""
        voice_name = voice_name or "de_DE-thorsten-medium"
        text = params.get("text", "")
        if not text:
            raise ValueError("tts requires text param")
        if not self._tts_registry:
            raise ValueError("TTS not available")
        from speech_pipeline.TTSProducer import TTSProducer
        voice = self._tts_registry.ensure_loaded(voice_name)
        syn = self._tts_registry.create_synthesis_config(voice, {})
        return TTSProducer(voice, syn, text, sentence_silence=0.0)
