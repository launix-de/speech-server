"""Generic DSL pipe executor for telephony calls.

Left-to-right chain. Normal stages are connected via pipe().
Special substitutions:
- call:ID → ConferenceLeg (single, bidirectional) OR ConferenceSource/
  ConferenceSink pair (two occurrences → coupled via _Coupling)
- tee:ID as first element + already exists → add_sidechain mode
- sip/originate/codec without ID → reuse first same-type predecessor's session

Element syntax: ``type:id{json_params}``
Separator: ``->`` or ``|``

Invariante: ``call:ID`` = virtueller Leg auf der Konferenz
------------------------------------------------------------
Wenn ``call`` in einer DSL verwendet wird, wird **genau ein** Leg auf
die Konferenz abgezweigt.  Wird derselbe ``call:ID`` **zweimal** in
derselben DSL verwendet, meinen diese beiden Positionen **Input und
Output desselben Legs**:

    sip:A -> call:C -> sip:A       # Bridge aus SIP-Sicht
    call:C -> sip:A -> call:C      # dieselbe Bridge aus Konferenz-Sicht

Beide DSLs beschreiben dieselbe bidirektionale Kopplung — mix-minus
subtrahiert den eigenen Input aus dem Output.

DELETE-Kaskade
--------------
``DELETE sip:ID`` → ``leg.hangup()`` → SIPSource/SIPSink beenden →
``close()`` cascadiert → ConferenceSink/Source schließen → mixer
source drained → idle-timeout → ``_auto_cleanup`` → call weg.

``DELETE call:ID`` → ``call_state.delete_call`` → mixer cancel →
alle Endpoints cancellen → verbundene SIP legs via ``delete_leg``
hangupt.
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from typing import Any, Dict, List, Tuple

from speech_pipeline.dsl_parser import parse_dsl  # noqa: F401 — re-export
from speech_pipeline.telephony import _shared
from urllib.parse import urlparse as _urlparse
from .id_scope import expand_for_account


def _same_origin(url: str, base: str | None) -> bool:
    """True iff *url* shares scheme+host+port with *base*."""
    if not base or not url:
        return False
    try:
        a, b = _urlparse(url), _urlparse(base)
    except Exception:
        return False
    return (a.scheme == b.scheme
            and (a.hostname or "").lower() == (b.hostname or "").lower()
            and a.port == b.port)

_LOGGER = logging.getLogger("telephony.pipe-executor")

_PLAY_CHUNK_SECONDS = 0.05
_PLAY_PREFILL_SECONDS = 0.12
_TTS_CHUNK_SECONDS = 0.10

# ---------------------------------------------------------------------------
# Pipe Executor
# ---------------------------------------------------------------------------

class CallPipeExecutor:
    def __init__(self, call=None, tts_registry=None, subscriber=None, ws=None,
                 account_id=None):
        self.call = call
        self._account_id = account_id or (call.account_id if call else None)
        self._tts_registry = tts_registry
        self._subscriber = subscriber
        self._ws = ws                        # WebSocket connection (for ws: elements)
        self._stages: Dict[str, Any] = {}   # play handles for kill
        self._tees: Dict[str, Any] = {}     # tee_id -> AudioTee
        self._sidechain_specs: set[tuple] = set()
        self._sessions: Dict[str, Any] = {} # typ -> first session (for ID reuse)
        self._lock = threading.Lock()
        self._text_input_queue = None        # set by text_input element
        self._call_couplings: Dict[str, Any] = {}  # call_id -> _Coupling

    def _session_leg(self, elem_id: str):
        leg = self._sessions.get(f"_leg:{elem_id}")
        if leg is not None:
            return leg
        scoped = expand_for_account(elem_id, self._account_id)
        if scoped != elem_id:
            return self._sessions.get(f"_leg:{scoped}")
        return None

    def _session_call(self, elem_id: str):
        session = self._sessions.get(f"sip:{elem_id}")
        if session is not None:
            return session
        scoped = expand_for_account(elem_id, self._account_id)
        if scoped != elem_id:
            return self._sessions.get(f"sip:{scoped}")
        return None

    # -- Public API --------------------------------------------------------

    def add_pipes(self, pipes: List[str]) -> List[dict]:
        results = []
        for pipe_str in pipes:
            try:
                elements = parse_dsl(pipe_str)
                self._last_originated_leg_id = None
                self._last_webclient_result = None
                self._execute_pipe(elements)
                result = {"pipe": pipe_str, "ok": True}
                if self._last_webclient_result:
                    result.update(self._last_webclient_result)
                # Originate:NUM -> call:C returns a leg_id so callers can
                # track the leg via callbacks.
                if self._last_originated_leg_id:
                    result["leg_id"] = self._last_originated_leg_id
                results.append(result)
            except Exception as e:
                _LOGGER.warning("Pipe failed: %s — %s", pipe_str, e, exc_info=True)
                results.append({"pipe": pipe_str, "ok": False, "error": str(e)})
        return results

    def kill_stage(self, stage_id: str) -> bool:
        with self._lock:
            handle = self._stages.pop(stage_id, None)
        if not handle:
            return False
        cleanup = getattr(handle, '_cleanup', None)
        if cleanup:
            try:
                cleanup()
            except Exception:
                pass
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

    def shutdown(self) -> None:
        """Cancel all live pipe resources for this call."""
        with self._lock:
            stage_ids = list(self._stages)
            tees = list(self._tees.values())
            self._tees.clear()
            self._sidechain_specs.clear()
            self._sessions.clear()

        for stage_id in stage_ids:
            self.kill_stage(stage_id)

        for tee in tees:
            try:
                tee.cancel()
            except Exception:
                pass

    # -- Pipe execution ----------------------------------------------------

    # -- Action elements (single-element, no pipe) --------------------------

    _ACTION_TYPES = {"answer", "webclient"}

    def _execute_action(self, typ, elem_id, params) -> bool:
        """Execute a single-element action. Returns True if handled."""
        if typ == "answer":
            if not elem_id:
                raise ValueError("answer requires a leg ID")
            from . import leg as leg_mod
            elem_id = expand_for_account(elem_id, self._account_id)
            leg = leg_mod.get_leg(elem_id)
            if not leg:
                raise ValueError(f"Leg '{elem_id}' not found")
            # Send 200 OK via sip_stack or pyVoIP
            from . import sip_stack
            # Inbound trunk legs store the deferred dialog under
            # ``sip_call_id``. Older code also looked for
            # ``_sip_call_id``; support both so answer:LEG actually
            # reaches ``answer_trunk_leg()`` for real trunk calls.
            sip_call_id = (getattr(leg, 'sip_call_id', '')
                           or getattr(leg, '_sip_call_id', ''))
            if sip_call_id:
                sip_stack.answer_trunk_leg(sip_call_id)
            elif leg.voip_call and hasattr(leg.voip_call, 'answer'):
                leg.voip_call.answer()
            return True

        if typ == "webclient":
            return self._execute_action_webclient(typ, elem_id, params)

        return False

    def _pre_create_originate_leg(self, number: str, params: dict,
                                   call) -> str:
        """Create the outbound leg synchronously (no SIP IO) so the
        HTTP response can return its ``leg_id`` before the async build
        runs.  Used by ``pipeline_api.POST /api/pipelines`` to keep the
        request fast — PHP callers (CRM) need immediate return because
        their session lock blocks follow-up calls.

        Idempotent: subsequent ``_execute_async_originate`` with the
        same number will reuse this leg if we pre-registered it.
        """
        from . import leg as leg_mod
        if call is None:
            raise ValueError("originate requires a call context")
        leg = leg_mod.create_leg(
            "outbound", number, call.pbx_id, call.subscriber_id,
        )
        leg.callbacks = {
            k: v for k, v in (params or {}).items()
            if k in ("ringing", "answered", "completed", "failed",
                     "no-answer", "busy", "canceled")
        }
        leg.call_id = call.call_id
        leg.caller_id = (params or {}).get("caller_id", "")
        # Stash for pickup by _execute_async_originate.
        self._sessions[f"_prealloc_leg:{number}"] = leg
        return leg.leg_id

    def _execute_async_originate(self, originate_elem, call_elem):
        """Async originate: fire-and-forget + auto-bridge on answer.

        DSL: originate:NUM{"answered":"/cb", ...} -> call:CALL_ID
        The CRM receives callbacks and bridges via a second pipeline.
        """
        typ, number, params = originate_elem
        _, call_id, _ = call_elem

        call = self._resolve_call(call_id)
        from . import leg as leg_mod, pbx as pbx_reg, auth as auth_mod

        if not number:
            raise ValueError("originate requires a phone number")
        pbx_entry = pbx_reg.get(call.pbx_id)
        if not pbx_entry:
            raise ValueError(f"PBX {call.pbx_id} not found")
        if call.account_id and call.account_id != "__admin__":
            if not auth_mod.check_pbx_access(call.account_id, call.pbx_id):
                raise ValueError(f"Account not allowed to use PBX {call.pbx_id}")

        # Reuse the leg pre-allocated by ``_pre_create_originate_leg``
        # (called inline from pipeline_api to return leg_id fast).
        leg = self._sessions.pop(f"_prealloc_leg:{number}", None)
        if leg is None:
            leg = leg_mod.create_leg("outbound", number, call.pbx_id,
                                      call.subscriber_id)
            leg.callbacks = {
                k: v for k, v in params.items()
                if k in ("ringing", "answered", "completed", "failed",
                         "no-answer", "busy", "canceled")}
            leg.call_id = call.call_id
            leg.caller_id = params.get("caller_id", "")

        # Return leg_id via a special callback for the CRM to track
        # Alternative: include leg_id in the 'ringing' callback payload.
        # For now, the CRM gets leg_id in the callback body (leg_id field).

        threading.Thread(
            target=leg_mod.originate_only,
            args=(leg, pbx_entry),
            daemon=True,
            name=f"orig-{leg.leg_id}",
        ).start()
        # Expose leg_id to ``add_pipes`` / HTTP response.
        self._last_originated_leg_id = leg.leg_id
        return True

    def _execute_action_webclient(self, typ, elem_id, params) -> bool:
        if typ == "webclient":
            if not self.call:
                raise ValueError("webclient requires a call context")
            from . import auth, webclient as wc
            user = elem_id or "anonymous"
            callback = params.get("callback", "")
            base_url = params.get("base_url", "")
            if not base_url:
                raise ValueError("webclient requires base_url")
            base_url = wc.normalize_base_url(base_url)

            forbidden = [
                key for key in ("pipes", "dsl", "stt_callback", "transcript")
                if params.get(key)
            ]
            if forbidden:
                raise ValueError(
                    "webclient only creates a slot; build codec/webhook "
                    "pipes separately after the answered callback"
                )

            if not auth.check_feature(self.call.account_id, "webclient"):
                raise ValueError("Account lacks webclient feature")

            nonce_entry = auth.create_nonce(
                account_id=self.call.account_id,
                subscriber_id=self.call.subscriber_id,
                user=user)
            nonce = nonce_entry["nonce"]

            sess = wc.register_webclient(self.call, user, nonce)
            session_id = sess["session_id"]

            self.call.register_participant(session_id, type="webclient",
                                           user=user, nonce=nonce,
                                           callback=callback)

            iframe_url = f"{base_url}/phone/{nonce}"
            _LOGGER.info("WebClient slot for call %s: %s",
                         self.call.call_id, iframe_url)
            sess["iframe_url"] = iframe_url

            # Surface identifiers to the HTTP response so the CRM can
            # store session_id as the participant's sid and embed the
            # iframe_url immediately — no need to wait for the async
            # ``ready`` callback.
            self._last_webclient_result = {
                "session_id": session_id,
                "nonce": nonce,
                "iframe_url": iframe_url,
            }

            # Do not fire the CRM callback here. The CRM treats
            # ``state=webclient`` as "browser actually joined", so an
            # eager webhook marks the participant answered before the
            # popup exists or the codec socket is connected.
            return True

        return False

    def _execute_pipe(self, elements):
        if not elements:
            return

        # Check for single-element actions (answer, webclient)
        if len(elements) == 1:
            typ, elem_id, params = elements[0]
            if typ in self._ACTION_TYPES:
                self._execute_action(typ, elem_id, params)
                return

        # Async originate: `originate:NUM{cb} -> call:CALL` (2 elements).
        # Fires SIP INVITE in background, callbacks fire on state changes.
        # CRM receives 'answered' callback and bridges via a second pipe.
        if (len(elements) == 2
                and elements[0][0] == "originate"
                and elements[1][0] in ("call", "conference")):
            self._execute_async_originate(elements[0], elements[1])
            return

        # Phase 1: fill in missing IDs (sip without ID = first sip predecessor)
        self._fill_ids(elements)
        self._validate_elements(elements)

        # Phase 2: existing tee as first element → sidechain mode
        first_typ, first_id, first_params = elements[0]
        if first_typ == "tee" and first_id in self._tees:
            self._add_sidechain(elements)
            return

        # Phase 3: resolve all elements to Stage objects
        resolved = []
        for typ, elem_id, params in elements:
            stage = self._create_stage(typ, elem_id, params)
            resolved.append((typ, elem_id, params, stage))

        # Phase 3b: resolve streaming TTS placeholders.
        # If tts has no fixed text, it needs an upstream text source
        # (text_input or similar). Wire them via StreamingTTSProducer.
        for i, (typ, elem_id, params, stage) in enumerate(resolved):
            if not getattr(stage, '_streaming', False):
                continue
            if i == 0:
                raise ValueError("Streaming tts requires an upstream text source")
            # Find the upstream stage and create a text iterator from it
            _, _, _, upstream = resolved[i - 1]
            def _text_iter_from_stage(src):
                for chunk in src.stream_pcm24k():
                    if isinstance(chunk, bytes):
                        yield chunk.decode("utf-8", errors="replace")
                    else:
                        yield str(chunk)
            from speech_pipeline.StreamingTTSProducer import StreamingTTSProducer
            tts_stage = StreamingTTSProducer(
                _text_iter_from_stage(upstream),
                stage.voice,
                stage.syn,
            )
            resolved[i] = (typ, elem_id, params, tts_stage)

        # Phase 4a: substitute duplicate ``call:ID`` positions with
        # ConferenceSource/ConferenceSink coupled via a shared
        # ``_Coupling`` — this is how ``call:C -> sip:A -> call:C``
        # gets the same mix-minus wiring as ``sip:A -> call:C -> sip:A``.
        resolved = self._wrap_calls(resolved)

        # Phase 4b: wrap SIP sessions — returns list of sub-chains (one
        # per connected piping segment; a middle `sip:L` splits).
        chains = self._wrap_sip(resolved)

        from speech_pipeline.ConferenceLeg import ConferenceLeg as _CL
        from speech_pipeline.StreamingTTSProducer import StreamingTTSProducer as _STTS
        from speech_pipeline.SIPSink import SIPSink

        for chain in chains:
            # Phase 5: link play handles to their terminal stage (for kill_stage)
            for i in range(len(chain) - 1):
                _, _, _, l_stage = chain[i]
                _, _, _, r_stage = chain[i + 1]
                play_handle = getattr(l_stage, '_play_handle', None)
                if play_handle and isinstance(r_stage, (_CL, SIPSink)):
                    play_handle._stage = r_stage

            # Phase 6: chain via pipe()
            for i in range(len(chain) - 1):
                l_typ, _, _, l_stage = chain[i]
                _, _, _, r_stage = chain[i + 1]
                if isinstance(r_stage, _CL) and i + 1 == len(chain) - 1:
                    continue  # source → terminal ConferenceLeg uses mixer.add_source
                if isinstance(r_stage, _STTS):
                    continue  # StreamingTTS reads text directly from its iterator
                l_stage.pipe(r_stage)

            self._register_bridge_handles(chain)

            # Phase 7: start terminal + monitors
            self._start_all(chain)

    # -- Fill missing IDs --------------------------------------------------

    def _fill_ids(self, elements):
        """Fill in missing IDs: sip without ID = reuse first predecessor."""
        seen = {}
        for i, (typ, elem_id, params) in enumerate(elements):
            if typ != "sip":
                continue
            if elem_id:
                seen.setdefault(typ, elem_id)
            elif typ in seen:
                elements[i] = (typ, seen[typ], params)

    def _validate_elements(self, elements):
        """Reject malformed or ambiguous telephony DSL before dispatch."""
        if not elements:
            raise ValueError("Empty pipe")

        call_positions = [i for i, (typ, _, _) in enumerate(elements) if typ == "call"]
        call_ids = [elements[i][1] for i in call_positions]
        if len(call_positions) > 2:
            raise ValueError("At most two call stages are allowed per pipe")
        if len(call_positions) == 2 and call_ids[0] != call_ids[1]:
            raise ValueError(
                "Two call stages in one pipe must reference the same call id "
                "(input and output of the same virtual leg)")

        webhook_positions = [i for i, (typ, _, _) in enumerate(elements) if typ == "webhook"]
        if webhook_positions and webhook_positions[-1] != len(elements) - 1:
            raise ValueError("webhook must be the final stage in a pipe")

        first_typ, first_id, _ = elements[0]
        sidechain_mode = first_typ == "tee" and first_id in self._tees
        if sidechain_mode:
            if len(elements) < 2:
                raise ValueError("tee sidechain requires at least one downstream stage")
            for typ, _, _ in elements[1:]:
                if typ in {"call", "sip", "play", "tts"}:
                    raise ValueError(f"{typ} is not allowed inside tee sidechains")
            return
        if first_typ == "tee":
            raise ValueError("tee may only start a pipe when attaching to an existing tee sidechain")

        sip_positions = [(i, elem_id) for i, (typ, elem_id, _) in enumerate(elements) if typ in ("sip", "originate")]
        sip_ids = [elem_id for _, elem_id in sip_positions]
        if len(sip_ids) > 2:
            raise ValueError("At most two sip stages are allowed per pipe")
        if len(sip_ids) == 2 and sip_ids[0] != sip_ids[1]:
            raise ValueError("Bidirectional sip pipes must use the same leg on both sides")
        if sip_ids and not call_positions:
            if len(elements) < 2:
                raise ValueError("sip requires at least one neighbouring stage")
            # Each sip:LEG position must have a neighbour — `_wrap_sip`
            # decides Source/Sink based on left/right neighbour.  The
            # legacy "terminal sink only" rule is intentionally dropped
            # so that patterns like `play -> sip:L -> record`, `sip:L
            # -> record`, and `sip:L -> hangup` all work.
            return

        # Treat 'conference' like 'call' for validation (alias).
        conf_positions = [i for i, (typ, _, _) in enumerate(elements)
                          if typ == "conference"]
        all_call_positions = call_positions + conf_positions
        if call_positions and conf_positions:
            raise ValueError("Use either 'call' or 'conference', not both in the same pipe")

        if call_positions:
            for cp in call_positions:
                cid = elements[cp][1]
                if not cid:
                    raise ValueError("call requires a call id")
                if self.call and expand_for_account(cid, self._account_id) != self.call.call_id:
                    raise ValueError(
                        f"Pipe targets call {cid}, but executor is bound to {self.call.call_id}"
                    )
            # Single-call legacy rules (one call only): first-element
            # block + layout constraints. Two-call (coupled) form is
            # exempt — the wrap phase handles it.
            if len(call_positions) == 1:
                call_pos = call_positions[0]
                if call_pos == 0:
                    raise ValueError("call stage cannot be the first element")
                if call_pos != len(elements) - 1:
                    if call_pos != len(elements) - 2 or elements[-1][0] not in ("sip", "codec"):
                        raise ValueError("call may only be followed by a single terminal sip/codec stage")
                if sip_ids and len(sip_ids) == 1 and call_pos != len(elements) - 2:
                    raise ValueError("A single sip stage is only valid as terminal sink after call")

    # -- Sidechain: tee:ID -> stage -> stage -> ...  -------------------------

    def _add_sidechain(self, elements):
        """Existing tee:ID as first element — add remaining as sidechain."""
        tee_id = elements[0][1]
        tee = self._tees.get(tee_id)
        if tee and getattr(tee, "cancelled", False):
            with self._lock:
                self._tees.pop(tee_id, None)
                self._sidechain_specs = {
                    spec for spec in self._sidechain_specs if spec[0] != tee_id
                }
            tee = None
        if not tee:
            raise ValueError(f"Tee {tee_id} not found")
        signature = (
            tee_id,
            tuple(
                (typ, elem_id, json.dumps(params, sort_keys=True))
                for typ, elem_id, params in elements[1:]
            ),
        )
        with self._lock:
            if signature in self._sidechain_specs:
                _LOGGER.info("Tee %s: sidechain already attached", tee_id)
                return

        # Build sidechain chain: create stages, pipe() them
        chain = []
        for typ, elem_id, params in elements[1:]:
            stage = self._create_stage(typ, elem_id, params)
            chain.append(stage)
        for i in range(len(chain) - 1):
            chain[i].pipe(chain[i + 1])

        if chain:
            tee.add_sidechain(chain[0])
            with self._lock:
                self._sidechain_specs.add(signature)
            _LOGGER.info("Tee %s: sidechain added (%d stages)", tee_id, len(chain))

    # -- Stage creation ----------------------------------------------------

    def _resolve_call(self, call_id: str):
        """Resolve a call by ID — uses self.call if bound, otherwise registry."""
        call_id = expand_for_account(call_id, self._account_id)
        if self.call and (not call_id or call_id == self.call.call_id):
            return self.call
        from . import call_state
        call = call_state.get_call(call_id)
        if not call:
            raise ValueError(f"Call {call_id} not found")
        return call

    def _create_stage(self, typ, elem_id, params):
        # -- call / conference: substitute with ConferenceLeg --
        if typ == "call":
            call = self._resolve_call(elem_id)
            from speech_pipeline.ConferenceLeg import ConferenceLeg
            conf_leg = ConferenceLeg(sample_rate=call.mixer.sample_rate)
            conf_leg.attach(call.mixer)
            return conf_leg

        # -- tee: get or create --
        if typ == "tee":
            with self._lock:
                existing = self._tees.get(elem_id)
                if existing and getattr(existing, "cancelled", False):
                    self._tees.pop(elem_id, None)
                    self._sidechain_specs = {
                        spec for spec in self._sidechain_specs if spec[0] != elem_id
                    }
                    existing = None
                if existing:
                    return existing
            from speech_pipeline.AudioTee import AudioTee
            tee = AudioTee(48000, "s16le")
            with self._lock:
                self._tees[elem_id] = tee
            return tee

        # -- originate: dial outbound, create SIP leg --
        if typ == "originate":
            return self._create_originate(elem_id, params)

        # -- sip: resolve leg session, create Source or Sink --
        if typ == "sip":
            return self._create_sip(elem_id, params)

        # -- play: AudioReader with loop handle --
        if typ == "play":
            return self._create_play(elem_id, params)

        # -- tts: TTSProducer --
        if typ == "tts":
            return self._create_tts(elem_id, params)

        if typ == "stt":
            lang = elem_id or "de"
            from speech_pipeline.WhisperSTT import WhisperTranscriber
            return WhisperTranscriber(model_size=params.get("model", "small"),
                                      language=lang)

        if typ == "webhook":
            if not elem_id:
                raise ValueError("webhook requires a target URL")
            # Trust the subscriber's own origin.  ``post_webhook`` already
            # posts to subscriber.base_url unchecked; WebhookSink URLs
            # derived from that same origin (e.g. STT transcript tap with
            # a relative path that got prefixed by base_url) must pass
            # too — otherwise the callbacks that worked during the same
            # request die here.
            sub_base = self._subscriber.get("base_url") if self._subscriber else None
            if not _same_origin(elem_id, sub_base):
                from speech_pipeline.url_safety import require_safe_url
                require_safe_url(elem_id)
            from speech_pipeline.WebhookSink import WebhookSink
            bearer = self._subscriber.get("bearer_token", "") if self._subscriber else ""
            return WebhookSink(elem_id, bearer_token=bearer)

        # -- conference: alias for call --
        if typ == "conference":
            call = self._resolve_call(elem_id)
            from speech_pipeline.ConferenceLeg import ConferenceLeg
            conf_leg = ConferenceLeg(sample_rate=call.mixer.sample_rate)
            conf_leg.attach(call.mixer)
            return conf_leg

        # -- gain: volume adjustment --
        if typ == "gain":
            from speech_pipeline.GainStage import GainStage
            factor = float(elem_id) if elem_id else params.get("factor", 1.0)
            rate = params.get("rate", 48000)
            return GainStage(rate, float(factor))

        # -- delay: delay line --
        if typ == "delay":
            from speech_pipeline.DelayLine import DelayLine
            ms = float(elem_id) if elem_id else params.get("ms", 0.0)
            rate = params.get("rate", 48000)
            return DelayLine(rate, float(ms))

        # -- pitch: pitch adjustment --
        if typ == "pitch":
            from speech_pipeline.PitchAdjuster import PitchAdjuster
            st = float(elem_id) if elem_id else params.get("semitones", 0.0)
            return PitchAdjuster("", pitch_disable=(abs(st) < 0.05),
                                 pitch_override_st=st, correction=1.0)

        # -- vc: voice conversion --
        if typ == "vc":
            ref_url = params.get("url", "")
            if not ref_url:
                raise ValueError("vc requires a url parameter")
            from speech_pipeline.VCConverter import VCConverter
            from speech_pipeline.media_refs import resolve_media_ref
            ref = resolve_media_ref(ref_url, getattr(_shared, "media_folder", None))
            bearer = params.get("bearer", "")
            return VCConverter(ref, bearer=bearer)

        # -- codec: webclient Fourier-codec WS endpoint --
        # Position-aware: if first → CodecSocketSource, last →
        # CodecSocketSink, both → bidirectional handled by the
        # CodecLeg wrapper analogue.  Same dual-position pattern as
        # ``sip:LEG``.
        if typ == "codec":
            if not elem_id:
                raise ValueError("codec requires a session id (e.g. codec:wc-abc)")
            elem_id = expand_for_account(elem_id, self._account_id)
            from speech_pipeline.CodecSocketSession import (
                CodecSocketSession, get_session,
            )
            session = get_session(elem_id)
            if session is None:
                session = CodecSocketSession(elem_id)
            # Stash the session in self._sessions so _wrap_codec sees
            # it and can build Source/Sink wrappers.
            self._sessions[f"codec:{elem_id}"] = session
            return session

        # -- record/save: managed file recording with download URL --
        if typ in ("record", "save"):
            return self._create_save(elem_id, params)

        # -- hangup: sink that immediately closes upstream (triggers BYE/CANCEL) --
        if typ == "hangup":
            from speech_pipeline.HangupSink import HangupSink
            return HangupSink()

        # -- text_input: queue-backed text source for streaming TTS --
        if typ == "text_input":
            import queue as _queue
            q = _queue.Queue()
            self._text_input_queue = q
            from speech_pipeline.base import AudioFormat, Stage

            class _TextInputSource(Stage):
                """Yields text strings from a queue."""
                def __init__(self, q):
                    super().__init__()
                    self._q = q
                    self.output_format = AudioFormat(0, "text")
                def stream_pcm24k(self):
                    while not self.cancelled:
                        item = self._q.get()
                        if item is None:
                            break
                        yield item.encode("utf-8") if isinstance(item, str) else item

            return _TextInputSource(q)

        raise ValueError(f"Unknown element type: {typ}")

    def _create_sip(self, leg_id, params):
        """Resolve SIP leg. Returns session (SIPSource/SIPSink wrap via pipe)."""
        from . import leg as leg_mod

        if not leg_id:
            raise ValueError("sip requires a leg id")
        leg_id = expand_for_account(leg_id, self._account_id)

        leg = leg_mod.get_leg(leg_id)
        if not leg:
            raise ValueError(f"Leg {leg_id} not found")
        if not leg.voip_call:
            raise ValueError(f"Leg {leg_id} has no SIP call")

        if params:
            leg.callbacks.update(params)

        if self.call:
            leg.call_id = self.call.call_id
            leg.status = "in-progress"
            leg.answered_at = leg.answered_at or time.time()
            self.call.register_participant(leg.leg_id, type="sip",
                                           direction=leg.direction,
                                           number=leg.number)

        # Check if we already have a session for this leg
        session_key = f"sip:{leg_id}"
        if session_key in self._sessions:
            self._sessions[f"_leg:{leg_id}"] = leg
            return self._sessions[session_key]

        session = getattr(leg, 'sip_session', None) or leg_mod.PyVoIPCallSession(leg.voip_call)
        self._sessions[session_key] = session
        self._sessions[f"_leg:{leg_id}"] = leg
        return session

    def _create_play(self, play_id, params):
        from speech_pipeline.AudioReader import AudioReader
        from speech_pipeline.media_refs import resolve_media_ref
        src_ref = params.get("url", "")
        if not src_ref:
            raise ValueError("play requires a url parameter")
        url = resolve_media_ref(src_ref, getattr(_shared, "media_folder", None))
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

        reader = AudioReader(
            url,
            chunk_seconds=_PLAY_CHUNK_SECONDS,
            realtime=True,
            prefill_seconds=_PLAY_PREFILL_SECONDS,
        )
        source = reader
        if volume != 100:
            from speech_pipeline.GainStage import GainStage
            g = GainStage(reader.output_format.sample_rate, float(volume) / 100.0)
            reader.pipe(g)
            source = g
        source._play_handle = handle
        return source

    def _create_originate(self, number, params):
        """Originate an outbound SIP call. Returns session (like _create_sip).

        DSL: originate:+49170...{"ringing":"/cb","answered":"/cb","completed":"/cb"}
        Bidirectional: originate:NUM{...} -> call:X -> originate:NUM
        """
        if not number:
            raise ValueError("originate requires a phone number")
        if not self.call:
            raise ValueError("originate requires a call context (use call:ID in the DSL)")

        from . import leg as leg_mod, pbx as pbx_reg, auth as auth_mod

        # Resolve PBX from call
        pbx_entry = pbx_reg.get(self.call.pbx_id)
        if not pbx_entry:
            raise ValueError(f"PBX {self.call.pbx_id} not found")

        # PBX access check
        if self.call.account_id and self.call.account_id != "__admin__":
            if not auth_mod.check_pbx_access(self.call.account_id, self.call.pbx_id):
                raise ValueError(f"Account not allowed to use PBX {self.call.pbx_id}")

        # Create leg
        leg = leg_mod.create_leg("outbound", number, self.call.pbx_id,
                                  self.call.subscriber_id)
        leg.callbacks = {k: v for k, v in params.items()
                         if k in ("ringing", "answered", "completed", "failed",
                                  "no-answer", "busy", "canceled", "caller_id")}
        leg.call_id = self.call.call_id
        leg.caller_id = params.get("caller_id", "")

        # Originate in background — fires callbacks, waits for answer
        # On answer, the leg gets a sip_session that SIPSource/SIPSink can use.
        leg_mod.originate_only(leg, pbx_entry)

        # Wait for answer (up to 30s)
        deadline = time.time() + 30
        while leg.status not in ("answered", "in-progress", "failed", "completed"):
            if time.time() > deadline:
                raise ValueError(f"Originate to {number} timed out")
            time.sleep(0.2)

        if leg.status in ("failed", "completed"):
            raise ValueError(f"Originate to {number} failed (status={leg.status})")

        # Leg is answered — get session for SIPSource/SIPSink wrapping
        session = leg.sip_session
        if not session:
            raise ValueError(f"Leg {leg.leg_id} answered but no SIP session")

        # Store in sessions cache (same as _create_sip)
        session_key = f"sip:{leg.leg_id}"
        self._sessions[session_key] = session
        self._sessions[f"_leg:{leg.leg_id}"] = leg

        leg.status = "in-progress"
        leg.answered_at = leg.answered_at or time.time()
        self.call.register_participant(leg.leg_id, type="sip",
                                       direction="outbound",
                                       number=number)
        return session

    def _create_save(self, save_id, params):
        """Create a managed file recording with download URL.

        Writes to a safe managed directory. When recording finishes,
        fires a webhook with the download URL (if ``completed`` param set).

        DSL: ``save:name{"completed":"/callback/path"}``
        """
        import os
        import tempfile

        ext = params.get("format", "wav")
        if ext not in ("wav", "mp3", "ogg", "flac"):
            ext = "wav"

        # Managed save directory — no user-controlled paths
        save_dir = os.path.join(tempfile.gettempdir(), "speech-pipeline-saves")
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{save_id or secrets.token_urlsafe(8)}.{ext}"
        # Sanitize: strip path separators
        filename = filename.replace("/", "_").replace("..", "_")
        filepath = os.path.join(save_dir, filename)

        rate = int(params.get("rate", 48000))
        from speech_pipeline.AudioTee import AudioTee
        from speech_pipeline.FileRecorder import FileRecorder

        tee = AudioTee(rate, "s16le")
        recorder = FileRecorder(filepath, rate)
        tee.add_sidechain(recorder)

        # Store metadata for download URL generation
        tee._save_filepath = filepath
        tee._save_filename = filename
        tee._save_completed = params.get("completed", "")

        return tee

    def _create_tts(self, voice_name, params):
        if voice_name:
            raise ValueError("tts voice must be passed via JSON params")
        voice_name = params.get("voice", "") or "de_DE-thorsten-medium"
        if not self._tts_registry:
            raise ValueError("TTS not available")
        voice = self._tts_registry.ensure_loaded(voice_name)
        syn = self._tts_registry.create_synthesis_config(voice, {})

        text = params.get("text", "")
        if text:
            # Fixed text mode: TTSProducer with known text
            from speech_pipeline.TTSProducer import TTSProducer
            return TTSProducer(
                voice,
                syn,
                text,
                sentence_silence=0.0,
                chunk_seconds=_TTS_CHUNK_SECONDS,
            )
        else:
            # Streaming mode: will be wired to upstream text source.
            # Mark as streaming — _execute_pipe will connect the iterator.
            from speech_pipeline.StreamingTTSProducer import StreamingTTSProducer
            placeholder = type("_StreamingTTSPlaceholder", (), {
                "voice": voice,
                "syn": syn,
                "_streaming": True,
            })()
            return placeholder

    # -- Call coupling ------------------------------------------------------

    def _get_call_coupling(self, call_id: str):
        """Return the shared ``_Coupling`` for a call id (lazy-create)."""
        from speech_pipeline.ConferenceEndpoint import _Coupling
        coupling = self._call_couplings.get(call_id)
        if coupling is None:
            call = self._resolve_call(call_id)
            coupling = _Coupling(call.mixer)
            self._call_couplings[call_id] = coupling
        return coupling

    def _wrap_calls(self, resolved):
        """Substitute duplicate ``call:ID`` positions with coupled endpoints.

        Invariant: a ``call:ID`` appearing twice in the same DSL
        represents **input and output of the same virtual leg** on the
        conference — mix-minus couples them automatically.

        Single-occurrence ``call:ID`` is left as ``ConferenceLeg``
        (bidirectional) or is handled by the existing terminal-mixer
        path in ``_start_all``.
        """
        from speech_pipeline.ConferenceEndpoint import ConferenceSource, ConferenceSink

        call_counts: dict[str, int] = {}
        for typ, elem_id, _, _ in resolved:
            if typ in ("call", "conference"):
                call_counts[elem_id] = call_counts.get(elem_id, 0) + 1

        seen: set[str] = set()
        out = []
        for typ, elem_id, params, stage in resolved:
            if typ not in ("call", "conference") or call_counts.get(elem_id, 0) < 2:
                out.append((typ, elem_id, params, stage))
                continue
            coupling = self._get_call_coupling(elem_id)
            if elem_id not in seen:
                out.append(("call_source", elem_id, params,
                            ConferenceSource(coupling)))
                seen.add(elem_id)
            else:
                out.append(("call_sink", elem_id, params,
                            ConferenceSink(coupling)))
        return out

    # -- SIP wrapping ------------------------------------------------------

    def _wrap_sip(self, resolved):
        """Split into sub-chains and wrap sip sessions with SIPSource/SIPSink.

        Rule: neighbour-based piping. For each ``sip:LEG`` element,
        decide which wrapper(s) the position needs based on whether
        a left and/or right neighbour exists:

        * Left neighbour only  (``A -> sip:L``)       → SIPSink (terminal)
        * Right neighbour only (``sip:L -> B``)       → SIPSource (originator)
        * Both (``A -> sip:L -> B``, single leg)      → split into two
          sub-chains: ``A -> SIPSink(L)`` and ``SIPSource(L) -> B``.
        * Same leg twice (``sip:L -> X -> sip:L``)    → SIPSource at the
          first position, SIPSink at the second — one chain (bridge).

        Returns ``list[list[(typ, id, params, stage)]]`` — one list of
        stage tuples per sub-chain.
        """
        from speech_pipeline.SIPSource import SIPSource
        from speech_pipeline.SIPSink import SIPSink
        from speech_pipeline.CodecSocketSource import CodecSocketSource
        from speech_pipeline.CodecSocketSink import CodecSocketSink

        ENDPOINT_TYPES = ("sip", "originate", "codec")

        ep_counts: dict[tuple[str, str], int] = {}
        for typ, elem_id, _, _ in resolved:
            if typ in ENDPOINT_TYPES:
                kind = "codec" if typ == "codec" else "sip"
                ep_counts[(kind, elem_id)] = ep_counts.get((kind, elem_id), 0) + 1

        def _make_source(typ, stage, leg):
            if typ == "codec":
                return ("codec_source", CodecSocketSource(stage))
            return ("sip_source", SIPSource(stage, leg=leg))

        def _make_sink(typ, stage, leg):
            if typ == "codec":
                return ("codec_sink", CodecSocketSink(stage))
            return ("sip_sink", SIPSink(stage, leg=leg))

        chains: list[list[tuple]] = [[]]
        seen_source: set[tuple[str, str]] = set()
        n = len(resolved)
        for i, (typ, elem_id, params, stage) in enumerate(resolved):
            if typ not in ENDPOINT_TYPES:
                chains[-1].append((typ, elem_id, params, stage))
                continue

            kind = "codec" if typ == "codec" else "sip"
            leg = self._session_leg(elem_id) if kind == "sip" else None
            has_left = i > 0
            has_right = i < n - 1
            count = ep_counts.get((kind, elem_id), 0)

            if count >= 2:
                key = (kind, elem_id)
                if key not in seen_source:
                    role, wrapper = _make_source(typ, stage, leg)
                    chains[-1].append((role, elem_id, params, wrapper))
                    seen_source.add(key)
                else:
                    role, wrapper = _make_sink(typ, stage, leg)
                    chains[-1].append((role, elem_id, params, wrapper))
                continue

            if has_left and has_right:
                # Middle position, single occurrence → split.
                role_sink, sink = _make_sink(typ, stage, leg)
                role_src, src = _make_source(typ, stage, leg)
                chains[-1].append((role_sink, elem_id, params, sink))
                chains.append([(role_src, elem_id, params, src)])
            elif has_right:
                role, wrapper = _make_source(typ, stage, leg)
                chains[-1].append((role, elem_id, params, wrapper))
            elif has_left:
                role, wrapper = _make_sink(typ, stage, leg)
                chains[-1].append((role, elem_id, params, wrapper))
            else:
                raise ValueError(
                    f"{typ}:{elem_id} standalone has no neighbours to pipe into")

        return [c for c in chains if c]

    def _register_bridge_handles(self, resolved):
        """Register bidirectional SIP bridges so they can be detached live."""
        from speech_pipeline.ConferenceLeg import ConferenceLeg

        bridge_leg_id = None
        conf_leg = None
        tee_ids = []
        for typ, elem_id, _, stage in resolved:
            if typ == "sip_source" and bridge_leg_id is None:
                bridge_leg_id = elem_id
            if typ == "tee":
                tee_ids.append(elem_id)
            if isinstance(stage, ConferenceLeg):
                conf_leg = stage
                break

        if not bridge_leg_id or conf_leg is None:
            return

        leg = self._session_leg(bridge_leg_id)
        if not leg:
            return

        def _on_attached(conf):
            leg.conf_leg = conf
            leg._src_id = getattr(conf, "_src_id", None)

        def _on_detached(conf):
            if getattr(leg, "_conf_leg", None) is conf:
                leg.conf_leg = None
                leg._src_id = None

        conf_leg.on_attached = _on_attached
        conf_leg.on_detached = _on_detached

        def _cleanup():
            with self._lock:
                for tee_id in tee_ids:
                    tee = self._tees.pop(tee_id, None)
                    if tee:
                        try:
                            tee.cancel()
                        except Exception:
                            pass
                self._sidechain_specs = {
                    spec for spec in self._sidechain_specs
                    if spec[0] not in tee_ids
                }

        handle = type("_BridgeHandle", (), {
            "_stage": conf_leg,
            "_current_src": None,
            "_stop": None,
            "_cleanup": _cleanup,
        })()
        with self._lock:
            self._stages[f"bridge:{bridge_leg_id}"] = handle

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
                leg = self._session_leg(elem_id)
                session = self._session_call(elem_id)
                if leg and session:
                    self._start_sip_monitors(leg, session)

        # Any non-ConferenceLeg stage directly before a terminal ConferenceLeg → mixer.add_source
        from speech_pipeline.ConferenceLeg import ConferenceLeg as _CL
        for i in range(len(resolved) - 1):
            l_typ, l_id, l_params, l_stage = resolved[i]
            _, _, _, r_stage = resolved[i + 1]
            if isinstance(r_stage, _CL) and i + 1 == len(resolved) - 1:
                src_id = self.call.mixer.add_source(l_stage)
                # Wire play loop if applicable
                handle = getattr(l_stage, '_play_handle', None)
                if handle:
                    handle._current_src = src_id
                    if handle._loop:
                        self._start_play_loop(handle, src_id)
                # Fire completed callback when source finishes
                completed_cb = l_params.get("completed", "") if l_params else ""
                if completed_cb:
                    self._start_source_completed_monitor(src_id, completed_cb)

    # -- Source completed monitor ------------------------------------------

    def _start_source_completed_monitor(self, src_id, callback_path):
        """Wait for a mixer source to finish, then fire a webhook callback."""
        from . import subscriber as sub_mod

        sub = sub_mod.get(self.call.subscriber_id) if self.call.subscriber_id else None
        if not sub:
            return

        def _monitor():
            self.call.mixer.wait_source(src_id)
            from . import _shared as _sh
            url = _sh.subscriber_url(sub, callback_path)
            try:
                import requests
                requests.post(url, json={"call_id": self.call.call_id, "source": src_id},
                              headers={"Authorization": f"Bearer {sub['bearer_token']}"},
                              timeout=10)
                _LOGGER.info("Source %s completed → %s", src_id, callback_path)
            except Exception as e:
                _LOGGER.warning("Source completed callback failed: %s", e)

        threading.Thread(target=_monitor, daemon=True,
                         name=f"src-done-{src_id}").start()

    # -- SIP monitors ------------------------------------------------------

    def _start_sip_monitors(self, leg, session):
        from . import leg as leg_mod
        leg_id = leg.leg_id

        def _dtmf():
            while leg.status == "in-progress":
                try:
                    d = leg.voip_call.get_dtmf(length=1)
                    if d:
                        leg_mod.fire_callback(leg, "dtmf", digit=d,
                                               call_id=self.call.call_id)
                except Exception:
                    time.sleep(0.1)
        threading.Thread(target=_dtmf, daemon=True, name=f"dtmf-{leg_id}").start()

        if getattr(leg, "completion_monitor_started", False):
            _LOGGER.info("SIP leg %s wired", leg_id)
            return

        def _monitor():
            ended = False
            while leg.status == "in-progress":
                ended = leg_mod.remote_end_detected(leg, session)
                if ended:
                    break
                time.sleep(0.5)
            if not ended:
                _LOGGER.info("Leg %s monitor stopped without remote hangup", leg_id)
                return
            leg.status = "completed"
            dur = time.time() - leg.answered_at if leg.answered_at else 0
            if hasattr(leg, 'conf_leg') and leg.conf_leg:
                try: leg.conf_leg.cancel()
                except: pass
            if self.call is not None:
                try:
                    self.call.unregister_participant(leg.leg_id)
                except Exception:
                    pass
            leg_mod.fire_callback(leg, "completed", duration=dur)
            _LOGGER.info("Leg %s ended (%.1fs)", leg_id, dur)
            leg_mod.delete_leg(leg_id)
        leg.completion_monitor_started = True
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
                    reader = AudioReader(
                        url,
                        chunk_seconds=_PLAY_CHUNK_SECONDS,
                        realtime=True,
                        prefill_seconds=_PLAY_PREFILL_SECONDS,
                    )
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
