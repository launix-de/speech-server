"""Virtual-leg endpoints for the ``call:ID`` DSL element.

Invariante
----------
Wenn ``call`` in einem DSL verwendet wird, wird **ein** virtueller Leg
auf die Konferenz abgezweigt.  Zwei Vorkommen desselben ``call:ID``
(egal ob im selben Pipe-String oder über mehrere Pipes verteilt) meinen
**Input und Output desselben Legs**:

* ``sip:A -> call:C -> sip:A``  – voller SIP-Leg ↔ Konferenz-Bridge.
* ``call:C -> sip:A -> call:C`` – äquivalent, nur aus Konferenz-Sicht
  geschrieben: `call:C` links = Audio aus der Konferenz (mix-minus),
  `call:C` rechts = Audio in die Konferenz hinein.

Beide DSLs beschreiben dieselbe bidirektionale Kopplung.  Der
Conference-Leg wird genau einmal beim Mixer registriert und mit
``mute_source`` auf sich selbst gekoppelt (Echo-Cancel / Mix-Minus).

Semantik pro Position
---------------------
* Nur rechter Nachbar (``call:C -> X``)         → ``ConferenceSource``
* Nur linker Nachbar  (``X -> call:C``)         → ``ConferenceSink``
* Beide Nachbarn (``A -> call:C -> B``)         → ``ConferenceLeg``
  (bidirektional, ein Stage-Objekt)
* Zwei Vorkommen (``call:C -> … -> call:C``)    → Source + Sink mit
  gemeinsamem ``_Coupling``.

DELETE-Kaskade
--------------
``DELETE sip:ID`` löst ``leg.hangup()`` aus.  Das schließt die Session,
``SIPSource`` endet mit ``session.hungup`` → ``close()`` propagiert
downstream → ``ConferenceSink.close()`` setzt EOF auf die mixer-queue →
Mixer entleert den Source-Eintrag → Mix-Minus-Sink bekommt kein Audio
mehr → ``ConferenceSource`` yieldet Stille → idle-Timeout des Mixers
entsorgt den Call.

``DELETE call:ID`` ruft ``call_state.delete_call`` auf, das alle Legs
explizit hangupt.  Die ``ConferenceEndpoint``-Objekte sehen den Mixer
cancelled und beenden ihre Streams ordentlich.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Iterator, Optional

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("conf-endpoint")


class _Coupling:
    """Shared registration for two ``call:ID`` positions of the same call.

    Holds the in-queue (audio INTO the conference) and out-queue (audio
    FROM the conference with mix-minus).  Registers with the mixer
    lazily the first time either endpoint is activated.
    """

    def __init__(self, mixer) -> None:
        self.mixer = mixer
        self.in_queue: queue.Queue = queue.Queue(maxsize=200)
        self.out_queue: Optional[queue.Queue] = None
        self.src_id: Optional[str] = None
        self.sink_id: Optional[str] = None
        self._lock = threading.Lock()
        self._endpoints: set[Stage] = set()
        self._reference = None

    def retain(self, endpoint: Stage) -> None:
        with self._lock:
            if not self._endpoints and isinstance(self.mixer, Stage):
                self._reference = self.mixer.retain()
            self._endpoints.add(endpoint)

    def release(self, endpoint: Stage) -> None:
        with self._lock:
            self._endpoints.discard(endpoint)
            output = None
            reference = None
            if not any(isinstance(e, ConferenceSource) for e in self._endpoints):
                output, self.out_queue = self.out_queue, None
            if not self._endpoints:
                reference, self._reference = self._reference, None
                self.src_id = None
                input_queue = self.in_queue
            else:
                input_queue = None
        if output is not None:
            self.mixer.remove_output(output)
        if input_queue is not None:
            if endpoint.cancelled:
                self.mixer.remove_input(input_queue)
            else:
                try:
                    input_queue.put_nowait(None)
                except queue.Full:
                    pass
        if reference is not None:
            reference.release()

    def activate(self) -> None:
        """Register with the mixer (idempotent)."""
        with self._lock:
            if self.src_id is None:
                self.in_queue, self.src_id = self.mixer.add_input_with_id()
            if self.out_queue is None:
                self.out_queue = self.mixer.add_output(mute_source=self.src_id)


class ConferenceSink(Stage):
    """Pushes upstream audio INTO the conference (acts as mixer source).

    Used for the ``X -> call:C`` direction.  EOF on upstream terminates
    the mixer source cleanly (None sentinel into the in-queue).
    """

    def __init__(self, coupling: _Coupling) -> None:
        super().__init__()
        self.coupling = coupling
        coupling.retain(self)
        self.input_format = AudioFormat(coupling.mixer.sample_rate, "s16le")

    def run(self) -> None:
        if not self.upstream:
            return
        self.coupling.activate()
        q = self.coupling.in_queue
        try:
            for chunk in self.upstream.stream_pcm24k():
                if self.cancelled:
                    break
                try:
                    q.put(chunk, timeout=1)
                except queue.Full:
                    _LOGGER.warning("ConferenceSink: in-queue full, dropping")
        finally:
            try:
                q.put_nowait(None)  # EOF for the mixer
            except queue.Full:
                pass
            self.close()

    def _on_close(self) -> None:
        try:
            self.coupling.in_queue.put_nowait(None)
        except queue.Full:
            pass
        self.coupling.release(self)


class ConferenceSource(Stage):
    """Yields conference mix (with mix-minus) DOWNSTREAM.

    Used for the ``call:C -> Y`` direction.
    """

    def __init__(self, coupling: _Coupling) -> None:
        super().__init__()
        self.coupling = coupling
        coupling.retain(self)
        self.output_format = AudioFormat(coupling.mixer.sample_rate, "s16le")

    def stream_pcm24k(self) -> Iterator[bytes]:
        self.coupling.activate()
        q = self.coupling.out_queue
        assert q is not None
        try:
            while not self.cancelled and not self.closed:
                try:
                    frame = q.get(timeout=0.5)
                except queue.Empty:
                    if self.coupling.mixer.cancelled:
                        break
                    continue
                if frame is None:
                    break
                yield frame
        finally:
            _LOGGER.info("ConferenceSource: stream ended")
            self.close()

    def _on_close(self) -> None:
        self.coupling.release(self)
