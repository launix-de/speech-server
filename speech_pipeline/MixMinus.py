"""Mix-minus stage: subtracts own audio from the conference mix.

Prevents participants from hearing themselves. Takes two inputs:

1. The full conference mix (from AudioMixer output)
2. The participant's own audio (the same signal fed into the mixer)

Outputs ``full_mix - own_audio`` — i.e. everyone else.

Usage::

    # Per participant:
    own_tee = AudioTee(8000)         # splits participant's audio
    own_tee_to_mixer = ...           # one copy → mixer input
    own_tee_to_minus = ...           # other copy → MixMinus.set_own()

    mix_minus = MixMinus(8000)
    mix_minus.set_own(own_queue)     # participant's own audio
    mixer_output | mix_minus         # full mix as upstream
    mix_minus | SIPSink(...)         # cleaned output → back to participant
"""
from __future__ import annotations

import audioop
import logging
import queue
from typing import Iterator, Optional

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("mix-minus")


class MixMinus(Stage):
    """Processor: subtracts own audio from upstream (full mix).

    Upstream provides the full conference mix.  ``own_queue`` provides
    the participant's own audio (same frames, same timing).
    Output is ``upstream - own``.
    """

    def __init__(self, sample_rate: int = 8000) -> None:
        super().__init__()
        fmt = AudioFormat(sample_rate, "s16le")
        self.input_format = fmt
        self.output_format = fmt
        self._own_queue: Optional[queue.Queue] = None

    def set_own(self, q: queue.Queue) -> None:
        """Set the queue that carries this participant's own audio."""
        self._own_queue = q

    def stream_pcm24k(self) -> Iterator[bytes]:
        if not self.upstream:
            return

        own_buf = bytearray()

        for mix_frame in self.upstream.stream_pcm24k():
            if self.cancelled:
                break

            if self._own_queue:
                # Drain queue into buffer
                try:
                    while True:
                        own_buf.extend(self._own_queue.get_nowait())
                except queue.Empty:
                    pass

                need = len(mix_frame)
                if len(own_buf) >= need:
                    own_frame = bytes(own_buf[:need])
                    del own_buf[:need]
                    negated = audioop.mul(own_frame, 2, -1)
                    yield audioop.add(mix_frame, negated, 2)
                    continue

            # Not enough own audio — pass full mix through
            yield mix_frame
