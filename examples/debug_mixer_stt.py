#!/usr/bin/env python3
"""Debug: feed the recorded WAV through ConferenceMixer → STT.

Reproduces the exact chain that fails in production, but locally
so we can inspect every step.
"""
import threading
import queue

from speech_pipeline.ConferenceMixer import ConferenceMixer
from speech_pipeline.AudioReader import AudioReader
from speech_pipeline.QueueSource import QueueSource
from speech_pipeline.WhisperSTT import WhisperTranscriber

WAV = "/tmp/stt_debug.wav"

print("=== Test: WAV → add_source → mixer → add_output → QueueSource → STT ===\n")

# 1. Create mixer (same config as telephony)
mixer = ConferenceMixer("test", sample_rate=48000, frame_samples=1024)
t_mixer = threading.Thread(target=mixer.run, daemon=True)
t_mixer.start()

# 2. Add output for STT (like conference:CALL first element does)
out_q = mixer.add_output()
src = QueueSource(out_q, 48000, "s16le")
stt = WhisperTranscriber("small", language="de")
src.pipe(stt)  # auto-inserts SampleRateConverter 48k→16k

# 3. Feed WAV as source (like webclient does)
reader = AudioReader(WAV, chunk_seconds=0.5)
mixer.add_source(reader)

# 4. Consume STT output
print("Running STT...")
results = []
for chunk in stt.stream_pcm24k():
    text = chunk.decode().strip()
    if text:
        print(f"  STT: {text}")
        results.append(text)

mixer.cancel()

print(f"\n=== Results: {len(results)} segments ===")
if not results:
    print("FAILURE: no STT output")
else:
    print("SUCCESS")
