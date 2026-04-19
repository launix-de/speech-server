from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from speech_pipeline.WhisperSTT import WhisperTranscriber
from speech_pipeline.base import AudioFormat, Stage


class _PCMSource(Stage):
    def __init__(self, chunks: list[bytes], sample_rate: int = 16000) -> None:
        super().__init__()
        self._chunks = chunks
        self.output_format = AudioFormat(sample_rate, "s16le")

    def stream_pcm24k(self):
        for chunk in self._chunks:
            if self.cancelled:
                break
            yield chunk


class _RecordingModel:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def transcribe(self, samples, **kwargs):
        self.calls.append(len(samples))
        segment = SimpleNamespace(
            text=f"chunk-{len(self.calls)}",
            start=0.0,
            end=len(samples) / 16000.0,
            no_speech_prob=0.0,
            avg_logprob=-0.2,
        )
        return [segment], None


def _pcm(duration_s: float, sample_value: int, sample_rate: int = 16000) -> bytes:
    samples = int(duration_s * sample_rate)
    return int(sample_value).to_bytes(2, "little", signed=True) * samples


def _decode(lines: list[bytes]) -> list[dict]:
    return [json.loads(line.decode()) for line in lines]


def test_chunk_seconds_controls_pause_flush(monkeypatch):
    model = _RecordingModel()
    monkeypatch.setattr("speech_pipeline.WhisperSTT._get_model", lambda _: model)

    speech = _pcm(1.2, 2000)
    pause = _pcm(0.4, 0)
    source = _PCMSource([speech, pause, speech, pause])
    stt = WhisperTranscriber(chunk_seconds=3.0, language="de")
    source.pipe(stt)

    out = list(stt.stream_pcm24k())

    assert len(model.calls) == 1, "3s chunking must not flush after the first short pause"
    decoded = _decode(out)
    assert [row["text"] for row in decoded] == ["chunk-1"]


def test_short_chunk_seconds_allows_earlier_pause_flush(monkeypatch):
    model = _RecordingModel()
    monkeypatch.setattr("speech_pipeline.WhisperSTT._get_model", lambda _: model)

    speech = _pcm(1.2, 2000)
    pause = _pcm(0.4, 0)
    source = _PCMSource([speech, pause, speech, pause])
    stt = WhisperTranscriber(chunk_seconds=1.0, language="de")
    source.pipe(stt)

    out = list(stt.stream_pcm24k())

    assert len(model.calls) == 2
    decoded = _decode(out)
    assert [row["text"] for row in decoded] == ["chunk-1", "chunk-2"]

