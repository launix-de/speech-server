from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from speech_pipeline.WhisperSTT import WhisperTranscriber
from speech_pipeline import WhisperSTT as whisper_mod
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


def test_hard_limit_prefers_longest_pause_in_recent_window():
    sample_rate = 16000
    pcm = b"".join([
        _pcm(2.45, 2000, sample_rate),
        _pcm(0.10, 0, sample_rate),
        _pcm(0.15, 2000, sample_rate),
        _pcm(0.40, 0, sample_rate),
        _pcm(0.50, 2000, sample_rate),
    ])

    target_bytes = int(3.10 * sample_rate * 2)
    search_window_bytes = int(0.75 * sample_rate * 2)
    cut = whisper_mod._find_recent_pause_cut_bytes(
        pcm,
        sample_rate=sample_rate,
        target_bytes=target_bytes,
        search_window_bytes=search_window_bytes,
        rms_floor=220,
    )

    assert cut is not None
    lower = int(2.80 * sample_rate * 2)
    upper = int(3.20 * sample_rate * 2)
    assert lower <= cut <= upper, (
        "smart hard-cut should land inside the longest recent pause, "
        f"got cut={cut} outside [{lower}, {upper}]"
    )


def test_hard_limit_without_pause_keeps_small_overlap():
    sample_rate = 16000
    bps = sample_rate * 2
    pcm = _pcm(16.0, 2000, sample_rate)

    split = whisper_mod._choose_hard_split_bytes(
        pcm,
        sample_rate=sample_rate,
        max_chunk_bytes=int(15.0 * bps),
        search_window_bytes=int(0.75 * bps),
        rms_floor=220,
        overlap_bytes=int(0.30 * bps),
    )

    assert split == int((15.0 - 0.30) * bps), (
        "hard limit without a usable pause must keep a small overlap "
        "instead of cutting exactly at the limit"
    )
