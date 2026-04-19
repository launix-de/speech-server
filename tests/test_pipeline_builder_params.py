from __future__ import annotations

import argparse

from speech_pipeline.PipelineBuilder import PipelineBuilder


def test_builder_parse_supports_inline_json_params():
    builder = PipelineBuilder(
        ws=None,
        registry=None,
        args=argparse.Namespace(media_folder=None),
    )
    parsed = builder.parse(
        'cli:text | tts{"voice":"de_DE-thorsten-medium"} | vc{"url":"https://cdn.example.com/voice.wav"}'
    )
    assert parsed[0] == ("cli", ["text"], {})
    assert parsed[1] == ("tts", [], {"voice": "de_DE-thorsten-medium"})
    assert parsed[2] == ("vc", [], {"url": "https://cdn.example.com/voice.wav"})
