"""Unit tests for the telephony DSL parser (parse_dsl).

Tests the parser in isolation — no Flask app or fixtures needed.
"""
import pytest
from speech_pipeline.dsl_parser import parse_dsl


class TestDSLArrowSyntax:
    def test_simple_arrow(self):
        result = parse_dsl("play:intro -> call:c1")
        assert len(result) == 2
        assert result[0] == ("play", "intro", {})
        assert result[1] == ("call", "c1", {})

    def test_three_elements(self):
        result = parse_dsl("sip:l1 -> call:c1 -> sip:l1")
        assert len(result) == 3
        assert result[0] == ("sip", "l1", {})
        assert result[1] == ("call", "c1", {})
        assert result[2] == ("sip", "l1", {})

    def test_four_elements(self):
        result = parse_dsl("sip:l1 -> tee:t1 -> call:c1 -> sip:l1")
        assert len(result) == 4


class TestDSLPipeSyntax:
    def test_pipe_separator(self):
        result = parse_dsl("stt:de | webhook:https://example.com")
        assert len(result) == 2
        assert result[0][0] == "stt"
        assert result[1] == ("webhook", "https://example.com", {})


class TestDSLMixedSyntax:
    def test_mixed_separators(self):
        result = parse_dsl("sip:l1 -> call:c1 | sip:l1")
        assert len(result) == 3


class TestDSLElementIDs:
    def test_element_without_id(self):
        result = parse_dsl("stt -> webhook:https://example.com")
        assert result[0] == ("stt", "", {})

    def test_single_element(self):
        result = parse_dsl("play:intro")
        assert len(result) == 1
        assert result[0] == ("play", "intro", {})

    def test_url_as_id(self):
        result = parse_dsl("webhook:https://example.com/hook")
        assert result[0][1] == "https://example.com/hook"

    def test_hyphenated_id(self):
        result = parse_dsl("sip:leg-abc-123")
        assert result[0][1] == "leg-abc-123"


class TestDSLJSONParams:
    def test_simple_json(self):
        result = parse_dsl('play:intro{"url":"https://example.com/a.wav","loop":true}')
        assert result[0][0] == "play"
        assert result[0][1] == "intro"
        assert result[0][2] == {"url": "https://example.com/a.wav", "loop": True}

    def test_nested_json(self):
        result = parse_dsl('tts:de{"text":"Hallo","opts":{"speed":1.2}}')
        assert result[0][2]["opts"]["speed"] == 1.2

    def test_json_with_numbers(self):
        result = parse_dsl('play:x{"volume":80,"loop":3}')
        assert result[0][2]["volume"] == 80
        assert result[0][2]["loop"] == 3

    def test_json_with_array(self):
        result = parse_dsl('play:x{"tags":["a","b"]}')
        assert result[0][2]["tags"] == ["a", "b"]

    def test_json_empty_object(self):
        result = parse_dsl('play:x{}')
        assert result[0][2] == {}

    def test_json_with_escaped_quotes(self):
        result = parse_dsl(r'tts:de{"text":"Er sagte \"Hallo\""}')
        assert '"Hallo"' in result[0][2]["text"]

    def test_json_params_then_arrow(self):
        result = parse_dsl('play:x{"url":"a.wav"} -> call:c1')
        assert len(result) == 2
        assert result[0][2]["url"] == "a.wav"
        assert result[1] == ("call", "c1", {})

    def test_no_json_means_empty_dict(self):
        result = parse_dsl("play:x -> call:c1")
        assert result[0][2] == {}
        assert result[1][2] == {}


class TestDSLWhitespace:
    def test_leading_trailing_spaces(self):
        result = parse_dsl("  play:x  ->  call:c1  ")
        assert len(result) == 2

    def test_no_spaces_around_arrow_needs_whitespace(self):
        """Arrow without surrounding spaces is not recognized as separator."""
        import pytest
        with pytest.raises(ValueError):
            parse_dsl("play:x->call:c1")

    def test_no_spaces_around_pipe(self):
        result = parse_dsl("stt:de|webhook:https://example.com")
        assert len(result) == 2


class TestDSLErrors:
    def test_empty_string(self):
        result = parse_dsl("")
        assert result == []

    def test_whitespace_only(self):
        result = parse_dsl("   ")
        assert result == []

    def test_trailing_arrow(self):
        with pytest.raises(ValueError, match="must not end"):
            parse_dsl("play:x ->")

    def test_trailing_pipe(self):
        with pytest.raises(ValueError, match="must not end"):
            parse_dsl("play:x |")

    def test_invalid_start(self):
        with pytest.raises(ValueError, match="Invalid DSL"):
            parse_dsl("123invalid")

    def test_double_arrow(self):
        with pytest.raises(ValueError, match="Invalid DSL"):
            parse_dsl("play:x -> -> call:c1")

    def test_unterminated_json(self):
        with pytest.raises(ValueError, match="Unterminated"):
            parse_dsl('play:x{"url":"broken')

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_dsl('play:x{not json}')

    def test_missing_separator(self):
        with pytest.raises(ValueError, match="Expected pipe separator"):
            parse_dsl("play:x call:c1")
