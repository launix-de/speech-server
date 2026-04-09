"""Unified DSL parser for audio pipelines.

Element syntax: ``type:id{json_params}``
Separator: ``->`` or ``|``

Examples::

    sip:leg1{"completed":"/cb"} -> call:call-xxx -> sip:leg1
    play:hold{"url":"music.mp3","loop":true} -> call:call-xxx
    tts:de{"text":"Hallo"} -> call:call-xxx
    tee:tap -> stt:de -> webhook:https://example.com/stt
    codec:wc-abc | tee:stt | conference:call-xxx | codec:wc-abc
    text_input | tts:de_DE-thorsten-medium | conference:call-xxx
"""
from __future__ import annotations

import json
import re
from typing import List, Tuple

_ARROW = re.compile(r'\s*(->|\|)\s*')
_ELEMENT = re.compile(r'\s*([a-z_]+)(?::([^{\s|>]+))?')


def _consume_json(s: str, pos: int) -> tuple[dict, int]:
    """Parse an inline JSON object starting at *pos*."""
    if pos >= len(s) or s[pos] != '{':
        return {}, pos
    depth, in_str, esc, start = 0, False, False, pos
    while pos < len(s):
        ch = s[pos]
        if esc:
            esc = False
        elif ch == '\\' and in_str:
            esc = True
        elif ch == '"' and not esc:
            in_str = not in_str
        elif not in_str:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    pos += 1
                    try:
                        return json.loads(s[start:pos]), pos
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON params: {e.msg}") from e
        pos += 1
    raise ValueError("Unterminated JSON params")


def parse_dsl(dsl: str) -> List[Tuple[str, str, dict]]:
    """Parse a DSL string into a list of ``(type, id, params)`` tuples.

    Separators ``->`` and ``|`` are interchangeable.  Each element has
    the form ``type``, ``type:id``, or ``type:id{json}``.
    """
    elements: list[tuple[str, str, dict]] = []
    pos = 0
    s = dsl.strip()
    expect_element = True
    while pos < len(s):
        if expect_element:
            m = _ELEMENT.match(s, pos)
            if not m:
                snippet = s[pos:pos + 32]
                raise ValueError(f"Invalid DSL syntax near: {snippet!r}")
            typ, elem_id = m.group(1), m.group(2) or ""
            pos = m.end()
            params, pos = _consume_json(s, pos)
            elements.append((typ, elem_id, params))
            expect_element = False
        else:
            m = _ARROW.match(s, pos)
            if not m:
                snippet = s[pos:pos + 32]
                raise ValueError(f"Expected pipe separator near: {snippet!r}")
            pos = m.end()
            expect_element = True
    if expect_element and elements:
        raise ValueError("DSL must not end with a pipe separator")
    return elements
