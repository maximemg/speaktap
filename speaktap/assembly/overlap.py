"""Order chunks and remove repeated words introduced by audio overlap."""

from __future__ import annotations

import re

from ..domain import TranscriptChunk

_EDGE_PUNCTUATION = re.compile(r"(^[^\w']+|[^\w']+$)")


def assemble_transcript(chunks: tuple[TranscriptChunk, ...]) -> str:
    assembled: list[str] = []
    for chunk in sorted(chunks, key=lambda item: item.sequence):
        incoming = chunk.raw_text.strip().split()
        if not incoming:
            continue
        overlap = _word_overlap(assembled, incoming)
        assembled.extend(incoming[overlap:])
    return " ".join(assembled).strip()


def _word_overlap(existing: list[str], incoming: list[str], limit: int = 16) -> int:
    # Forced audio overlap is only a fraction of a second, so 16 words is a
    # deliberately generous ceiling. The cap bounds comparison work and avoids
    # deleting a long, legitimately repeated passage after a coincidental match.
    upper = min(len(existing), len(incoming), limit)
    normalized_existing = [_normalize(word) for word in existing]
    normalized_incoming = [_normalize(word) for word in incoming]
    for size in range(upper, 0, -1):
        if normalized_existing[-size:] == normalized_incoming[:size]:
            return size
    return 0


def _normalize(word: str) -> str:
    return _EDGE_PUNCTUATION.sub("", word.casefold())
