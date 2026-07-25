"""High-confidence, model-free transcript cleanup."""

from __future__ import annotations

import re
import time

from ..domain import CleanupResult, CleanupStatus, TranscriptChunk

_WORDS = re.compile(r"\w+(?:'\w+)*", flags=re.UNICODE)
_SPANS = re.compile(r"\S+")
_FILLERS = frozenset({"ah", "er", "euh", "heu", "hmm", "uh", "um"})
_MAX_REPEATED_PHRASE_WORDS = 4

# Words whose immediate repetition is not a disfluency, so deleting the second
# token would change meaning rather than remove noise. Cleanup only claims
# unambiguous edits, so these are exempt from single-word repetition removal.
# Repeated multi-word phrases are still removed, and a longer stutter such as
# "nous nous nous nous" collapses onto the grammatical pair rather than past it.
_GRAMMATICAL_DOUBLES = frozenset(
    {
        # French subject pronoun followed by the identical reflexive pronoun,
        # as in "nous nous levons" and "vous vous trompez".
        "nous",
        "vous",
        # French emphatic affirmation and negation: "si si", "non non".
        "si",
        "non",
        "oui",
        # English doubling that grammar requires: past perfect "had had",
        # complementizer plus demonstrative "that that", and the pseudo-cleft
        # "what it is is".
        "had",
        "that",
        "is",
        # Conventional intensifiers and interjections where the repetition
        # carries the emphasis: "very very cold", "a long long time".
        "very",
        "well",
        "no",
        "yes",
        "many",
        "long",
        "far",
    }
)


def _token_key(token: str) -> str:
    match = _WORDS.search(token)
    return match.group().casefold() if match else token.casefold()


def _is_filler(tokens: tuple[str, ...], keys: tuple[str, ...], index: int) -> bool:
    if keys[index] not in _FILLERS:
        return False
    word = _WORDS.search(tokens[index])
    # Preserve uppercase tokens such as "ER": they are likely acronyms, not the
    # lowercase hesitation "er".
    if word and len(word.group()) > 1 and word.group().isupper():
        return False
    # Preserve unit expressions such as "20 Ah" (ampere-hours), where "Ah" is
    # semantic content rather than a hesitation.
    return not (
        keys[index] == "ah"
        and index > 0
        and any(character.isdigit() for character in tokens[index - 1])
    )


def safe_cleanup_text(text: str) -> str:
    """Remove explicit fillers and immediately repeated words or short phrases."""
    tokens = tuple(match.group() for match in _SPANS.finditer(text))
    keys = tuple(_token_key(token) for token in tokens)
    keep: list[int] = []
    index = 0
    while index < len(tokens):
        if _is_filler(tokens, keys, index):
            index += 1
            continue
        repeated_width = 0
        max_width = min(
            _MAX_REPEATED_PHRASE_WORDS,
            (len(tokens) - index) // 2,
        )
        # Widths are tried longest first, so a repeated phrase is always matched
        # before the single-word rule and stays removable.
        for width in range(max_width, 0, -1):
            if keys[index : index + width] != keys[index + width : index + 2 * width]:
                continue
            if width == 1 and keys[index] in _GRAMMATICAL_DOUBLES:
                break
            repeated_width = width
            break
        if repeated_width:
            keep.extend(range(index, index + repeated_width))
            index += 2 * repeated_width
            continue
        keep.append(index)
        index += 1
    return " ".join(tokens[token_index] for token_index in keep)


class SafeCleaner:
    """Apply only deterministic edits with a narrow, inspectable rule set."""

    cleaner_id = "safe"

    def clean(
        self,
        assembled_text: str,
        chunks: tuple[TranscriptChunk, ...],
        *,
        timeout_seconds: float,
    ) -> CleanupResult:
        del chunks, timeout_seconds
        started = time.monotonic()
        text = safe_cleanup_text(assembled_text)
        duration_ms = round((time.monotonic() - started) * 1000)
        return CleanupResult(
            text=text,
            status=CleanupStatus.SUCCESS,
            cleaner_id=self.cleaner_id,
            duration_ms=duration_ms,
        )

    def close(self) -> None:
        return None
