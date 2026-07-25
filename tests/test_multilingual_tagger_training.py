from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any, cast

_SCRIPT = Path(__file__).parents[1] / "benchmark" / "cleanup" / "train_multilingual_tagger.py"
_MODULE = runpy.run_path(str(_SCRIPT))


def test_fluent_column_does_not_match_disfluent_header() -> None:
    find_column = cast(Any, _MODULE["_find_column"])
    headers = (
        "Sentence Number",
        "French Disfluent Sentence",
        "French Fluent Sentence",
    )

    assert find_column(headers, "disfluent") == 1
    assert find_column(headers, "fluent") == 2


def test_deletion_labels_keep_the_rightmost_fluent_subsequence() -> None:
    derive_example = cast(Any, _MODULE["derive_example"])

    example = derive_example(
        "Send uhm send ETA via whatsapp",
        "Send ETA via whatsapp",
        "en",
    )

    assert example is not None
    assert example.labels == (1, 1, 0, 0, 0, 0)


def test_deletion_labels_handle_french_accents_and_punctuation() -> None:
    derive_example = cast(Any, _MODULE["derive_example"])

    example = derive_example(
        "Annule euh annule la course.",
        "Annule la course.",
        "fr",
    )

    assert example is not None
    assert example.labels == (1, 1, 0, 0, 0)
