import pytest

from benchmark.cleanup.run_candidate import (
    apply_token_selection,
    deterministic_disfluency_cleanup,
    parse_delete_indices,
    parse_keep_indices,
)


def test_parse_keep_indices_accepts_only_strict_json_schema() -> None:
    assert parse_keep_indices('{"keep":[0,2,3]}', 4) == (0, 2, 3)

    for raw in (
        '{"keep":[2,1]}',
        '{"keep":[1,1]}',
        '{"keep":[4]}',
        '{"keep":[]}',
        '{"delete":[1]}',
        "keep 0 1",
    ):
        with pytest.raises(ValueError):
            parse_keep_indices(raw, 4)


def test_parse_delete_indices_allows_an_empty_noop() -> None:
    assert parse_delete_indices('{"delete":[]}', 4) == ()
    assert parse_delete_indices('{"delete":[0,2]}', 4) == (0, 2)

    with pytest.raises(ValueError, match="invalid-schema"):
        parse_delete_indices('{"keep":[0,2]}', 4)


def test_selection_can_only_remove_fillers_and_adjacent_duplicates() -> None:
    decision = apply_token_selection(
        "um I I need HTTP 429",
        (1, 3, 4, 5),
    )

    assert decision.accepted
    assert decision.output == "I need HTTP 429"


def test_selection_rejects_arbitrary_or_protected_deletions() -> None:
    arbitrary = apply_token_selection(
        "please send report tomorrow",
        (0, 1, 3),
    )
    protected = apply_token_selection(
        "hello Sarah tomorrow",
        (0, 2),
    )

    assert not arbitrary.accepted
    assert arbitrary.reason == "unsafe-deletion-2"
    assert arbitrary.output == "please send report tomorrow"
    assert not protected.accepted
    assert protected.reason == "protected-token-1"


def test_selection_accepts_only_prefix_suffix_correction_splice() -> None:
    text = "send the report on Thursday no sorry send it on Friday"

    accepted = apply_token_selection(text, (0, 1, 2, 3, 10))
    missing_correction = apply_token_selection(text, (0, 1, 2, 3, 4))
    invalid_middle = apply_token_selection(text, (0, 2, 10))
    incoherent_boundary = apply_token_selection(text, (0, 1, 8, 9, 10))

    assert accepted.accepted
    assert accepted.output == "send the report on Friday"
    assert not missing_correction.accepted
    assert missing_correction.reason == "final-token-deleted"
    assert not invalid_middle.accepted
    assert invalid_middle.reason == "invalid-correction-prefix"
    assert not incoherent_boundary.accepted
    assert incoherent_boundary.reason == "unsafe-correction-boundary"


def test_selection_may_preserve_an_unresolved_correction_but_not_edit_around_it() -> None:
    text = "send the report Thursday no sorry send it Friday"

    preserved = apply_token_selection(text, tuple(range(9)))
    unsafe = apply_token_selection(text, (0, 2, 3, 4, 5, 6, 7, 8))

    assert preserved.accepted
    assert preserved.output == text
    assert not unsafe.accepted
    assert unsafe.reason == "unsafe-deletion-1"


def test_deterministic_cleanup_removes_only_fillers_and_repetition() -> None:
    assert deterministic_disfluency_cleanup("um I I need this today") == "I need this today"
    assert (
        deterministic_disfluency_cleanup("euh je vais je vais partir demain")
        == "je vais partir demain"
    )
    assert (
        deterministic_disfluency_cleanup(
            "send Thursday no sorry send Friday",
        )
        == "send Thursday no sorry send Friday"
    )
