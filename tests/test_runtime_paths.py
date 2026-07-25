from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import pytest

from speaktap.runtime import config_directory, state_directory


@pytest.mark.parametrize(
    ("variable", "resolve"),
    [
        ("XDG_STATE_HOME", state_directory),
        ("XDG_CONFIG_HOME", config_directory),
    ],
)
def test_private_directory_is_created_with_owner_only_access(
    monkeypatch: Any,
    tmp_path: Path,
    variable: str,
    resolve: Any,
) -> None:
    monkeypatch.setenv(variable, str(tmp_path))

    directory = resolve()

    assert stat.S_IMODE(directory.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    ("variable", "resolve"),
    [
        ("XDG_STATE_HOME", state_directory),
        ("XDG_CONFIG_HOME", config_directory),
    ],
)
def test_existing_directory_has_its_permissions_tightened(
    monkeypatch: Any,
    tmp_path: Path,
    variable: str,
    resolve: Any,
) -> None:
    """mkdir(mode=...) is ignored when the directory already exists.

    These directories hold dictation diagnostics and the installed profile, so
    a pre-existing world-readable directory must not stay that way.
    """

    monkeypatch.setenv(variable, str(tmp_path))
    existing = tmp_path / "speaktap"
    existing.mkdir(parents=True)
    existing.chmod(0o755)

    directory = resolve()

    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
