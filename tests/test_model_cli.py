from __future__ import annotations

import json
from pathlib import Path

from speaktap.config import SpeakTapConfig
from speaktap.model_cli import write_installed_profile


def test_write_installed_profile_is_loadable(tmp_path: Path) -> None:
    path = tmp_path / "config.json"

    written = write_installed_profile("canary-1b-v2-int8", path=path)
    config = SpeakTapConfig.load({}, path=path)

    assert written == path
    assert config.asr_profile == "canary-1b-v2-int8"
    assert config.asr_model == "nemo-canary-1b-v2"
    assert json.loads(path.read_text())["schema_version"] == 1
    assert path.stat().st_mode & 0o777 == 0o600
