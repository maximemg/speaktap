"""Private per-user paths for the local SpeakTap service."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def runtime_directory() -> Path:
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    user_runtime = Path(f"/run/user/{os.getuid()}")
    if xdg_runtime:
        base = Path(xdg_runtime)
        name = "speaktap"
    elif (
        user_runtime.is_dir()
        and user_runtime.stat().st_uid == os.getuid()
        and os.access(user_runtime, os.W_OK | os.X_OK)
    ):
        base = user_runtime
        name = "speaktap"
    else:
        # Safe fallback: the derived directory is ownership-checked, rejects
        # symlinks, and is forced to mode 0700 immediately below.
        base = Path("/tmp")  # nosec B108
        name = f"speaktap-{os.getuid()}"
    directory = base / name
    if directory.exists() or directory.is_symlink():
        metadata = directory.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"runtime directory must not be a symlink: {directory}")
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise RuntimeError(f"runtime directory is not private to this user: {directory}")
    else:
        directory.mkdir(mode=0o700, parents=True)
    directory.chmod(0o700)
    return directory


def socket_path() -> Path:
    return runtime_directory() / "service.sock"


def lock_path() -> Path:
    return runtime_directory() / "service.lock"


def log_path() -> Path:
    return runtime_directory() / "service.log"


def state_directory() -> Path:
    configured = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(configured) if configured else Path.home() / ".local" / "state"
    directory = base / "speaktap"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


def session_log_path() -> Path:
    return state_directory() / "sessions.jsonl"


def config_directory() -> Path:
    configured = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(configured) if configured else Path.home() / ".config"
    directory = base / "speaktap"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


def installed_config_path() -> Path:
    return config_directory() / "config.json"
