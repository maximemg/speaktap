"""Inspect and persist the install-time ASR model choice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SpeakTapConfig
from .profiles import ProfileStatus, get_asr_profile, list_asr_profiles
from .runtime import installed_config_path


def write_installed_profile(profile_id: str, *, path: Path | None = None) -> Path:
    profile = get_asr_profile(profile_id)
    destination = installed_config_path() if path is None else path
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "asr_profile": profile.profile_id,
        "model": profile.model,
        "quantization": profile.quantization,
    }
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(destination)
    return destination


def _print_profiles() -> None:
    for profile in list_asr_profiles():
        marker = "recommended" if profile.status is ProfileStatus.SUPPORTED else "candidate"
        memory = (
            f"~{profile.expected_memory_mib} MiB"
            if profile.expected_memory_mib is not None
            else "memory unmeasured"
        )
        print(
            f"{profile.profile_id}\t{marker}\t{profile.display_name}\t"
            f"{profile.architecture}\t{memory}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="speaktap-models")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list installable and benchmark candidate profiles")
    current = subparsers.add_parser("current", help="show the installed profile")
    current.add_argument("--id-only", action="store_true")
    check = subparsers.add_parser("check", help="validate a profile ID without changing it")
    check.add_argument("profile")
    check.add_argument("--id-only", action="store_true")
    select = subparsers.add_parser("select", help="persist an install-time profile choice")
    select.add_argument("profile")
    args = parser.parse_args()

    if args.command == "list":
        _print_profiles()
        return
    if args.command == "current":
        config = SpeakTapConfig.load()
        if args.id_only:
            print(config.asr_profile)
        else:
            profile = config.model_profile
            print(
                f"{profile.display_name} [{profile.profile_id}] "
                f"({profile.status.value}, {profile.model}:{profile.quantization})"
            )
        return
    if args.command == "check":
        profile = get_asr_profile(args.profile)
        print(profile.profile_id if args.id_only else profile.display_name)
        return

    profile = get_asr_profile(args.profile)
    path = write_installed_profile(profile.profile_id)
    print(f"Installed ASR profile: {profile.display_name}")
    print(f"Configuration: {path}")


if __name__ == "__main__":
    main()
