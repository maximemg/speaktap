# Security policy

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do not
open a public issue containing an unpatched vulnerability, private transcript,
audio, access token, or local filesystem details.

Include the affected commit, impact, reproduction steps, and any suggested
mitigation. You should receive an acknowledgement within seven days.

## Supported versions

Until SpeakTap reaches a stable release, only the latest commit on `main` is
supported.

## Local-data boundary

SpeakTap processes audio and text locally. Reports should nevertheless treat
session diagnostics under `~/.local/state/speaktap/` and the Hugging Face cache
as sensitive local data. The project never needs model-hub write credentials.
