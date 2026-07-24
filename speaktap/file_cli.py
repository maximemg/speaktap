"""Safe file-based end-to-end runner: no microphone and no desktop output."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from .capture import AudioFileSource
from .config import SpeakTapConfig
from .factory import make_pipeline
from .transcription import OnnxAsrBackend


def main() -> None:
    parser = argparse.ArgumentParser(prog="speaktap-transcribe-file")
    parser.add_argument("audio", type=Path, help="16 kHz mono PCM16 WAV or raw file")
    parser.add_argument("--language")
    args = parser.parse_args()

    config = SpeakTapConfig.load()
    language = config.language if args.language is None else args.language
    backend = OnnxAsrBackend(
        profile=config.model_profile,
        provider=config.execution_provider,
        threads=config.asr_threads,
        inter_op_threads=config.asr_inter_op_threads,
        execution_mode=config.asr_execution_mode,
    )
    backend.load()
    backend.warmup()
    file_config = replace(
        config,
        language=language,
    )
    pipeline = make_pipeline(
        file_config,
        backend,
    )
    pipeline.start(
        AudioFileSource(
            args.audio,
            sample_rate=config.sample_rate,
            frame_milliseconds=config.frame_milliseconds,
        ),
        session_id=uuid4().hex,
        cleanup_enabled=False,
    )
    try:
        result = pipeline.wait()
    finally:
        backend.close()
    print(result.output_text)


if __name__ == "__main__":
    main()
