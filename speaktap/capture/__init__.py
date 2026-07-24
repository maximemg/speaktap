"""Audio-source adapters."""

from .base import AudioSource

__all__ = ["AudioSource"]
from .file import AudioFileSource
from .linux_arecord import ArecordSource

__all__ = ["ArecordSource", "AudioFileSource"]
