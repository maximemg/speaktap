"""ASR architecture-adapter contracts."""

from .base import AsrBackend

__all__ = ["AsrBackend"]
from .onnx_backend import OnnxAsrBackend

__all__ = ["OnnxAsrBackend"]
