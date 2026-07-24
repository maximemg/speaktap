"""Result-output adapter contracts."""

from .base import OutputAdapter

__all__ = ["OutputAdapter"]
from .linux import deliver_outputs, make_outputs, notify_status, play_sound

__all__ = ["deliver_outputs", "make_outputs", "notify_status", "play_sound"]
