"""Thread-safe state transitions for the long-lived SpeakTap service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from uuid import uuid4

from .domain import ServiceState, SessionOptions


class InvalidTransitionError(RuntimeError):
    """Raised when a command is invalid for the current service state."""


@dataclass(frozen=True, slots=True)
class ServiceSnapshot:
    state: ServiceState
    session_id: str | None
    options: SessionOptions | None
    last_error: str | None


class ServiceStateMachine:
    """Own service state without depending on capture or model implementations."""

    def __init__(self, id_factory: Callable[[], str] | None = None) -> None:
        self._id_factory = id_factory or (lambda: uuid4().hex)
        self._lock = RLock()
        self._state = ServiceState.IDLE
        self._session_id: str | None = None
        self._options: SessionOptions | None = None
        self._last_error: str | None = None

    def snapshot(self) -> ServiceSnapshot:
        with self._lock:
            return ServiceSnapshot(
                state=self._state,
                session_id=self._session_id,
                options=self._options,
                last_error=self._last_error,
            )

    def start(self, options: SessionOptions) -> ServiceSnapshot:
        with self._lock:
            self._require(ServiceState.IDLE)
            session_id = self._id_factory()
            if not session_id:
                raise ValueError("id_factory returned an empty session ID")
            self._state = ServiceState.RECORDING
            self._session_id = session_id
            self._options = options
            self._last_error = None
            return self.snapshot()

    def begin_finalization(self) -> ServiceSnapshot:
        with self._lock:
            self._require(ServiceState.RECORDING)
            self._state = ServiceState.FINALIZING
            return self.snapshot()

    def begin_output(self) -> ServiceSnapshot:
        with self._lock:
            self._require(ServiceState.FINALIZING)
            self._state = ServiceState.OUTPUTTING
            return self.snapshot()

    def complete(self) -> ServiceSnapshot:
        with self._lock:
            self._require(ServiceState.OUTPUTTING)
            self._clear_session()
            return self.snapshot()

    def cancel(self) -> ServiceSnapshot:
        with self._lock:
            if self._state is ServiceState.IDLE:
                raise InvalidTransitionError("cannot cancel while idle")
            self._clear_session()
            return self.snapshot()

    def fail(self, error: BaseException | str) -> ServiceSnapshot:
        with self._lock:
            self._last_error = str(error)
            self._clear_session(clear_error=False)
            return self.snapshot()

    def _require(self, expected: ServiceState) -> None:
        if self._state is not expected:
            raise InvalidTransitionError(
                f"expected state {expected.value}, current state is {self._state.value}"
            )

    def _clear_session(self, *, clear_error: bool = True) -> None:
        self._state = ServiceState.IDLE
        self._session_id = None
        self._options = None
        if clear_error:
            self._last_error = None
