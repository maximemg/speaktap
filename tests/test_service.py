from __future__ import annotations

import pytest

from speaktap.domain import ServiceState, SessionOptions
from speaktap.service import InvalidTransitionError, ServiceStateMachine


def _machine() -> ServiceStateMachine:
    return ServiceStateMachine(id_factory=lambda: "session-1")


def _options() -> SessionOptions:
    return SessionOptions(asr_profile="fast", cleanup_enabled=False)


def test_happy_path() -> None:
    machine = _machine()

    started = machine.start(_options())
    finalizing = machine.begin_finalization()
    outputting = machine.begin_output()
    completed = machine.complete()

    assert started.state is ServiceState.RECORDING
    assert started.session_id == "session-1"
    assert finalizing.state is ServiceState.FINALIZING
    assert outputting.state is ServiceState.OUTPUTTING
    assert completed.state is ServiceState.IDLE
    assert completed.session_id is None


def test_profile_is_pinned_for_session() -> None:
    machine = _machine()
    options = SessionOptions(asr_profile="quality", cleanup_enabled=True)

    snapshot = machine.start(options)

    assert snapshot.options == options


def test_invalid_transition_rejected() -> None:
    machine = _machine()

    with pytest.raises(InvalidTransitionError, match="expected state recording"):
        machine.begin_finalization()


def test_invalid_session_id_does_not_mutate_state() -> None:
    machine = ServiceStateMachine(id_factory=lambda: "")

    with pytest.raises(ValueError, match="empty session ID"):
        machine.start(_options())

    assert machine.snapshot().state is ServiceState.IDLE


@pytest.mark.parametrize(
    "advance",
    [
        lambda machine: None,
        lambda machine: machine.begin_finalization(),
        lambda machine: (machine.begin_finalization(), machine.begin_output()),
    ],
)
def test_cancel_returns_active_session_to_idle(advance: object) -> None:
    machine = _machine()
    machine.start(_options())
    advance(machine)  # type: ignore[operator]

    snapshot = machine.cancel()

    assert snapshot.state is ServiceState.IDLE
    assert snapshot.session_id is None


def test_failure_records_error_and_returns_to_idle() -> None:
    machine = _machine()
    machine.start(_options())

    snapshot = machine.fail(RuntimeError("capture failed"))

    assert snapshot.state is ServiceState.IDLE
    assert snapshot.last_error == "capture failed"
