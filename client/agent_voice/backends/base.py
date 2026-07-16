"""Agent backend contract for the agent_voice loop."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class AgentEvent:
    """One event from the agent.

    kind: 'init' (text = session id; backend-internal), 'delta' (text to
    speak), 'tool' (text = tool name), 'turn_end', 'fatal' (text = reason),
    'tick' — periodic heartbeat while idle; consumers use it to poll flags
    and MUST otherwise ignore it.
    """

    kind: str
    text: str = ""


class AgentBackend(ABC):
    """A pluggable agent brain driven over stdio."""

    @abstractmethod
    def start(self) -> None:
        """Spawn the long-lived agent process; raise RuntimeError on failure."""

    @abstractmethod
    def send(self, text: str) -> None:
        """Send one user utterance, starting a turn."""

    @abstractmethod
    def events(self) -> Iterator[AgentEvent]:
        """Yield events across turns; the loop consumes until 'turn_end'."""

    @abstractmethod
    def cancel(self) -> None:
        """Interrupt the in-flight turn (best effort).

        After cancel(), the backend MUST still deliver a terminal 'turn_end'
        (or 'fatal') through events() — the loop blocks draining until one
        arrives.
        """

    @abstractmethod
    def stop(self) -> None:
        """Terminate the agent process gracefully."""
