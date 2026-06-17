"""Deterministic mock agents for zero-cost simulator testing."""
from __future__ import annotations
from dataclasses import dataclass, field
from codenames_benchmark.game import Identity, Team
from codenames_benchmark.protocol import GuesserAction, GuesserObservation, SpymasterAction, SpymasterObservation

@dataclass
class OracleSpymasterAgent:
    agent_id: str
    team: Team
    max_count: int = 2
    def choose_clue(self, observation: SpymasterObservation) -> SpymasterAction:
        data = observation.to_dict()
        identity = self.team.value
        revealed = set(data["board"].get("revealed", []))
        targets = [w for w, ident in data["board"]["identities"].items() if ident == identity and w not in revealed]
        count = max(1, min(self.max_count, len(targets)))
        return SpymasterAction(clue=f"{identity}clue", count=count, rationale=",".join(targets[:count]))

@dataclass
class OracleGuesserAgent:
    requires_oracle_targets = True
    agent_id: str
    team: Team
    def choose_guesses(self, observation: GuesserObservation) -> GuesserAction:
        # Deterministic oracle baseline: uses a private injected payload when the runner supplies one.
        data = observation.to_dict()
        targets = list(data.get("oracle_targets", []))
        if not targets:
            words = [w for w in data["board"]["words"] if w not in set(data["board"].get("revealed", []))]
            targets = words[:1]
        return GuesserAction(targets, {w: 1.0 for w in targets}, stop=False, rationale="deterministic oracle baseline")

@dataclass
class DeterministicMockTeam:
    name: str
    team: Team
    spymaster: OracleSpymasterAgent = field(init=False)
    guessers: list[OracleGuesserAgent] = field(init=False)
    def __post_init__(self):
        if not isinstance(self.team, Team): self.team = Team(self.team)
        self.spymaster = OracleSpymasterAgent(f"{self.name}-spymaster", self.team)
        self.guessers = [OracleGuesserAgent(f"{self.name}-guesser-{i}", self.team) for i in range(3)]
