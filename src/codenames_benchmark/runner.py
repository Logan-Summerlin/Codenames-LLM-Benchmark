"""Game and matchup runners."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from codenames_benchmark.agents.aggregate import aggregate_guesser_actions
from codenames_benchmark.boards import generate_board, mirror_board
from codenames_benchmark.game import Board, GameState, Identity, Team
from codenames_benchmark.legality import check_clue
from codenames_benchmark.protocol import build_guesser_observation, build_spymaster_observation

@dataclass(frozen=True)
class GameRecord:
    team_red: str; team_blue: str; seed: int | str; winner: str | None; terminal: bool; reason: str | None; board: dict[str, Any]; public_events: list[dict[str, Any]]; private_events: list[dict[str, Any]]
    def to_dict(self):
        return {"team_red":self.team_red,"team_blue":self.team_blue,"seed":self.seed,"winner":self.winner,"terminal":self.terminal,"reason":self.reason,"board":self.board,"public_events":[dict(e) for e in self.public_events],"private_events":[dict(e) for e in self.private_events]}

def _remaining_targets(game: GameState, team: Team, limit: int) -> list[str]:
    ident = Identity.for_team(team)
    return [w for w, i in game.board.words.items() if i is ident and w not in game.board.revealed][:limit]

def _run_on_board(red_team, blue_team, board: Board, *, seed: int | str, max_turns: int = 50) -> GameRecord:
    game = GameState.new(board, starting_team=Team.RED)
    teams = {Team.RED: red_team, Team.BLUE: blue_team}
    private=[]
    for _ in range(max_turns):
        if game.winner:
            break
        active = game.current_team
        team = teams[active]
        spy_obs = build_spymaster_observation(game, team=active, agent_id=team.spymaster.agent_id)
        clue_action = team.spymaster.choose_clue(spy_obs)
        if clue_action is None:
            game.history.append({"event": "invalid_clue_format", "team": active.value, "agent_id": team.spymaster.agent_id, "reason": "invalid_spymaster_output"})
            game._end_turn()
            continue
        private.append({"event": "spymaster_action", "team": active.value, "agent_id": team.spymaster.agent_id, "action": clue_action.to_dict()})
        legality = check_clue(clue_action.clue, game.board)
        if not legality.legal:
            game.history.append({"event": "illegal_clue", "team": active.value, "clue": clue_action.to_dict(), "reason": legality.reason, "matched_word": legality.matched_word})
            game._end_turn()
            continue
        game.give_clue(clue_action.to_clue())
        targets = _remaining_targets(game, active, game.guesses_remaining)
        guess_actions = []
        for guesser in team.guessers:
            obs = build_guesser_observation(game, team=active, agent_id=guesser.agent_id)
            guesser_observation = _ObservationProxy(obs, targets) if getattr(guesser, "requires_oracle_targets", False) else obs
            action = guesser.choose_guesses(guesser_observation)
            guess_actions.append(action)
            private.append({"event": "guesser_action", "team": active.value, "agent_id": guesser.agent_id, "action": action.to_dict()})
        agg = aggregate_guesser_actions(guess_actions, guesses_remaining=game.guesses_remaining, unavailable_words=set(game.board.revealed))
        valid_guesses = [word for word in agg.guesses if word in game.board.words and word not in game.board.revealed]
        for word in valid_guesses:
            result = game.guess(word)
            if result.terminal:
                break
            if game.phase.value == "awaiting_clue":
                break
        if game.phase.value == "guessing":
            game.stop_guessing()
    return GameRecord(red_team.name, blue_team.name, seed, game.winner.value if game.winner else None, bool(game.winner), game.terminal_reason, game.board.to_dict(include_hidden=True), game.history, private)

class _ObservationProxy:
    def __init__(self, obs, oracle_targets): self.obs=obs; self.oracle_targets=oracle_targets
    def to_dict(self):
        d=self.obs.to_dict(); d["oracle_targets"]=list(self.oracle_targets); return d

def run_game(red_team, blue_team, *, seed: int | str, max_turns: int = 50) -> GameRecord:
    return _run_on_board(red_team, blue_team, generate_board(seed=seed, starting_team=Team.RED), seed=seed, max_turns=max_turns)

def run_mirrored_matchup(red_team, blue_team, *, seed: int | str, max_turns: int = 50) -> list[GameRecord]:
    board = generate_board(seed=seed, starting_team=Team.RED)
    first = _run_on_board(red_team, blue_team, board, seed=f"{seed}:0", max_turns=max_turns)
    second = _run_on_board(red_team, blue_team, mirror_board(board), seed=f"{seed}:1", max_turns=max_turns)
    return [first, second]
