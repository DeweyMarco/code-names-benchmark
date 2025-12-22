from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from model_config import get_model_display_name, BAMLModel


def clean_model_name(name: str) -> str:
    """Convert internal model names to human-readable display names."""
    if not name:
        return name
    for model in BAMLModel:
        if model.value == name:
            return get_model_display_name(model)
    if name.startswith("OpenRouter"):
        return name[len("OpenRouter") :]
    return name


@dataclass
class Guess:
    word: str
    correct: bool
    color: Optional[str]
    hit_bomb: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "Guess":
        return cls(
            word=raw.get("word", ""),
            correct=bool(raw.get("correct", False)),
            color=raw.get("color"),
            hit_bomb=bool(raw.get("hit_bomb", False)),
            metadata={k: v for k, v in raw.items() if k not in {"word", "correct", "color", "hit_bomb"}},
        )


@dataclass
class Turn:
    team: str
    hint_word: str
    hint_count: int
    guesses: List[Guess]
    turn_number: int

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "Turn":
        return cls(
            team=raw.get("team", "").lower(),
            hint_word=raw.get("hint_word", ""),
            hint_count=int(raw.get("hint_count", 0) or 0),
            guesses=[Guess.from_raw(g) for g in raw.get("guesses", [])],
            turn_number=int(raw.get("turn_number", 0) or 0),
        )


@dataclass
class Game:
    game_id: str
    winner: Optional[str]
    total_turns: int
    turns: List[Turn]
    models: Dict[str, str]
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "Game":
        snapshot = raw.get("snapshot", {}) or {}
        turn_history = snapshot.get("turn_history", []) or []
        models_raw = raw.get("models", {}) or {}
        return cls(
            game_id=str(raw.get("game_id", "")),
            winner=(raw.get("winner") or "").lower() or None,
            total_turns=int(raw.get("total_turns", len(turn_history))),
            turns=[Turn.from_raw(t) for t in turn_history],
            models={k: clean_model_name(v) if isinstance(v, str) else v for k, v in models_raw.items()},
            raw=raw,
        )


@dataclass
class TeamCombination:
    combination_key: str
    blue_hint_giver: str
    blue_guesser: str
    red_hint_giver: str
    red_guesser: str
    games_played: int
    blue_wins: int
    red_wins: int
    draws: int
    total_turns: int
    errors: int

    @classmethod
    def from_raw(cls, key: str, raw: Dict[str, Any]) -> "TeamCombination":
        return cls(
            combination_key=key,
            blue_hint_giver=clean_model_name(raw.get("blue_hint_giver", "")),
            blue_guesser=clean_model_name(raw.get("blue_guesser", "")),
            red_hint_giver=clean_model_name(raw.get("red_hint_giver", "")),
            red_guesser=clean_model_name(raw.get("red_guesser", "")),
            games_played=int(raw.get("games_played", 0)),
            blue_wins=int(raw.get("blue_wins", 0)),
            red_wins=int(raw.get("red_wins", 0)),
            draws=int(raw.get("draws", 0)),
            total_turns=int(raw.get("total_turns", 0)),
            errors=int(raw.get("errors", 0)),
        )


@dataclass
class ModelPerformance:
    model: str
    role: str
    team: str
    wins: int
    games_played: int
    hints_given: int = 0
    hint_count_total: int = 0
    successful_hints: int = 0
    guesses_made: int = 0
    correct_guesses: int = 0
    wrong_guesses: int = 0
    bomb_hits: int = 0
    first_guess_attempts: int = 0
    first_guess_correct: int = 0
    turns_played: int = 0
    empty_turns: int = 0
    invalid_offboard: int = 0
    invalid_revealed: int = 0
    invalid_other: int = 0

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "ModelPerformance":
        return cls(
            model=clean_model_name(raw.get("model", "")),
            role=raw.get("role", ""),
            team=raw.get("team", ""),
            wins=int(raw.get("wins", 0)),
            games_played=int(raw.get("games_played", 0)),
            hints_given=int(raw.get("hints_given", 0)),
            hint_count_total=int(raw.get("hint_count_total", 0)),
            successful_hints=int(raw.get("successful_hints", 0)),
            guesses_made=int(raw.get("guesses_made", 0)),
            correct_guesses=int(raw.get("correct_guesses", 0)),
            wrong_guesses=int(raw.get("wrong_guesses", 0)),
            bomb_hits=int(raw.get("bomb_hits", 0)),
            first_guess_attempts=int(raw.get("first_guess_attempts", 0)),
            first_guess_correct=int(raw.get("first_guess_correct", 0)),
            turns_played=int(raw.get("turns_played", 0)),
            empty_turns=int(raw.get("empty_turns", 0)),
            invalid_offboard=int(raw.get("invalid_offboard", 0)),
            invalid_revealed=int(raw.get("invalid_revealed", 0)),
            invalid_other=int(raw.get("invalid_other", 0)),
        )


@dataclass
class BenchmarkData:
    games: List[Game]
    team_combinations: Dict[str, TeamCombination]
    model_performance: Dict[str, ModelPerformance]
    benchmark_id: str
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_team_combo_df(self) -> pd.DataFrame:
        rows = []
        for combo in self.team_combinations.values():
            games = combo.games_played
            denom = combo.blue_wins + combo.red_wins
            blue_win_rate = combo.blue_wins / denom if denom else 0
            red_win_rate = combo.red_wins / denom if denom else 0
            avg_turns = combo.total_turns / games if games else 0
            rows.append(
                {
                    "combination": combo.combination_key,
                    "blue_hint_giver": combo.blue_hint_giver,
                    "blue_guesser": combo.blue_guesser,
                    "red_hint_giver": combo.red_hint_giver,
                    "red_guesser": combo.red_guesser,
                    "games_played": games,
                    "blue_wins": combo.blue_wins,
                    "red_wins": combo.red_wins,
                    "draws": combo.draws,
                    "blue_win_rate": blue_win_rate,
                    "red_win_rate": red_win_rate,
                    "avg_turns": avg_turns,
                    "errors": combo.errors,
                    "total_turns": combo.total_turns,
                }
            )
        return pd.DataFrame(rows)

    def to_model_perf_df(self) -> pd.DataFrame:
        rows = []
        for perf in self.model_performance.values():
            rows.append(
                {
                    "model": perf.model,
                    "role": perf.role,
                    "team": perf.team,
                    "wins": perf.wins,
                    "games_played": perf.games_played,
                    "hints_given": perf.hints_given,
                    "hint_count_total": perf.hint_count_total,
                    "successful_hints": perf.successful_hints,
                    "guesses_made": perf.guesses_made,
                    "correct_guesses": perf.correct_guesses,
                    "wrong_guesses": perf.wrong_guesses,
                    "bomb_hits": perf.bomb_hits,
                    "first_guess_attempts": perf.first_guess_attempts,
                    "first_guess_correct": perf.first_guess_correct,
                    "turns_played": perf.turns_played,
                    "empty_turns": perf.empty_turns,
                    "invalid_offboard": perf.invalid_offboard,
                    "invalid_revealed": perf.invalid_revealed,
                    "invalid_other": perf.invalid_other,
                }
            )
        return pd.DataFrame(rows)

    def turns_dataframe(self) -> pd.DataFrame:
        rows = []
        for game in self.games:
            for turn in game.turns:
                correct_guesses = sum(1 for g in turn.guesses if g.correct)
                bomb_hits = sum(1 for g in turn.guesses if g.hit_bomb)
                rows.append(
                    {
                        "game_id": game.game_id,
                        "team": turn.team,
                        "hint_word": turn.hint_word,
                        "hint_count": turn.hint_count,
                        "turn_number": turn.turn_number,
                        "guesses_made": len(turn.guesses),
                        "correct_guesses": correct_guesses,
                        "bomb_hits": bomb_hits,
                        "winner": game.winner,
                    }
                )
        return pd.DataFrame(rows)
