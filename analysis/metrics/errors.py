from __future__ import annotations

from collections import defaultdict
from typing import Dict

import pandas as pd

from analysis.models import BenchmarkData


def error_patterns(data: BenchmarkData) -> dict:
    error_analysis = {
        "bomb_hits_by_model": defaultdict(int),
        "bomb_contexts": [],
        "invalid_by_type": defaultdict(lambda: {"offboard": 0, "revealed": 0, "other": 0}),
        "wrong_guess_colors": defaultdict(lambda: {"neutral": 0, "opponent": 0, "bomb": 0}),
        "total_errors_by_model": defaultdict(int),
    }

    for perf in data.model_performance.values():
        if perf.role == "guesser":
            model = perf.model
            error_analysis["bomb_hits_by_model"][model] += perf.bomb_hits
            error_analysis["invalid_by_type"][model]["offboard"] += perf.invalid_offboard
            error_analysis["invalid_by_type"][model]["revealed"] += perf.invalid_revealed
            error_analysis["invalid_by_type"][model]["other"] += perf.invalid_other
            total_errors = (
                perf.bomb_hits + perf.invalid_offboard + perf.invalid_revealed + perf.invalid_other
            )
            error_analysis["total_errors_by_model"][model] += total_errors

    for game in data.games:
        for turn in game.turns:
            for guess in turn.guesses:
                if guess.hit_bomb:
                    error_analysis["bomb_contexts"].append(
                        {
                            "game_id": game.game_id,
                            "turn": turn.turn_number,
                            "team": turn.team,
                            "word": guess.word,
                            "hint_word": turn.hint_word,
                            "hint_count": turn.hint_count,
                        }
                    )
                elif not guess.correct:
                    color = guess.color or "unknown"
                    team = turn.team.lower()
                    if color == "neutral":
                        error_analysis["wrong_guess_colors"][team]["neutral"] += 1
                    elif color == "bomb":
                        error_analysis["wrong_guess_colors"][team]["bomb"] += 1
                    else:
                        error_analysis["wrong_guess_colors"][team]["opponent"] += 1
    return error_analysis


def error_summary(errors: dict) -> pd.DataFrame:
    rows = []
    all_models = set(errors["bomb_hits_by_model"].keys()) | set(errors["invalid_by_type"].keys())
    for model in all_models:
        bombs = errors["bomb_hits_by_model"].get(model, 0)
        invalid = errors["invalid_by_type"].get(model, {"offboard": 0, "revealed": 0, "other": 0})
        total = errors["total_errors_by_model"].get(model, 0)
        rows.append(
            {
                "model": model,
                "bomb_hits": bombs,
                "invalid_offboard": invalid["offboard"],
                "invalid_revealed": invalid["revealed"],
                "invalid_other": invalid["other"],
                "total_errors": total,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("total_errors", ascending=True) if not df.empty else df

