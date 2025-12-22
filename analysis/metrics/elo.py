from __future__ import annotations

from collections import defaultdict
from typing import Dict

import pandas as pd

from analysis.models import BenchmarkData


def compute_elo(data: BenchmarkData, k_factor: float = 32, initial_rating: float = 1500) -> pd.DataFrame:
    """Calculate Elo ratings for models in hint/guesser roles."""
    ratings: Dict[str, float] = defaultdict(lambda: initial_rating)

    games_have_models = any(game.models for game in data.games)

    if games_have_models:
        for game in data.games:
            winner = game.winner
            if not winner or not game.models:
                continue
            blue_hg = game.models.get("blue_hint_giver")
            blue_g = game.models.get("blue_guesser")
            red_hg = game.models.get("red_hint_giver")
            red_g = game.models.get("red_guesser")
            if not all([blue_hg, blue_g, red_hg, red_g]):
                continue

            blue_rating = (ratings[f"{blue_hg}_hg"] + ratings[f"{blue_g}_g"]) / 2
            red_rating = (ratings[f"{red_hg}_hg"] + ratings[f"{red_g}_g"]) / 2
            expected_blue = 1 / (1 + 10 ** ((red_rating - blue_rating) / 400))
            expected_red = 1 - expected_blue

            actual_blue = 1 if winner == "blue" else 0
            actual_red = 1 - actual_blue

            blue_delta = k_factor * (actual_blue - expected_blue)
            red_delta = k_factor * (actual_red - expected_red)

            ratings[f"{blue_hg}_hg"] += blue_delta / 2
            ratings[f"{blue_g}_g"] += blue_delta / 2
            ratings[f"{red_hg}_hg"] += red_delta / 2
            ratings[f"{red_g}_g"] += red_delta / 2
    else:
        for combo in data.team_combinations.values():
            if not (combo.blue_hint_giver and combo.blue_guesser and combo.red_hint_giver and combo.red_guesser):
                continue
            blue_hg, blue_g, red_hg, red_g = (
                combo.blue_hint_giver,
                combo.blue_guesser,
                combo.red_hint_giver,
                combo.red_guesser,
            )

            for _ in range(combo.blue_wins):
                blue_rating = (ratings[f"{blue_hg}_hg"] + ratings[f"{blue_g}_g"]) / 2
                red_rating = (ratings[f"{red_hg}_hg"] + ratings[f"{red_g}_g"]) / 2
                expected_blue = 1 / (1 + 10 ** ((red_rating - blue_rating) / 400))
                blue_delta = k_factor * (1 - expected_blue)
                red_delta = k_factor * (0 - (1 - expected_blue))
                ratings[f"{blue_hg}_hg"] += blue_delta / 2
                ratings[f"{blue_g}_g"] += blue_delta / 2
                ratings[f"{red_hg}_hg"] += red_delta / 2
                ratings[f"{red_g}_g"] += red_delta / 2

            for _ in range(combo.red_wins):
                blue_rating = (ratings[f"{blue_hg}_hg"] + ratings[f"{blue_g}_g"]) / 2
                red_rating = (ratings[f"{red_hg}_hg"] + ratings[f"{red_g}_g"]) / 2
                expected_blue = 1 / (1 + 10 ** ((red_rating - blue_rating) / 400))
                blue_delta = k_factor * (0 - expected_blue)
                red_delta = k_factor * (1 - (1 - expected_blue))
                ratings[f"{blue_hg}_hg"] += blue_delta / 2
                ratings[f"{blue_g}_g"] += blue_delta / 2
                ratings[f"{red_hg}_hg"] += red_delta / 2
                ratings[f"{red_g}_g"] += red_delta / 2

    results = []
    model_roles = defaultdict(lambda: {"hint_giver": None, "guesser": None})
    for key, rating in ratings.items():
        if key.endswith("_hg"):
            model = key[:-3]
            model_roles[model]["hint_giver"] = rating
        elif key.endswith("_g"):
            model = key[:-2]
            model_roles[model]["guesser"] = rating

    for model, role_ratings in model_roles.items():
        hg_rating = role_ratings["hint_giver"] or initial_rating
        g_rating = role_ratings["guesser"] or initial_rating
        results.append(
            {
                "model": model,
                "elo_hint_giver": round(hg_rating),
                "elo_guesser": round(g_rating),
                "elo_combined": round((hg_rating + g_rating) / 2),
                "elo_best_role": "hint_giver" if hg_rating > g_rating else "guesser",
            }
        )

    df = pd.DataFrame(results)
    return df.sort_values("elo_combined", ascending=False) if not df.empty else df

