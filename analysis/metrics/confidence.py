from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def wilson_ci(model_perf_df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Calculate Wilson score intervals for win rates."""
    if model_perf_df.empty:
        return pd.DataFrame()

    rows = []
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    for _, perf in model_perf_df.iterrows():
        n = perf.get("games_played", 0)
        wins = perf.get("wins", 0)
        if n == 0:
            continue
        p = wins / n
        denominator = 1 + z ** 2 / n
        center = (p + z ** 2 / (2 * n)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denominator
        rows.append(
            {
                "model": perf["model"],
                "role": perf.get("role", ""),
                "team": perf.get("team", ""),
                "win_rate": p,
                "ci_lower": max(0, center - margin),
                "ci_upper": min(1, center + margin),
                "ci_width": 2 * margin,
                "sample_size": n,
                "wins": wins,
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("win_rate", ascending=False) if not df.empty else df


def pairwise_significance(model_perf_df: pd.DataFrame, model_a: str, model_b: str, role: str | None = None) -> dict:
    """Chi-squared test for significant difference between two models."""
    def _aggregate(model: str):
        subset = model_perf_df[(model_perf_df["model"] == model)]
        if role:
            subset = subset[subset["role"] == role]
        wins = subset["wins"].sum()
        games = subset["games_played"].sum()
        return wins, games

    wins_a, games_a = _aggregate(model_a)
    wins_b, games_b = _aggregate(model_b)

    if games_a == 0 or games_b == 0:
        return {"error": "Insufficient data", "p_value": None, "significant": False}

    contingency = [[wins_a, games_a - wins_a], [wins_b, games_b - wins_b]]
    try:
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        return {
            "model_a": model_a,
            "model_b": model_b,
            "win_rate_a": wins_a / games_a,
            "win_rate_b": wins_b / games_b,
            "chi2": chi2,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "sample_size_a": games_a,
            "sample_size_b": games_b,
        }
    except Exception as exc:  # pragma: no cover - safety
        return {"error": str(exc), "p_value": None, "significant": False}


def all_pairwise(model_perf_df: pd.DataFrame) -> pd.DataFrame:
    models = sorted(model_perf_df["model"].dropna().unique())
    rows = []
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            res = pairwise_significance(model_perf_df, model_a, model_b)
            if "error" not in res:
                rows.append(res)
    return pd.DataFrame(rows)

