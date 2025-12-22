from __future__ import annotations

from typing import Tuple

import pandas as pd

from analysis.models import BenchmarkData


def team_combo_stats(data: BenchmarkData) -> pd.DataFrame:
    """Return normalized team combination stats as DataFrame."""
    return data.to_team_combo_df()


def role_performance(data: BenchmarkData) -> pd.DataFrame:
    """Aggregate model performance by role using game turn data."""
    model_stats = {}

    for game in data.games:
        blue_won = game.winner == "blue"
        red_won = game.winner == "red"
        roles_in_game = set()

        for turn in game.turns:
            team = turn.team
            model_hg = game.models.get(f"{team}_hint_giver", f"{team}_hint_giver")
            model_g = game.models.get(f"{team}_guesser", f"{team}_guesser")

            # Hint giver
            if turn.hint_word and turn.hint_count > 0:
                key = (model_hg, "hint_giver")
                roles_in_game.add((key, team))
                stats = model_stats.setdefault(
                    key,
                    {
                        "hints_given": 0,
                        "successful_hints": 0,
                        "guesses_made": 0,
                        "correct_guesses": 0,
                        "wrong_guesses": 0,
                        "bomb_hits": 0,
                        "games": 0,
                        "wins": 0,
                    },
                )
                stats["hints_given"] += 1
                correct_guesses = sum(1 for g in turn.guesses if g.correct)
                if correct_guesses > 0:
                    stats["successful_hints"] += 1

            # Guesser
            for guess in turn.guesses:
                key = (model_g, "guesser")
                roles_in_game.add((key, team))
                stats = model_stats.setdefault(
                    key,
                    {
                        "hints_given": 0,
                        "successful_hints": 0,
                        "guesses_made": 0,
                        "correct_guesses": 0,
                        "wrong_guesses": 0,
                        "bomb_hits": 0,
                        "games": 0,
                        "wins": 0,
                    },
                )
                stats["guesses_made"] += 1
                if guess.correct:
                    stats["correct_guesses"] += 1
                else:
                    stats["wrong_guesses"] += 1
                    if guess.color == "bomb" or guess.hit_bomb:
                        stats["bomb_hits"] += 1

        for (model_name, role), team in roles_in_game:
            stats = model_stats[(model_name, role)]
            stats["games"] += 1
            if (team == "blue" and blue_won) or (team == "red" and red_won):
                stats["wins"] += 1

    rows = []
    for (model_name, role), stats in model_stats.items():
        guesses = stats["guesses_made"]
        hints = stats["hints_given"]
        rows.append(
            {
                "model": model_name,
                "role": role,
                "games_played": stats["games"],
                "wins": stats["wins"],
                "win_rate": stats["wins"] / stats["games"] if stats["games"] else 0,
                "hint_success_rate": stats["successful_hints"] / hints if hints else 0,
                "guess_accuracy": stats["correct_guesses"] / guesses if guesses else 0,
                "bomb_hit_rate": stats["bomb_hits"] / guesses if guesses else 0,
                **stats,
            }
        )

    return pd.DataFrame(rows)


def best_hint_givers(combo_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top hint givers aggregated by team side."""
    all_hint_givers = set(combo_df["blue_hint_giver"].dropna()) | set(combo_df["red_hint_giver"].dropna())
    rows = []
    for model in all_hint_givers:
        if not model:
            continue
        blue = combo_df[combo_df["blue_hint_giver"] == model]
        red = combo_df[combo_df["red_hint_giver"] == model]
        for label, df, wins_col in (
            ("Blue Hint Giver", blue, "blue_wins"),
            ("Red Hint Giver", red, "red_wins"),
        ):
            if not df.empty:
                denom = df["blue_wins"] + df["red_wins"]
                rows.append(
                    {
                        "model": model,
                        "role": label,
                        "avg_win_rate": (df[wins_col] / denom).mean() if not denom.empty else 0,
                        "games_played": df["games_played"].sum(),
                        "total_wins": df[wins_col].sum(),
                    }
                )
    df = pd.DataFrame(rows)
    return df.nlargest(top_n, "avg_win_rate") if not df.empty else df


def best_guessers(combo_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top guessers aggregated by team side."""
    all_guessers = set(combo_df["blue_guesser"].dropna()) | set(combo_df["red_guesser"].dropna())
    rows = []
    for model in all_guessers:
        if not model:
            continue
        blue = combo_df[combo_df["blue_guesser"] == model]
        red = combo_df[combo_df["red_guesser"] == model]
        for label, df, wins_col in (
            ("Blue Guesser", blue, "blue_wins"),
            ("Red Guesser", red, "red_wins"),
        ):
            if not df.empty:
                denom = df["blue_wins"] + df["red_wins"]
                rows.append(
                    {
                        "model": model,
                        "role": label,
                        "avg_win_rate": (df[wins_col] / denom).mean() if not denom.empty else 0,
                        "games_played": df["games_played"].sum(),
                        "total_wins": df[wins_col].sum(),
                    }
                )
    df = pd.DataFrame(rows)
    return df.nlargest(top_n, "avg_win_rate") if not df.empty else df


def dominant_combos(combo_df: pd.DataFrame, min_games: int = 2, top_n: int = 20) -> pd.DataFrame:
    filtered = combo_df[combo_df["games_played"] >= min_games]
    if filtered.empty:
        return filtered
    return filtered.nlargest(top_n, "blue_win_rate")[
        ["blue_hint_giver", "blue_guesser", "red_hint_giver", "red_guesser", "blue_win_rate", "games_played", "avg_turns"]
    ]


def model_synergies(combo_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    rows = []
    for _, row in combo_df.iterrows():
        rows.append(
            {
                "hint_giver": row["blue_hint_giver"],
                "guesser": row["blue_guesser"],
                "combination": f"{row['blue_hint_giver']} + {row['blue_guesser']}",
                "win_rate": row["blue_win_rate"],
                "games_played": row["games_played"],
            }
        )
    df = pd.DataFrame(rows)
    grouped = df.groupby("combination").agg({"win_rate": "mean", "games_played": "sum"}).reset_index()
    return grouped.nlargest(top_n, "win_rate")


def first_mover_advantage(combo_df: pd.DataFrame) -> dict:
    if combo_df.empty:
        return {
            "overall_blue_win_rate": 0,
            "overall_red_win_rate": 0,
            "blue_advantage": 0,
            "mirror_match_blue_rate": None,
            "mirror_match_count": 0,
            "total_games": 0,
        }

    total_blue_wins = combo_df["blue_wins"].sum()
    total_red_wins = combo_df["red_wins"].sum()
    total_games = combo_df["games_played"].sum()

    overall_blue_rate = total_blue_wins / total_games if total_games else 0
    overall_red_rate = total_red_wins / total_games if total_games else 0

    mirror_matches = combo_df[
        (combo_df["blue_hint_giver"] == combo_df["red_hint_giver"]) &
        (combo_df["blue_guesser"] == combo_df["red_guesser"])
    ]
    mirror_blue_rate = None
    if not mirror_matches.empty:
        mirror_games = mirror_matches["games_played"].sum()
        mirror_blue_wins = mirror_matches["blue_wins"].sum()
        mirror_blue_rate = mirror_blue_wins / mirror_games if mirror_games else None

    return {
        "overall_blue_win_rate": overall_blue_rate,
        "overall_red_win_rate": overall_red_rate,
        "blue_advantage": overall_blue_rate - 0.5,
        "mirror_match_blue_rate": mirror_blue_rate,
        "mirror_match_count": len(mirror_matches),
        "total_games": total_games,
    }

