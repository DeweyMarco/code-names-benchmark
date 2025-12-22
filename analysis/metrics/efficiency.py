from __future__ import annotations

import pandas as pd

from analysis.models import BenchmarkData


def game_efficiency(data: BenchmarkData) -> pd.DataFrame:
    rows = []
    for combo in data.team_combinations.values():
        games = combo.games_played
        if games == 0:
            continue
        total_turns = combo.total_turns
        avg_turns = total_turns / games if games else 0
        blue_wins = combo.blue_wins
        red_wins = combo.red_wins
        blue_efficiency = blue_wins / total_turns if total_turns else 0
        red_efficiency = red_wins / total_turns if total_turns else 0
        rows.append(
            {
                "blue_hint_giver": combo.blue_hint_giver,
                "blue_guesser": combo.blue_guesser,
                "red_hint_giver": combo.red_hint_giver,
                "red_guesser": combo.red_guesser,
                "games_played": games,
                "avg_turns": round(avg_turns, 1),
                "blue_wins": blue_wins,
                "red_wins": red_wins,
                "blue_win_rate": round(blue_wins / games, 3) if games else 0,
                "blue_efficiency": round(blue_efficiency, 4),
                "red_efficiency": round(red_efficiency, 4),
                "total_turns": total_turns,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("blue_efficiency", ascending=False) if not df.empty else df


def efficiency_by_model(efficiency_df: pd.DataFrame) -> pd.DataFrame:
    if efficiency_df.empty:
        return pd.DataFrame()
    rows = []
    all_hint_givers = set(efficiency_df["blue_hint_giver"].unique()) | set(efficiency_df["red_hint_giver"].unique())
    for model in all_hint_givers:
        if not model:
            continue
        blue_rows = efficiency_df[efficiency_df["blue_hint_giver"] == model]
        red_rows = efficiency_df[efficiency_df["red_hint_giver"] == model]
        total_games = blue_rows["games_played"].sum() + red_rows["games_played"].sum()
        total_wins = blue_rows["blue_wins"].sum() + red_rows["red_wins"].sum()
        total_turns = blue_rows["total_turns"].sum() + red_rows["total_turns"].sum()
        avg_turns_to_win = total_turns / total_wins if total_wins else float("inf")
        rows.append(
            {
                "model": model,
                "role": "hint_giver",
                "total_games": total_games,
                "total_wins": total_wins,
                "win_rate": round(total_wins / total_games, 3) if total_games else 0,
                "avg_turns_per_game": round(total_turns / total_games, 1) if total_games else 0,
                "avg_turns_to_win": round(avg_turns_to_win, 1) if avg_turns_to_win != float("inf") else None,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("win_rate", ascending=False) if not df.empty else df

