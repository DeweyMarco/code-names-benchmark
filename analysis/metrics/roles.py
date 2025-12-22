from __future__ import annotations

import pandas as pd


def hint_efficiency(model_perf_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, perf in model_perf_df.iterrows():
        if perf.get("role") != "hint_giver" or perf.get("hints_given", 0) <= 0:
            continue
        hints_given = perf["hints_given"]
        hint_count_total = perf.get("hint_count_total", 0)
        correct_guesses = perf.get("correct_guesses", 0)
        successful_hints = perf.get("successful_hints", 0)
        wrong_guesses = perf.get("wrong_guesses", 0)

        avg_hint_count = hint_count_total / hints_given if hints_given else 0
        guess_yield = correct_guesses / hints_given if hints_given else 0
        efficiency = correct_guesses / hint_count_total if hint_count_total else 0
        hint_success_rate = successful_hints / hints_given if hints_given else 0

        if avg_hint_count > 2.5:
            risk_profile = "aggressive"
        elif avg_hint_count < 1.5:
            risk_profile = "conservative"
        else:
            risk_profile = "balanced"

        ambiguity_rate = wrong_guesses / hints_given if hints_given else 0

        rows.append(
            {
                "model": perf["model"],
                "team": perf.get("team", "unknown"),
                "hints_given": hints_given,
                "avg_hint_count": round(avg_hint_count, 2),
                "guess_yield": round(guess_yield, 2),
                "efficiency": round(efficiency, 3),
                "hint_success_rate": round(hint_success_rate, 3),
                "risk_profile": risk_profile,
                "overcommit_rate": round(1 - efficiency, 3),
                "ambiguity_rate": round(ambiguity_rate, 2),
                "win_rate": perf.get("wins", 0) / perf.get("games_played", 1),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("efficiency", ascending=False) if not df.empty else df


def guesser_performance(model_perf_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, perf in model_perf_df.iterrows():
        if perf.get("role") != "guesser" or perf.get("guesses_made", 0) <= 0:
            continue
        guesses_made = perf["guesses_made"]
        correct_guesses = perf.get("correct_guesses", 0)
        bomb_hits = perf.get("bomb_hits", 0)
        first_attempts = perf.get("first_guess_attempts", 0)
        first_correct = perf.get("first_guess_correct", 0)
        turns_played = perf.get("turns_played", 1)
        empty_turns = perf.get("empty_turns", 0)

        invalid_offboard = perf.get("invalid_offboard", 0)
        invalid_revealed = perf.get("invalid_revealed", 0)
        invalid_other = perf.get("invalid_other", 0)
        total_invalid = invalid_offboard + invalid_revealed + invalid_other

        first_guess_accuracy = first_correct / first_attempts if first_attempts else 0
        overall_accuracy = correct_guesses / guesses_made if guesses_made else 0
        bomb_rate = bomb_hits / guesses_made if guesses_made else 0
        invalid_rate = total_invalid / guesses_made if guesses_made else 0
        guesses_per_turn = guesses_made / turns_played if turns_played else 0
        empty_turn_rate = empty_turns / turns_played if turns_played else 0
        risk_adjusted = (correct_guesses - 3 * bomb_hits) / guesses_made if guesses_made else 0

        rows.append(
            {
                "model": perf["model"],
                "team": perf.get("team", "unknown"),
                "games_played": perf.get("games_played", 0),
                "first_guess_accuracy": round(first_guess_accuracy, 3),
                "overall_accuracy": round(overall_accuracy, 3),
                "bomb_rate": round(bomb_rate, 4),
                "bomb_hits": bomb_hits,
                "invalid_rate": round(invalid_rate, 4),
                "invalid_breakdown": {
                    "offboard": invalid_offboard,
                    "revealed": invalid_revealed,
                    "other": invalid_other,
                },
                "guesses_per_turn": round(guesses_per_turn, 2),
                "empty_turn_rate": round(empty_turn_rate, 3),
                "risk_adjusted_accuracy": round(risk_adjusted, 3),
                "win_rate": perf.get("wins", 0) / perf.get("games_played", 1),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("risk_adjusted_accuracy", ascending=False) if not df.empty else df


def role_versatility(model_perf_df: pd.DataFrame) -> pd.DataFrame:
    grouped = model_perf_df.groupby(["model", "role"]).agg({"wins": "sum", "games_played": "sum"}).reset_index()
    models = grouped["model"].unique()
    rows = []
    for model in models:
        hg = grouped.loc[(grouped["model"] == model) & (grouped["role"] == "hint_giver")]
        g = grouped.loc[(grouped["model"] == model) & (grouped["role"] == "guesser")]
        hg_games = int(hg["games_played"].sum()) if not hg.empty else 0
        g_games = int(g["games_played"].sum()) if not g.empty else 0
        hg_rate = hg["wins"].sum() / hg_games if hg_games else 0
        g_rate = g["wins"].sum() / g_games if g_games else 0
        if hg_rate + g_rate > 0:
            versatility = 1 - abs(hg_rate - g_rate) / max(hg_rate, g_rate)
        else:
            versatility = 0
        total_wins = hg["wins"].sum() + g["wins"].sum()
        total_games = hg_games + g_games
        combined_win_rate = total_wins / total_games if total_games else 0
        rows.append(
            {
                "model": model,
                "hint_giver_win_rate": round(hg_rate, 3),
                "hint_giver_games": hg_games,
                "guesser_win_rate": round(g_rate, 3),
                "guesser_games": g_games,
                "versatility_score": round(versatility, 3),
                "best_role": "hint_giver" if hg_rate > g_rate else "guesser" if g_rate > hg_rate else "equal",
                "role_gap": round(abs(hg_rate - g_rate), 3),
                "combined_win_rate": round(combined_win_rate, 3),
                "total_games": total_games,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("combined_win_rate", ascending=False) if not df.empty else df


def matchup_matrix(combo_df: pd.DataFrame, by_role: str = "hint_giver") -> tuple[pd.DataFrame, pd.DataFrame]:
    models = set()
    if by_role == "hint_giver":
        models.update(combo_df["blue_hint_giver"].dropna().tolist())
        models.update(combo_df["red_hint_giver"].dropna().tolist())
    else:
        models.update(combo_df["blue_guesser"].dropna().tolist())
        models.update(combo_df["red_guesser"].dropna().tolist())
    models = sorted([m for m in models if m])

    win_counts = pd.DataFrame(0.0, index=models, columns=models)
    game_counts = pd.DataFrame(0, index=models, columns=models)

    for _, combo in combo_df.iterrows():
        if by_role == "hint_giver":
            blue_model = combo["blue_hint_giver"]
            red_model = combo["red_hint_giver"]
        else:
            blue_model = combo["blue_guesser"]
            red_model = combo["red_guesser"]
        games = combo.get("games_played", 0)
        blue_wins = combo.get("blue_wins", 0)
        red_wins = combo.get("red_wins", 0)
        if blue_model and red_model and games > 0:
            win_counts.loc[blue_model, red_model] += blue_wins
            win_counts.loc[red_model, blue_model] += red_wins
            game_counts.loc[blue_model, red_model] += games
            game_counts.loc[red_model, blue_model] += games

    with pd.option_context("mode.use_inf_as_na", True):
        win_rate_matrix = win_counts / game_counts.replace(0, pd.NA)
    return win_rate_matrix, game_counts
