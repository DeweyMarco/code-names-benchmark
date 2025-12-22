from __future__ import annotations

import pandas as pd

from analysis.models import Game


def game_momentum(games: list[Game]) -> pd.DataFrame:
    rows = []
    for game in games:
        scores = {"blue": 0, "red": 0}
        lead_changes = 0
        prev_leader = None
        max_blue_lead = 0
        max_red_lead = 0
        turns = game.turns
        for turn in turns:
            correct = sum(1 for g in turn.guesses if g.correct)
            if turn.team in scores:
                scores[turn.team] += correct
            lead = scores["blue"] - scores["red"]
            max_blue_lead = max(max_blue_lead, lead)
            max_red_lead = max(max_red_lead, -lead)

            if scores["blue"] > scores["red"]:
                leader = "blue"
            elif scores["red"] > scores["blue"]:
                leader = "red"
            else:
                leader = None
            if leader and leader != prev_leader and prev_leader is not None:
                lead_changes += 1
            prev_leader = leader

        winner = game.winner
        was_comeback = False
        deficit_overcome = 0
        if winner == "blue" and max_red_lead > 0:
            was_comeback = True
            deficit_overcome = max_red_lead
        elif winner == "red" and max_blue_lead > 0:
            was_comeback = True
            deficit_overcome = max_blue_lead

        rows.append(
            {
                "game_id": game.game_id,
                "winner": winner,
                "total_turns": game.total_turns,
                "lead_changes": lead_changes,
                "was_comeback": was_comeback,
                "deficit_overcome": deficit_overcome,
                "max_blue_lead": max_blue_lead,
                "max_red_lead": max_red_lead,
                "competitiveness": lead_changes / max(len(turns), 1),
            }
        )
    return pd.DataFrame(rows)


def momentum_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    total_games = len(df)
    comebacks = df["was_comeback"].sum()
    return {
        "total_games_analyzed": total_games,
        "comeback_rate": comebacks / total_games if total_games else 0,
        "avg_lead_changes": df["lead_changes"].mean(),
        "max_lead_changes": df["lead_changes"].max(),
        "avg_deficit_overcome": df[df["was_comeback"]]["deficit_overcome"].mean() if comebacks else 0,
        "avg_competitiveness": df["competitiveness"].mean(),
        "avg_game_length": df["total_turns"].mean(),
        "blue_win_rate": (df["winner"] == "blue").mean(),
    }

