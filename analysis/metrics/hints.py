from __future__ import annotations

from collections import Counter

import pandas as pd

from analysis.models import BenchmarkData


def hint_patterns(data: BenchmarkData) -> dict:
    hints = []
    hint_words = []
    for game in data.games:
        for turn in game.turns:
            hint_word = turn.hint_word
            hint_count = turn.hint_count
            guesses = turn.guesses
            if hint_word and hint_count > 0:
                correct_count = sum(1 for g in guesses if g.correct)
                hints.append(
                    {
                        "word": hint_word.lower(),
                        "count": hint_count,
                        "team": turn.team,
                        "turn_number": turn.turn_number,
                        "guesses_made": len(guesses),
                        "correct_guesses": correct_count,
                        "success": correct_count > 0,
                        "perfect": correct_count >= hint_count,
                        "word_length": len(hint_word),
                        "efficiency": correct_count / hint_count if hint_count > 0 else 0,
                    }
                )
                hint_words.append(hint_word.lower())

    if not hints:
        return {
            "total_hints": 0,
            "unique_hints": 0,
            "creativity_ratio": 0,
            "avg_hint_length": 0,
            "avg_hint_count": 0,
            "overall_success_rate": 0,
            "perfect_hint_rate": 0,
            "hint_count_distribution": {},
            "most_common_hints": [],
            "success_by_count": {},
            "avg_efficiency": 0,
        }

    df = pd.DataFrame(hints)
    hint_counter = Counter(hint_words)
    success_by_count = (
        df.groupby("count")
        .agg({"success": "mean", "perfect": "mean", "efficiency": "mean", "word": "count"})
        .rename(columns={"word": "occurrences"})
        .to_dict("index")
    )
    return {
        "total_hints": len(hints),
        "unique_hints": len(set(hint_words)),
        "creativity_ratio": len(set(hint_words)) / len(hints),
        "avg_hint_length": df["word_length"].mean(),
        "avg_hint_count": df["count"].mean(),
        "overall_success_rate": df["success"].mean(),
        "perfect_hint_rate": df["perfect"].mean(),
        "avg_efficiency": df["efficiency"].mean(),
        "hint_count_distribution": dict(Counter(df["count"])),
        "most_common_hints": hint_counter.most_common(15),
        "success_by_count": success_by_count,
    }

