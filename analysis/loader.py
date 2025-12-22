from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from analysis.models import BenchmarkData, Game, TeamCombination, ModelPerformance


def load_benchmark_results(path: Path) -> BenchmarkData:
    """Load and normalize benchmark results into typed structures."""
    payload: Dict = json.loads(Path(path).read_text())

    games = [Game.from_raw(g) for g in payload.get("games", []) or []]

    team_combinations = {
        key: TeamCombination.from_raw(key, value)
        for key, value in (payload.get("team_combinations", {}) or {}).items()
    }

    model_performance = {
        key: ModelPerformance.from_raw(value)
        for key, value in (payload.get("model_performance", {}) or {}).items()
    }

    return BenchmarkData(
        games=games,
        team_combinations=team_combinations,
        model_performance=model_performance,
        benchmark_id=payload.get("benchmark_id", "unknown"),
        raw=payload,
    )

