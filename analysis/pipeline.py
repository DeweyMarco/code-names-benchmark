from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analysis.loader import load_benchmark_results
from analysis.metrics import confidence, elo, momentum, win_rates
from analysis.metrics import efficiency as efficiency_metrics
from analysis.metrics import errors as error_metrics
from analysis.metrics import hints as hint_metrics
from analysis.metrics import roles as role_metrics


@dataclass
class AnalysisResult:
    benchmark_id: str
    combo_df: pd.DataFrame
    role_perf_df: pd.DataFrame
    model_perf_df: pd.DataFrame
    elo_df: pd.DataFrame
    ci_df: pd.DataFrame
    fma: dict
    momentum_df: pd.DataFrame
    momentum_summary: dict
    top_hint: pd.DataFrame
    top_guess: pd.DataFrame
    dominant: pd.DataFrame
    synergies: pd.DataFrame
    hint_efficiency: pd.DataFrame
    guesser_perf: pd.DataFrame
    role_versatility: pd.DataFrame
    error_summary: pd.DataFrame
    error_patterns: dict
    hint_patterns: dict
    game_efficiency: pd.DataFrame
    efficiency_by_model: pd.DataFrame
    matchup_matrix_hg: tuple[pd.DataFrame, pd.DataFrame]
    matchup_matrix_g: tuple[pd.DataFrame, pd.DataFrame]


def run_pipeline(results_path: Path) -> AnalysisResult:
    data = load_benchmark_results(results_path)
    combo_df = win_rates.team_combo_stats(data)
    role_perf_df = win_rates.role_performance(data)
    model_perf_df = data.to_model_perf_df()
    elo_df = elo.compute_elo(data)
    ci_df = confidence.wilson_ci(model_perf_df)
    fma = win_rates.first_mover_advantage(combo_df)
    momentum_df = momentum.game_momentum(data.games)
    momentum_summary = momentum.momentum_summary(momentum_df)
    top_hint = win_rates.best_hint_givers(combo_df, 10)
    top_guess = win_rates.best_guessers(combo_df, 10)
    dominant = win_rates.dominant_combos(combo_df, top_n=20)
    synergies = win_rates.model_synergies(combo_df)
    hint_eff = role_metrics.hint_efficiency(model_perf_df)
    guesser_perf = role_metrics.guesser_performance(model_perf_df)
    role_vers = role_metrics.role_versatility(model_perf_df)
    errors = error_metrics.error_patterns(data)
    error_df = error_metrics.error_summary(errors)
    hints = hint_metrics.hint_patterns(data)
    game_eff = efficiency_metrics.game_efficiency(data)
    eff_by_model = efficiency_metrics.efficiency_by_model(game_eff)
    matchup_hg = role_metrics.matchup_matrix(combo_df, by_role="hint_giver")
    matchup_g = role_metrics.matchup_matrix(combo_df, by_role="guesser")

    return AnalysisResult(
        benchmark_id=data.benchmark_id,
        combo_df=combo_df,
        role_perf_df=role_perf_df,
        model_perf_df=model_perf_df,
        elo_df=elo_df,
        ci_df=ci_df,
        fma=fma,
        momentum_df=momentum_df,
        momentum_summary=momentum_summary,
        top_hint=top_hint,
        top_guess=top_guess,
        dominant=dominant,
        synergies=synergies,
        hint_efficiency=hint_eff,
        guesser_perf=guesser_perf,
        role_versatility=role_vers,
        error_summary=error_df,
        error_patterns=errors,
        hint_patterns=hints,
        game_efficiency=game_eff,
        efficiency_by_model=eff_by_model,
        matchup_matrix_hg=matchup_hg,
        matchup_matrix_g=matchup_g,
    )

