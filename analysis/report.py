from __future__ import annotations

import pandas as pd

from analysis.pipeline import AnalysisResult


def _format_top_models(df: pd.DataFrame, label: str, limit: int = 5) -> list[str]:
    lines = [f"## {label}", ""]
    if df.empty:
        lines.append("_No data available._")
        lines.append("")
        return lines
    for idx, (_, row) in enumerate(df.head(limit).iterrows(), 1):
        win_rate = row.get("win_rate") or row.get("avg_win_rate") or row.get("elo_combined")
        suffix = ""
        if "games_played" in row:
            suffix = f" ({row['games_played']} games)"
        lines.append(f"{idx}. **{row['model']}**: {win_rate:.3f}{suffix}")
    lines.append("")
    return lines


def build_markdown_report(result: AnalysisResult) -> str:
    lines: list[str] = []
    lines.append("# Codenames Benchmark Insights")
    lines.append(f"Benchmark ID: `{result.benchmark_id}`")
    lines.append(f"Games: {result.combo_df['games_played'].sum() if not result.combo_df.empty else 0}")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Blue win rate: {result.fma['overall_blue_win_rate']:.1%} ({result.fma['total_games']} games)")
    lines.append(f"- Red win rate: {result.fma['overall_red_win_rate']:.1%}")
    if result.fma.get("mirror_match_blue_rate") is not None:
        lines.append(f"- Mirror matches (same models): Blue wins {result.fma['mirror_match_blue_rate']:.1%}")
    if result.momentum_summary:
        lines.append(f"- Comeback rate: {result.momentum_summary['comeback_rate']:.1%}")
        lines.append(f"- Avg game length: {result.momentum_summary['avg_game_length']:.1f} turns")
    lines.append("")

    if not result.elo_df.empty:
        lines.append("## Elo Rankings (Top 10)")
        for i, (_, row) in enumerate(result.elo_df.head(10).iterrows(), 1):
            lines.append(
                f"{i}. **{row['model']}**: {row['elo_combined']} (HG: {row['elo_hint_giver']}, "
                f"G: {row['elo_guesser']}, best: {row['elo_best_role']})"
            )
        lines.append("")

    lines.extend(_format_top_models(result.top_hint.rename(columns={"avg_win_rate": "win_rate"}), "Best Hint Givers"))
    lines.extend(_format_top_models(result.top_guess.rename(columns={"avg_win_rate": "win_rate"}), "Best Guessers"))

    lines.append("## Dominant Combinations")
    lines.append("")
    if result.dominant.empty:
        lines.append("_No combinations meeting the filter._")
    else:
        for i, (_, row) in enumerate(result.dominant.iterrows(), 1):
            lines.append(
                f"{i}. **{row['blue_hint_giver']} + {row['blue_guesser']}** vs "
                f"**{row['red_hint_giver']} + {row['red_guesser']}** — "
                f"Blue win rate: {row['blue_win_rate']:.1%} ({row['games_played']} games)"
            )
    lines.append("")

    lines.append("## Top Synergies")
    lines.append("")
    if result.synergies.empty:
        lines.append("_No synergy data available._")
    else:
        for _, row in result.synergies.head(5).iterrows():
            lines.append(f"- **{row['combination']}**: {row['win_rate']:.1%} ({row['games_played']} games)")
    lines.append("")

    lines.append("## Hint Giver Efficiency")
    lines.append("")
    if result.hint_efficiency.empty:
        lines.append("_No hint giver data._")
    else:
        lines.append("| Model | Team | Avg Hint | Yield | Efficiency | Risk | Win Rate |")
        lines.append("|-------|------|----------|-------|------------|------|----------|")
        for _, row in result.hint_efficiency.iterrows():
            lines.append(
                f"| {row['model']} | {row['team']} | {row['avg_hint_count']:.1f} | "
                f"{row['guess_yield']:.2f} | {row['efficiency']:.1%} | {row['risk_profile']} | "
                f"{row['win_rate']:.1%} |"
            )
    lines.append("")

    lines.append("## Guesser Performance")
    lines.append("")
    if result.guesser_perf.empty:
        lines.append("_No guesser data._")
    else:
        lines.append("| Model | Team | 1st Guess | Overall | Bomb Rate | Risk-Adj | Win Rate |")
        lines.append("|-------|------|-----------|---------|-----------|----------|----------|")
        for _, row in result.guesser_perf.iterrows():
            lines.append(
                f"| {row['model']} | {row['team']} | {row['first_guess_accuracy']:.1%} | "
                f"{row['overall_accuracy']:.1%} | {row['bomb_rate']:.2%} | "
                f"{row['risk_adjusted_accuracy']:.1%} | {row['win_rate']:.1%} |"
            )
    lines.append("")

    lines.append("## Confidence (95% Wilson CI)")
    lines.append("")
    if result.ci_df.empty:
        lines.append("_No confidence intervals available._")
    else:
        for _, row in result.ci_df.head(10).iterrows():
            lines.append(
                f"- **{row['model']}** ({row['role']}): {row['win_rate']:.1%} "
                f"[{row['ci_lower']:.1%} - {row['ci_upper']:.1%}] (n={row['sample_size']})"
            )
    lines.append("")

    lines.append("## Error Analysis")
    lines.append("")
    if result.error_summary.empty:
        lines.append("_No error data._")
    else:
        for _, row in result.error_summary.iterrows():
            if row["total_errors"] > 0:
                lines.append(
                    f"- **{row['model']}**: {row['total_errors']} total errors "
                    f"(bombs: {row['bomb_hits']}, invalid: {row['invalid_offboard'] + row['invalid_revealed'] + row['invalid_other']})"
                )
    if result.error_patterns.get("bomb_contexts"):
        lines.append("")
        lines.append("### Recent Bomb Hits (Context)")
        for ctx in result.error_patterns["bomb_contexts"][:5]:
            lines.append(
                f"- Turn {ctx['turn']}: '{ctx['word']}' guessed after hint '{ctx['hint_word']}' ({ctx['hint_count']})"
            )
    lines.append("")

    lines.append("## Hint Word Patterns")
    lines.append("")
    hp = result.hint_patterns
    if hp.get("total_hints", 0) == 0:
        lines.append("_No hint data._")
    else:
        lines.append(f"- Total hints: {hp['total_hints']}")
        lines.append(f"- Unique hints: {hp['unique_hints']} ({hp['creativity_ratio']:.1%} creativity)")
        lines.append(f"- Avg hint count: {hp['avg_hint_count']:.2f}")
        lines.append(f"- Overall success rate: {hp['overall_success_rate']:.1%}")
        lines.append(f"- Perfect hint rate: {hp['perfect_hint_rate']:.1%}")
        if hp["most_common_hints"]:
            lines.append("")
            lines.append("### Most Common Hints")
            for word, count in hp["most_common_hints"][:10]:
                lines.append(f"- '{word}': {count} times")
        if hp["success_by_count"]:
            lines.append("")
            lines.append("### Success Rate by Hint Count")
            for count, stats in sorted(hp["success_by_count"].items()):
                lines.append(
                    f"- **{count}**: {stats['success']:.1%} success, {stats['efficiency']:.1%} efficiency ({stats['occurrences']} hints)"
                )
    lines.append("")

    lines.append("## Game Dynamics")
    lines.append("")
    if result.momentum_summary:
        ms = result.momentum_summary
        lines.append(f"- Average lead changes: {ms['avg_lead_changes']:.1f}")
        lines.append(f"- Max lead changes: {ms['max_lead_changes']}")
        lines.append(f"- Average competitiveness: {ms['avg_competitiveness']:.2f}")
        if ms["comeback_rate"] > 0:
            lines.append(f"- Average deficit overcome: {ms['avg_deficit_overcome']:.1f} cards")
    lines.append("")

    lines.append("## Game Efficiency (Speed of Victory)")
    lines.append("")
    if result.efficiency_by_model.empty:
        lines.append("_No efficiency data._")
    else:
        for _, row in result.efficiency_by_model.head(5).iterrows():
            turns_to_win = row["avg_turns_to_win"]
            turns_str = f"{turns_to_win:.1f}" if turns_to_win else "N/A"
            lines.append(
                f"- **{row['model']}**: {row['win_rate']:.1%} win rate, "
                f"{row['avg_turns_per_game']:.1f} avg turns/game, {turns_str} turns/win"
            )
    lines.append("")

    lines.append("_Generated by the modular pipeline._")
    lines.append("")

    return "\n".join(lines)
