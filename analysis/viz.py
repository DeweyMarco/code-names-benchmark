from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from analysis.pipeline import AnalysisResult


def create_visualizations(result: AnalysisResult, output_dir: Path) -> None:
    """Generate visualization plots from the modular pipeline."""
    output_dir.mkdir(exist_ok=True)
    plt.style.use("default")
    sns.set_palette("husl")
    print("Generating visualizations...")

    combo_df = result.combo_df
    if not combo_df.empty:
        top_combos = combo_df.nlargest(15, "blue_win_rate")
        plt.figure(figsize=(14, 10))
        colors = ["#2ecc71" if rate > 0.5 else "#e74c3c" for rate in top_combos["blue_win_rate"]]
        plt.barh(range(len(top_combos)), top_combos["blue_win_rate"], color=colors)
        plt.yticks(
            range(len(top_combos)),
            [
                f"{row['blue_hint_giver'][:12]}+{row['blue_guesser'][:12]} vs {row['red_hint_giver'][:12]}+{row['red_guesser'][:12]}"
                for _, row in top_combos.iterrows()
            ],
            fontsize=8,
        )
        plt.xlabel("Blue Team Win Rate")
        plt.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="50% baseline")
        plt.title("Top 15 Team Combinations by Blue Win Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "team_combination_win_rates.png", dpi=300, bbox_inches="tight")
        plt.close()

    if not result.top_hint.empty or not result.top_guess.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        if not result.top_hint.empty:
            colors1 = sns.color_palette("Blues_r", len(result.top_hint))
            ax1.barh(range(len(result.top_hint)), result.top_hint["avg_win_rate"], color=colors1)
            ax1.set_yticks(range(len(result.top_hint)))
            ax1.set_yticklabels(
                [f"{row['model'][:15]} ({row['role'][:10]})" for _, row in result.top_hint.iterrows()], fontsize=9
            )
            ax1.set_xlabel("Average Win Rate")
            ax1.set_title("Best Hint Givers by Win Rate")
            ax1.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7)
        if not result.top_guess.empty:
            colors2 = sns.color_palette("Greens_r", len(result.top_guess))
            ax2.barh(range(len(result.top_guess)), result.top_guess["avg_win_rate"], color=colors2)
            ax2.set_yticks(range(len(result.top_guess)))
            ax2.set_yticklabels(
                [f"{row['model'][:15]} ({row['role'][:10]})" for _, row in result.top_guess.iterrows()], fontsize=9
            )
            ax2.set_xlabel("Average Win Rate")
            ax2.set_title("Best Guessers by Win Rate")
            ax2.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / "model_performance_by_role.png", dpi=300, bbox_inches="tight")
        plt.close()

    try:
        matchup_matrix, _ = result.matchup_matrix_hg
        if not matchup_matrix.empty and matchup_matrix.shape[0] > 1:
            plt.figure(figsize=(12, 10))
            short_names = [name[:12] for name in matchup_matrix.index]
            display_matrix = matchup_matrix.copy()
            display_matrix.index = short_names
            display_matrix.columns = short_names
            mask = np.isnan(display_matrix.values)
            sns.heatmap(
                display_matrix,
                annot=True,
                fmt=".0%",
                cmap="RdYlGn",
                center=0.5,
                mask=mask,
                square=True,
                linewidths=0.5,
                cbar_kws={"label": "Win Rate"},
            )
            plt.title("Head-to-Head Win Rate Matrix (Hint Givers)\nRow model vs Column model")
            plt.tight_layout()
            plt.savefig(output_dir / "matchup_heatmap.png", dpi=300, bbox_inches="tight")
            plt.close()
    except Exception as exc:  # pragma: no cover - visualization guard
        print(f"  Skipping matchup heatmap: {exc}")

    if not result.elo_df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        models = result.elo_df["model"].head(10)
        x = np.arange(len(models))
        width = 0.35
        hg_ratings = result.elo_df["elo_hint_giver"].head(10)
        g_ratings = result.elo_df["elo_guesser"].head(10)
        ax.bar(x - width / 2, hg_ratings, width, label="Hint Giver Elo", color="#3498db")
        ax.bar(x + width / 2, g_ratings, width, label="Guesser Elo", color="#2ecc71")
        ax.axhline(y=1500, color="gray", linestyle="--", alpha=0.7, label="Starting Elo (1500)")
        ax.set_xlabel("Model")
        ax.set_ylabel("Elo Rating")
        ax.set_title("Elo Ratings by Model and Role")
        ax.set_xticks(x)
        ax.set_xticklabels([m[:12] for m in models], rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(min(1400, min(hg_ratings.min(), g_ratings.min()) - 50), max(1600, max(hg_ratings.max(), g_ratings.max()) + 50))
        plt.tight_layout()
        plt.savefig(output_dir / "elo_ratings.png", dpi=300, bbox_inches="tight")
        plt.close()

    if not result.role_versatility.empty and len(result.role_versatility) > 1:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            result.role_versatility["hint_giver_win_rate"],
            result.role_versatility["guesser_win_rate"],
            s=result.role_versatility["total_games"] * 5,
            c=result.role_versatility["combined_win_rate"],
            cmap="RdYlGn",
            alpha=0.7,
            edgecolors="black",
            linewidths=1,
        )
        for _, row in result.role_versatility.iterrows():
            plt.annotate(row["model"][:10], (row["hint_giver_win_rate"], row["guesser_win_rate"]), fontsize=8, alpha=0.8, xytext=(5, 5), textcoords="offset points")
        plt.colorbar(scatter, label="Combined Win Rate")
        plt.xlabel("Hint Giver Win Rate")
        plt.ylabel("Guesser Win Rate")
        plt.title("Role Versatility: Hint Giver vs Guesser Performance\n(Size = Total Games)")
        lims = [0, 1]
        plt.plot(lims, lims, "k--", alpha=0.3, label="Equal performance")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "role_versatility.png", dpi=300, bbox_inches="tight")
        plt.close()

    hint_eff = result.hint_efficiency
    if not hint_eff.empty and len(hint_eff) > 1:
        plt.figure(figsize=(10, 8))
        risk_colors = {"aggressive": "#e74c3c", "balanced": "#f39c12", "conservative": "#3498db"}
        colors = [risk_colors.get(p, "#95a5a6") for p in hint_eff["risk_profile"]]
        plt.scatter(
            hint_eff["efficiency"],
            hint_eff["win_rate"],
            s=hint_eff["hints_given"] * 2,
            c=colors,
            alpha=0.7,
            edgecolors="black",
            linewidths=1,
        )
        for _, row in hint_eff.iterrows():
            plt.annotate(f"{row['model'][:8]} ({row['team'][0]})", (row["efficiency"], row["win_rate"]), fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points")
        for profile, color in risk_colors.items():
            plt.scatter([], [], c=color, label=profile.capitalize(), s=100)
        plt.xlabel("Hint Efficiency (Correct Guesses / Promised)")
        plt.ylabel("Win Rate")
        plt.title("Hint Giver Efficiency vs Win Rate\n(Size = Hints Given)")
        plt.legend(title="Risk Profile")
        plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "hint_efficiency_vs_winrate.png", dpi=300, bbox_inches="tight")
        plt.close()

    guesser_perf = result.guesser_perf
    if not guesser_perf.empty and len(guesser_perf) > 1:
        plt.figure(figsize=(10, 8))
        bomb_rates = guesser_perf["bomb_rate"]
        colors = plt.cm.Reds(bomb_rates / max(bomb_rates.max(), 0.01))
        plt.scatter(
            guesser_perf["first_guess_accuracy"],
            guesser_perf["overall_accuracy"],
            s=guesser_perf["games_played"] * 10,
            c=colors,
            alpha=0.7,
            edgecolors="black",
            linewidths=1,
        )
        for _, row in guesser_perf.iterrows():
            plt.annotate(f"{row['model'][:8]} ({row['team'][0]})", (row["first_guess_accuracy"], row["overall_accuracy"]), fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points")
        plt.xlabel("First Guess Accuracy")
        plt.ylabel("Overall Accuracy")
        plt.title("Guesser Performance: First Guess vs Overall\n(Size = Games, Color = Bomb Rate)")
        sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(0, bomb_rates.max()))
        sm.set_array([])
        plt.colorbar(sm, ax=plt.gca(), label="Bomb Hit Rate")
        plt.tight_layout()
        plt.savefig(output_dir / "guesser_accuracy.png", dpi=300, bbox_inches="tight")
        plt.close()

    ci_df = result.ci_df
    if not ci_df.empty:
        ci_top = ci_df.head(15)
        plt.figure(figsize=(12, 10))
        y_pos = range(len(ci_top))
        plt.hlines(y_pos, ci_top["ci_lower"], ci_top["ci_upper"], colors="#3498db", linewidth=2)
        plt.scatter(ci_top["win_rate"], y_pos, color="#e74c3c", s=100, zorder=5)
        plt.yticks(
            y_pos,
            [f"{row['model'][:12]} ({row['role'][:2]}, {row['team'][0]})" for _, row in ci_top.iterrows()],
            fontsize=9,
        )
        plt.xlabel("Win Rate (with 95% CI)")
        plt.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="50% baseline")
        plt.title("Win Rate Confidence Intervals (Wilson Score)")
        plt.legend()
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(output_dir / "confidence_intervals.png", dpi=300, bbox_inches="tight")
        plt.close()

    hp = result.hint_patterns
    if hp.get("hint_count_distribution"):
        plt.figure(figsize=(10, 6))
        counts = sorted(hp["hint_count_distribution"].items())
        x_vals = [c[0] for c in counts]
        y_vals = [c[1] for c in counts]
        plt.bar(x_vals, y_vals, color="#3498db", edgecolor="black")
        if hp["success_by_count"]:
            for count, freq in counts:
                if count in hp["success_by_count"]:
                    success = hp["success_by_count"][count]["success"]
                    plt.annotate(f"{success:.0%}", (count, freq), ha="center", va="bottom", fontsize=9, color="#27ae60")
        plt.xlabel("Hint Count (Number Promised)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Hint Counts\n(Green % = Success Rate)")
        plt.tight_layout()
        plt.savefig(output_dir / "hint_count_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

    error_df = result.error_summary
    if not error_df.empty and error_df["total_errors"].sum() > 0:
        error_df = error_df[error_df["total_errors"] > 0].head(10)
        if not error_df.empty:
            plt.figure(figsize=(12, 6))
            models = error_df["model"]
            x = np.arange(len(models))
            width = 0.6
            plt.bar(x, error_df["bomb_hits"], width, label="Bomb Hits", color="#e74c3c")
            plt.bar(x, error_df["invalid_offboard"], width, bottom=error_df["bomb_hits"], label="Invalid (Offboard)", color="#f39c12")
            plt.bar(
                x,
                error_df["invalid_revealed"],
                width,
                bottom=error_df["bomb_hits"] + error_df["invalid_offboard"],
                label="Invalid (Revealed)",
                color="#9b59b6",
            )
            plt.bar(
                x,
                error_df["invalid_other"],
                width,
                bottom=error_df["bomb_hits"] + error_df["invalid_offboard"] + error_df["invalid_revealed"],
                label="Invalid (Other)",
                color="#95a5a6",
            )
            plt.xlabel("Model")
            plt.ylabel("Error Count")
            plt.title("Error Breakdown by Model")
            plt.xticks(x, [m[:12] for m in models], rotation=45, ha="right")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "error_analysis.png", dpi=300, bbox_inches="tight")
            plt.close()

    momentum_df = result.momentum_df
    if not momentum_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(momentum_df["total_turns"], bins=15, color="#3498db", edgecolor="black", alpha=0.7)
        axes[0].axvline(momentum_df["total_turns"].mean(), color="#e74c3c", linestyle="--", label=f"Mean: {momentum_df['total_turns'].mean():.1f}")
        axes[0].set_xlabel("Total Turns")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Game Length Distribution")
        axes[0].legend()
        axes[1].hist(momentum_df["lead_changes"], bins=range(int(momentum_df["lead_changes"].max()) + 2), color="#2ecc71", edgecolor="black", alpha=0.7)
        axes[1].set_xlabel("Number of Lead Changes")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Game Competitiveness (Lead Changes)")
        plt.tight_layout()
        plt.savefig(output_dir / "game_dynamics.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Visualizations saved to {output_dir}/")

