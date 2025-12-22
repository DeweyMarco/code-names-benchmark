"""
Benchmark Results Analysis Tool (Modular Pipeline)
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from analysis.pipeline import run_pipeline
from analysis.report import build_markdown_report
from analysis.viz import create_visualizations


def main() -> None:
    parser = ArgumentParser(description="Analyze benchmark results.")
    parser.add_argument("results_file", help="Path to benchmark results JSON")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    try:
        result = run_pipeline(results_path)
        insights = build_markdown_report(result)

        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        print(insights)

        insights_file = results_path.parent / f"{results_path.stem}_insights.md"
        insights_file.write_text(insights)
        print(f"\nInsights report saved to: {insights_file}")

        if not args.no_plots:
            create_visualizations(result, results_path.parent / "analysis_plots")

        print("\nAnalysis complete!")
    except Exception as exc:
        print(f"Error during analysis: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
