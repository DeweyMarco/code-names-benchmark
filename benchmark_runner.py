"""
Benchmark Runner: Automate running N games with basic statistics.

This module provides functionality to run multiple games and collect
basic statistics about game outcomes, performance, and agent behavior.
"""
import os
import json
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from game import Board, Team, GameOutcome
from orchestrator import GameRunner, GameResult
from agents import HintGiver, Guesser
from utils import generate_word_list
from config import Config


@dataclass
class BenchmarkStats:
    """Statistics collected from running multiple games."""
    total_games: int
    blue_wins: int
    red_wins: int
    total_turns: int
    avg_turns_per_game: float
    min_turns: int
    max_turns: int
    blue_avg_score: float  # Average remaining words for blue
    red_avg_score: float   # Average remaining words for red
    errors: int
    total_time: float  # Total time in seconds
    avg_time_per_game: float

    @property
    def blue_win_rate(self) -> float:
        """Calculate blue team win rate."""
        return self.blue_wins / self.total_games if self.total_games > 0 else 0.0

    @property
    def red_win_rate(self) -> float:
        """Calculate red team win rate."""
        return self.red_wins / self.total_games if self.total_games > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'total_games': self.total_games,
            'blue_wins': self.blue_wins,
            'red_wins': self.red_wins,
            'blue_win_rate': self.blue_win_rate,
            'red_win_rate': self.red_win_rate,
            'total_turns': self.total_turns,
            'avg_turns_per_game': self.avg_turns_per_game,
            'min_turns': self.min_turns,
            'max_turns': self.max_turns,
            'blue_avg_score': self.blue_avg_score,
            'red_avg_score': self.red_avg_score,
            'errors': self.errors,
            'total_time': self.total_time,
            'avg_time_per_game': self.avg_time_per_game
        }

    def __str__(self) -> str:
        """Pretty print statistics."""
        return f"""
Benchmark Statistics
{'='*60}
Total Games: {self.total_games}
Errors: {self.errors}

Win Rates:
  Blue Wins: {self.blue_wins} ({self.blue_win_rate:.1%})
  Red Wins:  {self.red_wins} ({self.red_win_rate:.1%})

Turn Statistics:
  Total Turns: {self.total_turns}
  Avg Turns/Game: {self.avg_turns_per_game:.1f}
  Min Turns: {self.min_turns}
  Max Turns: {self.max_turns}

Final Scores (avg remaining words):
  Blue: {self.blue_avg_score:.1f}
  Red:  {self.red_avg_score:.1f}

Performance:
  Total Time: {self.total_time:.1f}s
  Avg Time/Game: {self.avg_time_per_game:.1f}s
{'='*60}
"""


@dataclass
class BenchmarkResult:
    """Complete result of a benchmark run."""
    benchmark_id: str
    stats: BenchmarkStats
    game_results: List[GameResult]
    config: Dict[str, Any]
    agent_names: Dict[str, str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            'benchmark_id': self.benchmark_id,
            'timestamp': self.timestamp.isoformat(),
            'stats': self.stats.to_dict(),
            'agent_names': self.agent_names,
            'config': self.config,
            'games': [game.to_dict() for game in self.game_results]
        }

    def save(self, output_dir: str = "benchmark_results"):
        """Save benchmark results to JSON file."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        filename = f"{self.benchmark_id}.json"
        filepath = Path(output_dir) / filename

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        return filepath


class BenchmarkRunner:
    """
    Run multiple games and collect statistics.

    This runner automates running N games with the same agent configuration
    and collects basic statistics about outcomes and performance.
    """

    def __init__(
        self,
        blue_hint_giver_factory: Callable[[], HintGiver],
        blue_guesser_factory: Callable[[], Guesser],
        red_hint_giver_factory: Callable[[], HintGiver],
        red_guesser_factory: Callable[[], Guesser],
        config: Optional[Config] = None,
        verbose: bool = False,
        save_individual_games: bool = True
    ):
        """
        Initialize benchmark runner.

        Args:
            blue_hint_giver_factory: Factory function to create blue hint giver
            blue_guesser_factory: Factory function to create blue guesser
            red_hint_giver_factory: Factory function to create red hint giver
            red_guesser_factory: Factory function to create red guesser
            config: Game configuration (uses default if None)
            verbose: Print progress during benchmark
            save_individual_games: Save each game result to JSON
        """
        self.blue_hint_giver_factory = blue_hint_giver_factory
        self.blue_guesser_factory = blue_guesser_factory
        self.red_hint_giver_factory = red_hint_giver_factory
        self.red_guesser_factory = red_guesser_factory
        self.config = config or Config.default()
        self.verbose = verbose
        self.save_individual_games = save_individual_games

    def _log(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)

    def run_single_game(self, game_number: int) -> GameResult:
        """Run a single game and return the result."""
        # Generate new board for each game
        words = generate_word_list(self.config.game.BOARD_SIZE)
        board = Board(words, config=self.config.game)

        # Create fresh agents for each game
        blue_hint_giver = self.blue_hint_giver_factory()
        blue_guesser = self.blue_guesser_factory()
        red_hint_giver = self.red_hint_giver_factory()
        red_guesser = self.red_guesser_factory()

        # Create game runner
        runner = GameRunner(
            board=board,
            blue_hint_giver=blue_hint_giver,
            blue_guesser=blue_guesser,
            red_hint_giver=red_hint_giver,
            red_guesser=red_guesser,
            max_turns=self.config.game.MAX_TURNS,
            verbose=False,  # Individual games are quiet
            game_id=f"benchmark_game_{game_number}"
        )

        # Run the game
        result = runner.run()

        return result

    def run(self, num_games: int) -> BenchmarkResult:
        """
        Run N games and collect statistics.

        Args:
            num_games: Number of games to run

        Returns:
            BenchmarkResult with statistics and all game results
        """
        self._log(f"\n{'='*60}")
        self._log(f"STARTING BENCHMARK: {num_games} games")
        self._log(f"{'='*60}\n")

        # Initialize tracking
        game_results: List[GameResult] = []
        start_time = time.time()

        # Run all games
        for i in range(1, num_games + 1):
            self._log(f"Running game {i}/{num_games}...")

            game_start = time.time()
            result = self.run_single_game(i)
            game_time = time.time() - game_start

            game_results.append(result)

            # Log individual game result
            outcome_str = result.outcome.value
            winner_str = result.winner.value if result.winner else "None"
            self._log(f"  → Game {i}: {outcome_str} (Winner: {winner_str}, "
                     f"Turns: {result.total_turns}, Time: {game_time:.1f}s)")

            if result.error:
                self._log(f"  → Error: {result.error}")

        total_time = time.time() - start_time

        # Calculate statistics
        stats = self._calculate_stats(game_results, total_time)

        # Create benchmark result
        benchmark_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Get agent names (from first game)
        if game_results:
            agent_names = {
                'blue_hint_giver': game_results[0].blue_hint_giver_name,
                'blue_guesser': game_results[0].blue_guesser_name,
                'red_hint_giver': game_results[0].red_hint_giver_name,
                'red_guesser': game_results[0].red_guesser_name
            }
        else:
            agent_names = {
                'blue_hint_giver': 'unknown',
                'blue_guesser': 'unknown',
                'red_hint_giver': 'unknown',
                'red_guesser': 'unknown'
            }

        benchmark_result = BenchmarkResult(
            benchmark_id=benchmark_id,
            stats=stats,
            game_results=game_results if self.save_individual_games else [],
            config=self.config.to_dict(),
            agent_names=agent_names
        )

        # Print summary
        self._log(f"\n{stats}")

        return benchmark_result

    def _calculate_stats(
        self,
        game_results: List[GameResult],
        total_time: float
    ) -> BenchmarkStats:
        """Calculate statistics from game results."""
        total_games = len(game_results)

        # Count outcomes
        blue_wins = sum(1 for r in game_results if r.outcome == GameOutcome.BLUE_WIN)
        red_wins = sum(1 for r in game_results if r.outcome == GameOutcome.RED_WIN)
        errors = sum(1 for r in game_results if r.error is not None)

        # Turn statistics
        turns_list = [r.total_turns for r in game_results]
        total_turns = sum(turns_list)
        avg_turns = total_turns / total_games if total_games > 0 else 0
        min_turns = min(turns_list) if turns_list else 0
        max_turns = max(turns_list) if turns_list else 0

        # Score statistics (remaining words)
        blue_scores = [r.final_scores[0] for r in game_results]
        red_scores = [r.final_scores[1] for r in game_results]
        blue_avg = sum(blue_scores) / total_games if total_games > 0 else 0
        red_avg = sum(red_scores) / total_games if total_games > 0 else 0

        # Time statistics
        avg_time = total_time / total_games if total_games > 0 else 0

        return BenchmarkStats(
            total_games=total_games,
            blue_wins=blue_wins,
            red_wins=red_wins,
            total_turns=total_turns,
            avg_turns_per_game=avg_turns,
            min_turns=min_turns,
            max_turns=max_turns,
            blue_avg_score=blue_avg,
            red_avg_score=red_avg,
            errors=errors,
            total_time=total_time,
            avg_time_per_game=avg_time
        )


def run_benchmark(
    num_games: int,
    blue_hint_giver_factory: Callable[[], HintGiver],
    blue_guesser_factory: Callable[[], Guesser],
    red_hint_giver_factory: Callable[[], HintGiver],
    red_guesser_factory: Callable[[], Guesser],
    output_file: Optional[str] = None,
    verbose: bool = True
) -> BenchmarkResult:
    """
    Convenience function to run a benchmark.

    Args:
        num_games: Number of games to run
        blue_hint_giver_factory: Factory for blue hint giver
        blue_guesser_factory: Factory for blue guesser
        red_hint_giver_factory: Factory for red hint giver
        red_guesser_factory: Factory for red guesser
        output_file: Optional path to save results
        verbose: Print progress

    Returns:
        BenchmarkResult with statistics
    """
    runner = BenchmarkRunner(
        blue_hint_giver_factory=blue_hint_giver_factory,
        blue_guesser_factory=blue_guesser_factory,
        red_hint_giver_factory=red_hint_giver_factory,
        red_guesser_factory=red_guesser_factory,
        verbose=verbose
    )

    result = runner.run(num_games)

    # Save if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        if verbose:
            print(f"\nResults saved to: {output_file}")
    else:
        # Save to default location
        filepath = result.save()
        if verbose:
            print(f"\nResults saved to: {filepath}")

    return result
