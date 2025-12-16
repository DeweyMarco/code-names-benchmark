"""
Quick Codenames Benchmark: Test Key Model Combinations

This is a more practical version of the comprehensive benchmark that tests
a focused set of model combinations to get quick insights about model performance.

Instead of testing all 625 combinations (5^4), this tests:
- Each model as hint giver vs each model as guesser (25 combinations)
- Key strategic combinations
- Total: ~50-100 games instead of 1,875 games
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from game import Board, Team, GameOutcome
from orchestrator import GameRunner, GameResult
from agents.llm import BAMLHintGiver, BAMLGuesser, BAMLModel
from utils import generate_word_list
from config import Config
from model_config import get_benchmark_models, get_model_display_name

# Load environment variables
load_dotenv()

# ============================================================================
# QUICK BENCHMARK CONFIGURATION
# ============================================================================

# Models to test (as specified by user)
BENCHMARK_MODELS = get_benchmark_models()

# Quick benchmark settings
GAMES_PER_COMBINATION = 2  # Reduced for quick results
OUTPUT_DIR = "quick_benchmark_results"
VERBOSE = True

@dataclass
class QuickBenchmarkResult:
    """Results from quick benchmark."""
    benchmark_id: str
    timestamp: datetime
    total_combinations: int
    total_games: int
    model_performance: Dict[str, Any]
    team_combinations: Dict[str, Any]
    game_results: List[GameResult]
    config: Dict[str, Any]
    
    def save(self, output_dir: str = OUTPUT_DIR):
        """Save results to JSON."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        result_dict = {
            'benchmark_id': self.benchmark_id,
            'timestamp': self.timestamp.isoformat(),
            'total_combinations': self.total_combinations,
            'total_games': self.total_games,
            'model_performance': self.model_performance,
            'team_combinations': self.team_combinations,
            'config': self.config,
            'games': [game.to_dict() for game in self.game_results]
        }
        
        # Save main results
        main_file = Path(output_dir) / f"{self.benchmark_id}_quick.json"
        with open(main_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        return main_file

class QuickBenchmarkRunner:
    """Run quick benchmark with focused model combinations."""
    
    def __init__(self, games_per_combination: int = GAMES_PER_COMBINATION, verbose: bool = VERBOSE):
        self.games_per_combination = games_per_combination
        self.verbose = verbose
        self.config = Config.default()
        
        # Initialize tracking
        self.model_performance: Dict[str, Any] = {}
        self.team_combinations: Dict[str, Any] = {}
        self.game_results: List[GameResult] = []
        
        # Initialize model performance tracking
        self._initialize_model_performance()
    
    def _log(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)
    
    def _initialize_model_performance(self):
        """Initialize performance tracking for all models."""
        for model in BENCHMARK_MODELS:
            for role in ["hint_giver", "guesser"]:
                for team in ["blue", "red"]:
                    key = f"{model.value}_{role}_{team}"
                    self.model_performance[key] = {
                        'model': model.value,
                        'role': role,
                        'team': team,
                        'games_played': 0,
                        'wins': 0,
                        'turns_played': 0,
                        'empty_turns': 0,
                        'first_guess_attempts': 0,
                        'first_guess_correct': 0,
                        'hint_count_total': 0,
                        'hints_given': 0,
                        'successful_hints': 0,
                        'guesses_made': 0,
                        'correct_guesses': 0,
                        'wrong_guesses': 0,
                        'bomb_hits': 0,
                        'invalid_offboard': 0,
                        'invalid_revealed': 0,
                        'invalid_other': 0
                    }
    
    def _get_team_combination_key(self, blue_hint: BAMLModel, blue_guess: BAMLModel, 
                                red_hint: BAMLModel, red_guess: BAMLModel) -> str:
        """Generate a unique key for a team combination."""
        return f"B{blue_hint.value}_{blue_guess.value}_R{red_hint.value}_{red_guess.value}"
    
    def _check_api_keys(self) -> bool:
        """Check if all required API keys are available."""
        required_keys = {
            BAMLModel.GPT5: ["OPENAI_API_KEY"],
            BAMLModel.GEMINI_25_PRO: ["GOOGLE_API_KEY"],
            BAMLModel.CLAUDE_HAIKU_45: ["ANTHROPIC_API_KEY"],
            BAMLModel.DEEPSEEK_REASONER: ["DEEPSEEK_API_KEY"],
            BAMLModel.GROK4: ["XAI_API_KEY"],
        }
        
        missing_requirements = []
        for model, keys in required_keys.items():
            if model not in BENCHMARK_MODELS:
                continue
            if not any(os.getenv(key) for key in keys):
                missing_requirements.append(f"{model.value} (one of {', '.join(keys)})")
        
        if missing_requirements:
            self._log("ERROR: Missing API keys for:")
            for req in missing_requirements:
                self._log(f"  - {req}")
            self._log("Please set the required API keys in your .env file")
            return False
        
        return True
    
    def _run_single_game(self, blue_hint_giver: BAMLModel, blue_guesser: BAMLModel,
                        red_hint_giver: BAMLModel, red_guesser: BAMLModel,
                        game_number: int) -> GameResult:
        """Run a single game with specified model combination."""
        
        # Generate new board
        words = generate_word_list(self.config.game.BOARD_SIZE)
        board = Board(words, config=self.config.game)
        
        # Create agents
        blue_hint_agent = BAMLHintGiver(Team.BLUE, blue_hint_giver)
        blue_guess_agent = BAMLGuesser(Team.BLUE, blue_guesser)
        red_hint_agent = BAMLHintGiver(Team.RED, red_hint_giver)
        red_guess_agent = BAMLGuesser(Team.RED, red_guesser)
        
        # Create game runner
        runner = GameRunner(
            board=board,
            blue_hint_giver=blue_hint_agent,
            blue_guesser=blue_guess_agent,
            red_hint_giver=red_hint_agent,
            red_guesser=red_guess_agent,
            max_turns=self.config.game.MAX_TURNS,
            verbose=False,  # Individual games are quiet
            game_id=f"quick_game_{game_number}"
        )
        
        # Run the game
        return runner.run()
    
    def _update_team_combination(self, game_result: GameResult, 
                               blue_hint: BAMLModel, blue_guess: BAMLModel,
                               red_hint: BAMLModel, red_guess: BAMLModel):
        """Update team combination results."""
        key = self._get_team_combination_key(blue_hint, blue_guess, red_hint, red_guess)
        
        if key not in self.team_combinations:
            self.team_combinations[key] = {
                'blue_hint_giver': blue_hint.value,
                'blue_guesser': blue_guess.value,
                'red_hint_giver': red_hint.value,
                'red_guesser': red_guess.value,
                'games_played': 0,
                'blue_wins': 0,
                'red_wins': 0,
                'draws': 0,
                'total_turns': 0,
                'errors': 0,
                'turns_played': 0,
                'empty_turns': 0,
                'first_guess_attempts': 0,
                'first_guess_correct': 0,
                'hint_count_total': 0,
                'guesses_made': 0,
                'correct_guesses': 0,
                'wrong_guesses': 0
            }
        
        combo = self.team_combinations[key]
        combo['games_played'] += 1
        
        if game_result.outcome == GameOutcome.BLUE_WIN:
            combo['blue_wins'] += 1
        elif game_result.outcome == GameOutcome.RED_WIN:
            combo['red_wins'] += 1
        else:
            combo['draws'] += 1
        
        combo['total_turns'] += game_result.total_turns
        
        if game_result.error:
            combo['errors'] += 1

        # Per-turn aggregates for this combination
        turn_history = []
        if isinstance(game_result.snapshot, dict):
            turn_history = game_result.snapshot.get('turn_history', [])

        for turn in turn_history:
            guesses = turn.get('guesses', []) or []
            hint_count = turn.get('hint_count', 0)
            total_guesses = sum(1 for g in guesses if isinstance(g, dict))
            correct_guesses = sum(1 for g in guesses if isinstance(g, dict) and g.get('correct'))
            first_guess = guesses[0] if guesses else None

            combo['turns_played'] += 1
            combo['hint_count_total'] += hint_count
            combo['guesses_made'] += total_guesses
            combo['correct_guesses'] += correct_guesses
            combo['wrong_guesses'] += max(total_guesses - correct_guesses, 0)
            if not guesses:
                combo['empty_turns'] += 1
            if first_guess is not None:
                combo['first_guess_attempts'] += 1
                if isinstance(first_guess, dict) and first_guess.get('correct'):
                    combo['first_guess_correct'] += 1

    def _update_model_performance(self, game_result: GameResult,
                                  blue_hint: BAMLModel, blue_guess: BAMLModel,
                                  red_hint: BAMLModel, red_guess: BAMLModel):
        """Update per-model performance metrics from a completed game."""

        def _key(model: BAMLModel, role: str, team: str) -> str:
            return f"{model.value}_{role}_{team}"

        winner_team = None
        if game_result.outcome == GameOutcome.BLUE_WIN:
            winner_team = "blue"
        elif game_result.outcome == GameOutcome.RED_WIN:
            winner_team = "red"

        assignments = [
            (blue_hint, "hint_giver", "blue"),
            (blue_guess, "guesser", "blue"),
            (red_hint, "hint_giver", "red"),
            (red_guess, "guesser", "red"),
        ]

        for model, role, team in assignments:
            metrics = self.model_performance.get(_key(model, role, team))
            if not metrics:
                continue
            metrics['games_played'] += 1
            if winner_team == team:
                metrics['wins'] += 1

        # Walk turn history for granular stats
        turn_history = []
        if isinstance(game_result.snapshot, dict):
            turn_history = game_result.snapshot.get('turn_history', [])

        for turn in turn_history:
            turn_team = turn.get('team')
            guesses = turn.get('guesses', []) or []
            invalid_reason = turn.get('invalid_guess_reason')
            hint_count = turn.get('hint_count', 0)
            total_guesses = sum(1 for g in guesses if isinstance(g, dict))
            correct_guesses = sum(1 for g in guesses if isinstance(g, dict) and g.get('correct'))
            first_guess = guesses[0] if guesses else None

            if turn_team == "blue":
                hint_model, guess_model = blue_hint, blue_guess
            else:
                hint_model, guess_model = red_hint, red_guess

            hint_key = _key(hint_model, "hint_giver", turn_team)
            guess_key = _key(guess_model, "guesser", turn_team)

            hint_metrics = self.model_performance.get(hint_key)
            if hint_metrics is not None:
                hint_metrics['turns_played'] += 1
                hint_metrics['hint_count_total'] += hint_count
                hint_metrics['guesses_made'] += total_guesses
                hint_metrics['correct_guesses'] += correct_guesses
                hint_metrics['wrong_guesses'] += max(total_guesses - correct_guesses, 0)
                hint_metrics['hints_given'] += 1
                if any(g.get('correct') for g in guesses if isinstance(g, dict)):
                    hint_metrics['successful_hints'] += 1

            guess_metrics = self.model_performance.get(guess_key)
            if guess_metrics is not None:
                guess_metrics['turns_played'] += 1
                guess_metrics['hint_count_total'] += hint_count
                guess_metrics['guesses_made'] += total_guesses
                guess_metrics['correct_guesses'] += correct_guesses
                guess_metrics['wrong_guesses'] += max(total_guesses - correct_guesses, 0)
                if not guesses:
                    guess_metrics['empty_turns'] += 1
                if first_guess is not None:
                    guess_metrics['first_guess_attempts'] += 1
                    if isinstance(first_guess, dict) and first_guess.get('correct'):
                        guess_metrics['first_guess_correct'] += 1
                if invalid_reason:
                    if invalid_reason == "not_on_board":
                        guess_metrics['invalid_offboard'] += 1
                    elif invalid_reason == "already_revealed":
                        guess_metrics['invalid_revealed'] += 1
                    else:
                        guess_metrics['invalid_other'] += 1
                for g in guesses:
                    if not isinstance(g, dict):
                        continue
                    if g.get('hit_bomb'):
                        guess_metrics['bomb_hits'] += 1
                    # Per-turn aggregates already accounted for above
    
    def run(self) -> QuickBenchmarkResult:
        """Run the quick benchmark."""
        
        self._log("=" * 80)
        self._log("QUICK CODENAMES BENCHMARK")
        self._log("=" * 80)
        self._log(f"Testing {len(BENCHMARK_MODELS)} models in focused combinations")
        self._log(f"Models: {[get_model_display_name(m) for m in BENCHMARK_MODELS]}")
        self._log(f"Games per combination: {self.games_per_combination}")
        
        # Check API keys
        if not self._check_api_keys():
            raise RuntimeError("Missing required API keys")
        
        # Define test combinations (focused set)
        test_combinations = []
        
        # 1. Each model as hint giver vs each model as guesser (25 combinations)
        # Each model can only play one role per game
        for blue_hint in BENCHMARK_MODELS:
            for red_guess in BENCHMARK_MODELS:
                # Use different models for blue guesser and red hint giver to avoid conflicts
                for blue_guess in BENCHMARK_MODELS:
                    if blue_guess != blue_hint:  # Blue hint giver and guesser must be different
                        for red_hint in BENCHMARK_MODELS:
                            if (red_hint != red_guess and  # Red hint giver and guesser must be different
                                red_hint != blue_hint and  # Red hint giver can't be blue hint giver
                                red_hint != blue_guess and  # Red hint giver can't be blue guesser
                                red_guess != blue_hint and  # Red guesser can't be blue hint giver
                                red_guess != blue_guess):   # Red guesser can't be blue guesser
                                test_combinations.append((blue_hint, blue_guess, red_hint, red_guess))
        
        # 2. Key strategic combinations
        # Best models vs each other - ensure no model appears twice
        top_models = [BAMLModel.GPT5, BAMLModel.GEMINI_25_PRO, BAMLModel.CLAUDE_HAIKU_45]
        for blue_hint in top_models:
            for blue_guess in top_models:
                if blue_guess != blue_hint:  # Different models for blue team
                    for red_hint in top_models:
                        if (red_hint != blue_hint and red_hint != blue_guess):  # Red hint different from blue team
                            for red_guess in top_models:
                                if (red_guess != blue_hint and red_guess != blue_guess and 
                                    red_guess != red_hint):  # Red guesser different from all others
                                    if (blue_hint, blue_guess, red_hint, red_guess) not in test_combinations:
                                        test_combinations.append((blue_hint, blue_guess, red_hint, red_guess))
        
        total_combinations = len(test_combinations)
        total_games = total_combinations * self.games_per_combination
        
        self._log(f"Total combinations: {total_combinations}")
        self._log(f"Total games: {total_games}")
        self._log("=" * 80)
        
        start_time = time.time()
        game_count = 0
        
        # Run all combinations
        for i, (blue_hint, blue_guess, red_hint, red_guess) in enumerate(test_combinations, 1):
            
            combination_name = f"{get_model_display_name(blue_hint)}+{get_model_display_name(blue_guess)} vs {get_model_display_name(red_hint)}+{get_model_display_name(red_guess)}"
            
            self._log(f"\n[{i}/{total_combinations}] Testing: {combination_name}")
            self._log("-" * 60)
            
            # Run games for this combination
            for game_num in range(1, self.games_per_combination + 1):
                game_count += 1
                
                self._log(f"  Game {game_num}/{self.games_per_combination} ({game_count}/{total_games})")
                
                try:
                    game_start = time.time()
                    result = self._run_single_game(
                        blue_hint, blue_guess, red_hint, red_guess, game_count
                    )
                    game_time = time.time() - game_start
                    
                    # Store result
                    self.game_results.append(result)
                    
                    # Update team combination stats
                    self._update_team_combination(
                        result, blue_hint, blue_guess, red_hint, red_guess
                    )
                    # Update per-model stats
                    self._update_model_performance(
                        result, blue_hint, blue_guess, red_hint, red_guess
                    )
                    
                    # Log result
                    outcome = result.outcome.value
                    winner = result.winner.value if result.winner else "Draw"
                    self._log(f"    → {outcome} (Winner: {winner}, Turns: {result.total_turns}, Time: {game_time:.1f}s)")
                    
                    if result.error:
                        self._log(f"    → Error: {result.error}")
                    
                except Exception as e:
                    self._log(f"    → Exception: {str(e)}")
                    continue
        
        total_time = time.time() - start_time
        
        # Create result
        benchmark_id = f"quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = QuickBenchmarkResult(
            benchmark_id=benchmark_id,
            timestamp=datetime.now(),
            total_combinations=total_combinations,
            total_games=len(self.game_results),
            model_performance=self.model_performance,
            team_combinations=self.team_combinations,
            game_results=self.game_results,
            config=self.config.to_dict()
        )
        
        # Print summary
        self._print_summary(result, total_time)
        
        return result
    
    def _print_summary(self, result: QuickBenchmarkResult, total_time: float):
        """Print summary of results."""
        
        self._log("\n" + "=" * 80)
        self._log("QUICK BENCHMARK SUMMARY")
        self._log("=" * 80)
        
        self._log(f"Total combinations tested: {result.total_combinations}")
        self._log(f"Total games played: {result.total_games}")
        self._log(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        if result.total_games > 0:
            self._log(f"Average time per game: {total_time/result.total_games:.1f} seconds")
        
        # Find best performing team combinations
        self._log("\nTOP 10 TEAM COMBINATIONS (by Blue Win Rate):")
        self._log("-" * 60)
        
        sorted_combos = sorted(
            result.team_combinations.items(),
            key=lambda x: x[1]['blue_wins'] / max(x[1]['games_played'], 1),
            reverse=True
        )[:10]
        
        for i, (key, combo) in enumerate(sorted_combos, 1):
            blue_team = f"{get_model_display_name(BAMLModel(combo['blue_hint_giver']))}+{get_model_display_name(BAMLModel(combo['blue_guesser']))}"
            red_team = f"{get_model_display_name(BAMLModel(combo['red_hint_giver']))}+{get_model_display_name(BAMLModel(combo['red_guesser']))}"
            blue_win_rate = combo['blue_wins'] / max(combo['games_played'], 1)
            avg_turns = combo['total_turns'] / max(combo['games_played'], 1)
            
            self._log(f"{i:2d}. {blue_team} vs {red_team}")
            self._log(f"    Blue Win Rate: {blue_win_rate:.1%} ({combo['blue_wins']}/{combo['games_played']})")
            self._log(f"    Avg Turns: {avg_turns:.1f}")
        
        self._log("\n" + "=" * 80)

def main():
    """Run the quick benchmark."""
    
    print("Starting Quick Codenames Benchmark...")
    print("This will test focused model combinations for quick insights.")
    print("This should take 30-60 minutes to complete.")
    print()
    
    # Ask for confirmation
    response = input("Continue? (y/N): ").strip().lower()
    if response != 'y':
        print("Benchmark cancelled.")
        return
    
    try:
        # Create and run benchmark
        runner = QuickBenchmarkRunner(
            games_per_combination=GAMES_PER_COMBINATION,
            verbose=VERBOSE
        )
        
        result = runner.run()
        
        # Save results
        main_file = result.save(OUTPUT_DIR)
        print(f"\nResults saved to: {main_file}")
        
        print("\nQuick benchmark completed successfully!")
        print("\nTo run the full comprehensive benchmark, use:")
        print("  python comprehensive_benchmark.py")
        print("\nTo analyze these results, use:")
        print(f"  python analyze_benchmark_results.py {main_file}")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
