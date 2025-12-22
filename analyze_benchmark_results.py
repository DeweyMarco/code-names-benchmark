"""
Benchmark Results Analysis Tool

This script analyzes the results from the comprehensive benchmark to provide
detailed insights about model performance, team dominance, and strategic insights.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from model_config import get_model_display_name, BAMLModel

# Model display names - should match the actual models used in benchmarks
# This maps internal model identifiers to display names
MODEL_DISPLAY_NAMES = {}

class BenchmarkAnalyzer:
    """Analyze comprehensive benchmark results."""
    
    def __init__(self, results_file: str):
        """Load benchmark results from JSON file."""
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        
        self.games = self.data.get('games', [])
        self.team_combinations = self.data.get('team_combinations', {})
        self.model_performance = self.data.get('model_performance', {})
        
        print(f"Loaded benchmark results:")
        print(f"  Total games: {len(self.games)}")
        print(f"  Team combinations: {len(self.team_combinations)}")
        print(f"  Benchmark ID: {self.data.get('benchmark_id', 'Unknown')}")
    
    def analyze_model_performance(self) -> pd.DataFrame:
        """Analyze how each model performs in different roles."""
        
        # Track model performance by role
        model_stats = defaultdict(lambda: {
            'hint_giver_games': 0,
            'hint_giver_wins': 0,
            'guesser_games': 0,
            'guesser_wins': 0,
            'total_games': 0,
            'total_wins': 0,
            'hints_given': 0,
            'successful_hints': 0,
            'guesses_made': 0,
            'correct_guesses': 0,
            'wrong_guesses': 0,
            'bomb_hits': 0
        })
        
        # Analyze each game
        for game in self.games:
            if 'snapshot' not in game:
                continue

            snapshot = game['snapshot']
            turn_history = snapshot.get('turn_history', [])

            # Determine which team won (winner value is lowercase from Team enum)
            blue_won = game.get('winner') == 'blue'
            red_won = game.get('winner') == 'red'

            # Track game participation for each role (to avoid double counting)
            roles_in_game = set()

            # Analyze each turn
            for turn in turn_history:
                team = turn.get('team', '').lower()
                hint_word = turn.get('hint_word', '')
                hint_count = turn.get('hint_count', 0)
                guesses = turn.get('guesses', [])
                
                # Count hints and their success
                if hint_word and hint_count > 0:
                    model_key = f"{team}_hint_giver"  # This is simplified
                    roles_in_game.add((model_key, team))
                    model_stats[model_key]['hints_given'] += 1

                    # Check if hint was successful (led to at least one correct guess)
                    correct_guesses = sum(1 for g in guesses if g.get('correct', False))
                    if correct_guesses > 0:
                        model_stats[model_key]['successful_hints'] += 1

                # Count guesses
                for guess in guesses:
                    model_key = f"{team}_guesser"  # This is simplified
                    roles_in_game.add((model_key, team))
                    model_stats[model_key]['guesses_made'] += 1

                    if guess.get('correct', False):
                        model_stats[model_key]['correct_guesses'] += 1
                    else:
                        model_stats[model_key]['wrong_guesses'] += 1

                        if guess.get('color') == 'bomb':
                            model_stats[model_key]['bomb_hits'] += 1

            # Update game counts for each role that participated in this game
            for model_key, team in roles_in_game:
                model_stats[model_key]['total_games'] += 1
                if (team == 'blue' and blue_won) or (team == 'red' and red_won):
                    model_stats[model_key]['total_wins'] += 1
        
        # Create analysis DataFrame
        analysis_data = []
        for model_role, stats in model_stats.items():
            if stats['total_games'] > 0:
                analysis_data.append({
                    'model_role': model_role,
                    'total_games': stats['total_games'],
                    'win_rate': stats['total_wins'] / stats['total_games'] if stats['total_games'] > 0 else 0,
                    'hint_success_rate': stats['successful_hints'] / stats['hints_given'] if stats['hints_given'] > 0 else 0,
                    'guess_accuracy': stats['correct_guesses'] / stats['guesses_made'] if stats['guesses_made'] > 0 else 0,
                    'bomb_hit_rate': stats['bomb_hits'] / stats['guesses_made'] if stats['guesses_made'] > 0 else 0
                })
        
        return pd.DataFrame(analysis_data)
    
    def analyze_team_combinations(self) -> pd.DataFrame:
        """Analyze team combination performance."""
        
        combo_data = []
        for combo_key, combo_stats in self.team_combinations.items():
            combo_data.append({
                'combination': combo_key,
                'blue_hint_giver': MODEL_DISPLAY_NAMES.get(combo_stats['blue_hint_giver'], combo_stats['blue_hint_giver']),
                'blue_guesser': MODEL_DISPLAY_NAMES.get(combo_stats['blue_guesser'], combo_stats['blue_guesser']),
                'red_hint_giver': MODEL_DISPLAY_NAMES.get(combo_stats['red_hint_giver'], combo_stats['red_hint_giver']),
                'red_guesser': MODEL_DISPLAY_NAMES.get(combo_stats['red_guesser'], combo_stats['red_guesser']),
                'games_played': combo_stats['games_played'],
                'blue_wins': combo_stats['blue_wins'],
                'red_wins': combo_stats['red_wins'],
                'draws': combo_stats['draws'],
                'blue_win_rate': combo_stats['blue_wins'] / combo_stats['games_played'] if combo_stats['games_played'] > 0 else 0,
                'red_win_rate': combo_stats['red_wins'] / combo_stats['games_played'] if combo_stats['games_played'] > 0 else 0,
                'avg_turns': combo_stats['total_turns'] / combo_stats['games_played'] if combo_stats['games_played'] > 0 else 0,
                'errors': combo_stats['errors']
            })
        
        return pd.DataFrame(combo_data)
    
    def find_best_hint_givers(self, top_n: int = 10) -> pd.DataFrame:
        """Find the best hint givers based on team combination performance."""

        combo_df = self.analyze_team_combinations()

        # Get unique models from the data
        all_hint_givers = set(combo_df['blue_hint_giver'].unique()) | set(combo_df['red_hint_giver'].unique())

        # Group by hint giver and calculate average performance
        hint_giver_performance = []

        for model in all_hint_givers:
            # Blue hint giver performance
            blue_performance = combo_df[combo_df['blue_hint_giver'] == model]
            if len(blue_performance) > 0:
                hint_giver_performance.append({
                    'model': model,
                    'role': 'Blue Hint Giver',
                    'avg_win_rate': blue_performance['blue_win_rate'].mean(),
                    'games_played': blue_performance['games_played'].sum(),
                    'total_wins': blue_performance['blue_wins'].sum()
                })

            # Red hint giver performance
            red_performance = combo_df[combo_df['red_hint_giver'] == model]
            if len(red_performance) > 0:
                hint_giver_performance.append({
                    'model': model,
                    'role': 'Red Hint Giver',
                    'avg_win_rate': red_performance['red_win_rate'].mean(),
                    'games_played': red_performance['games_played'].sum(),
                    'total_wins': red_performance['red_wins'].sum()
                })

        hint_df = pd.DataFrame(hint_giver_performance)
        if hint_df.empty:
            return pd.DataFrame(columns=['model', 'role', 'avg_win_rate', 'games_played', 'total_wins'])
        return hint_df.nlargest(top_n, 'avg_win_rate')
    
    def find_best_guessers(self, top_n: int = 10) -> pd.DataFrame:
        """Find the best guessers based on team combination performance."""

        combo_df = self.analyze_team_combinations()

        # Get unique models from the data
        all_guessers = set(combo_df['blue_guesser'].unique()) | set(combo_df['red_guesser'].unique())

        # Group by guesser and calculate average performance
        guesser_performance = []

        for model in all_guessers:
            # Blue guesser performance
            blue_performance = combo_df[combo_df['blue_guesser'] == model]
            if len(blue_performance) > 0:
                guesser_performance.append({
                    'model': model,
                    'role': 'Blue Guesser',
                    'avg_win_rate': blue_performance['blue_win_rate'].mean(),
                    'games_played': blue_performance['games_played'].sum(),
                    'total_wins': blue_performance['blue_wins'].sum()
                })

            # Red guesser performance
            red_performance = combo_df[combo_df['red_guesser'] == model]
            if len(red_performance) > 0:
                guesser_performance.append({
                    'model': model,
                    'role': 'Red Guesser',
                    'avg_win_rate': red_performance['red_win_rate'].mean(),
                    'games_played': red_performance['games_played'].sum(),
                    'total_wins': red_performance['red_wins'].sum()
                })

        guesser_df = pd.DataFrame(guesser_performance)
        if guesser_df.empty:
            return pd.DataFrame(columns=['model', 'role', 'avg_win_rate', 'games_played', 'total_wins'])
        return guesser_df.nlargest(top_n, 'avg_win_rate')

    def find_best_hint_giver_overall(self, top_n: int = 10) -> pd.DataFrame:
        """Find the best hint givers aggregated across both teams."""

        combo_df = self.analyze_team_combinations()

        # Get unique models from the data
        all_hint_givers = set(combo_df['blue_hint_giver'].unique()) | set(combo_df['red_hint_giver'].unique())

        # Aggregate performance across both teams
        hint_giver_performance = []

        for model in all_hint_givers:
            total_games = 0
            total_wins = 0

            # Blue hint giver performance
            blue_performance = combo_df[combo_df['blue_hint_giver'] == model]
            if len(blue_performance) > 0:
                total_games += blue_performance['games_played'].sum()
                total_wins += blue_performance['blue_wins'].sum()

            # Red hint giver performance
            red_performance = combo_df[combo_df['red_hint_giver'] == model]
            if len(red_performance) > 0:
                total_games += red_performance['games_played'].sum()
                total_wins += red_performance['red_wins'].sum()

            if total_games > 0:
                hint_giver_performance.append({
                    'model': model,
                    'win_rate': total_wins / total_games,
                    'games_played': total_games,
                    'total_wins': total_wins
                })

        hint_df = pd.DataFrame(hint_giver_performance)
        if hint_df.empty:
            return pd.DataFrame(columns=['model', 'win_rate', 'games_played', 'total_wins'])
        return hint_df.nlargest(top_n, 'win_rate')

    def find_best_guesser_overall(self, top_n: int = 10) -> pd.DataFrame:
        """Find the best guessers aggregated across both teams."""

        combo_df = self.analyze_team_combinations()

        # Get unique models from the data
        all_guessers = set(combo_df['blue_guesser'].unique()) | set(combo_df['red_guesser'].unique())

        # Aggregate performance across both teams
        guesser_performance = []

        for model in all_guessers:
            total_games = 0
            total_wins = 0

            # Blue guesser performance
            blue_performance = combo_df[combo_df['blue_guesser'] == model]
            if len(blue_performance) > 0:
                total_games += blue_performance['games_played'].sum()
                total_wins += blue_performance['blue_wins'].sum()

            # Red guesser performance
            red_performance = combo_df[combo_df['red_guesser'] == model]
            if len(red_performance) > 0:
                total_games += red_performance['games_played'].sum()
                total_wins += red_performance['red_wins'].sum()

            if total_games > 0:
                guesser_performance.append({
                    'model': model,
                    'win_rate': total_wins / total_games,
                    'games_played': total_games,
                    'total_wins': total_wins
                })

        guesser_df = pd.DataFrame(guesser_performance)
        if guesser_df.empty:
            return pd.DataFrame(columns=['model', 'win_rate', 'games_played', 'total_wins'])
        return guesser_df.nlargest(top_n, 'win_rate')

    def find_dominant_combinations(self, top_n: int = 20) -> pd.DataFrame:
        """Find the most dominant team combinations."""
        
        combo_df = self.analyze_team_combinations()

        # Filter for combinations with sufficient games (minimum 2 for statistical relevance)
        combo_df = combo_df[combo_df['games_played'] >= 2]

        if combo_df.empty:
            return pd.DataFrame(columns=['blue_hint_giver', 'blue_guesser', 'red_hint_giver', 'red_guesser',
                                        'blue_win_rate', 'games_played', 'avg_turns'])

        # Sort by blue win rate
        dominant_combos = combo_df.nlargest(top_n, 'blue_win_rate')

        return dominant_combos[['blue_hint_giver', 'blue_guesser', 'red_hint_giver', 'red_guesser',
                               'blue_win_rate', 'games_played', 'avg_turns']]
    
    def analyze_model_synergies(self) -> pd.DataFrame:
        """Analyze which model combinations work best together."""
        
        combo_df = self.analyze_team_combinations()
        
        # Analyze blue team synergies
        blue_synergies = []
        for _, row in combo_df.iterrows():
            blue_synergies.append({
                'hint_giver': row['blue_hint_giver'],
                'guesser': row['blue_guesser'],
                'combination': f"{row['blue_hint_giver']} + {row['blue_guesser']}",
                'win_rate': row['blue_win_rate'],
                'games_played': row['games_played']
            })
        
        synergy_df = pd.DataFrame(blue_synergies)
        
        # Group by combination and calculate average performance
        synergy_performance = synergy_df.groupby('combination').agg({
            'win_rate': 'mean',
            'games_played': 'sum'
        }).reset_index()

        if synergy_performance.empty:
            return pd.DataFrame(columns=['combination', 'win_rate', 'games_played'])
        return synergy_performance.nlargest(10, 'win_rate')
    
    def generate_insights_report(self) -> str:
        """Generate a comprehensive insights report."""
        
        insights = []
        
        # Basic statistics
        total_games = len(self.games)
        total_combinations = len(self.team_combinations)
        
        insights.append("# Comprehensive Codenames Benchmark Insights")
        insights.append(f"Generated from {total_games} games across {total_combinations} team combinations")
        insights.append("")

        # Overall best hint givers (regardless of team)
        insights.append("## Overall Best Hint Givers")
        insights.append("*Aggregated across both Blue and Red teams*")
        insights.append("")
        best_hint_overall = self.find_best_hint_giver_overall(5)
        for i, (_, row) in enumerate(best_hint_overall.iterrows(), 1):
            insights.append(f"{i}. **{row['model']}**: {row['win_rate']:.1%} win rate ({row['total_wins']}/{row['games_played']} games)")
        insights.append("")

        # Overall best guessers (regardless of team)
        insights.append("## Overall Best Guessers")
        insights.append("*Aggregated across both Blue and Red teams*")
        insights.append("")
        best_guesser_overall = self.find_best_guesser_overall(5)
        for i, (_, row) in enumerate(best_guesser_overall.iterrows(), 1):
            insights.append(f"{i}. **{row['model']}**: {row['win_rate']:.1%} win rate ({row['total_wins']}/{row['games_played']} games)")
        insights.append("")

        # Best hint givers by team
        insights.append("## Best Hint Givers (by Team)")
        insights.append("")
        best_hint_givers = self.find_best_hint_givers(5)
        for _, row in best_hint_givers.iterrows():
            insights.append(f"- **{row['model']}** ({row['role']}): {row['avg_win_rate']:.1%} win rate ({row['total_wins']}/{row['games_played']} games)")
        insights.append("")
        
        # Best guessers by team
        insights.append("## Best Guessers (by Team)")
        insights.append("")
        best_guessers = self.find_best_guessers(5)
        for _, row in best_guessers.iterrows():
            insights.append(f"- **{row['model']}** ({row['role']}): {row['avg_win_rate']:.1%} win rate ({row['total_wins']}/{row['games_played']} games)")
        insights.append("")
        
        # Dominant combinations
        insights.append("## Most Dominant Team Combinations")
        insights.append("")
        dominant_combos = self.find_dominant_combinations(10)
        for i, (_, row) in enumerate(dominant_combos.iterrows(), 1):
            insights.append(f"{i:2d}. **{row['blue_hint_giver']} + {row['blue_guesser']}** vs **{row['red_hint_giver']} + {row['red_guesser']}**")
            insights.append(f"    Blue Win Rate: {row['blue_win_rate']:.1%} ({row['games_played']} games, {row['avg_turns']:.1f} avg turns)")
        insights.append("")
        
        # Model synergies
        insights.append("## Best Model Synergies")
        insights.append("")
        synergies = self.analyze_model_synergies()
        for _, row in synergies.iterrows():
            insights.append(f"- **{row['combination']}**: {row['win_rate']:.1%} win rate ({row['games_played']} games)")
        insights.append("")
        
        # Strategic insights
        insights.append("## Strategic Insights")
        insights.append("")
        
        # Find models that appear most frequently in winning combinations
        combo_df = self.analyze_team_combinations()
        winning_combos = combo_df[combo_df['blue_win_rate'] > 0.6]  # 60%+ win rate
        
        if len(winning_combos) > 0:
            # Get all unique models from the data
            all_models = set(combo_df['blue_hint_giver'].unique()) | set(combo_df['blue_guesser'].unique()) | \
                        set(combo_df['red_hint_giver'].unique()) | set(combo_df['red_guesser'].unique())

            # Count model appearances in winning combinations
            model_counts = {}
            for model in all_models:
                blue_hint_count = len(winning_combos[winning_combos['blue_hint_giver'] == model])
                blue_guess_count = len(winning_combos[winning_combos['blue_guesser'] == model])
                red_hint_count = len(winning_combos[winning_combos['red_hint_giver'] == model])
                red_guess_count = len(winning_combos[winning_combos['red_guesser'] == model])

                total_appearances = blue_hint_count + blue_guess_count + red_hint_count + red_guess_count
                if total_appearances > 0:
                    model_counts[model] = total_appearances
            
            if model_counts:
                insights.append("### Models Most Frequently in Winning Combinations:")
                sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
                for model, count in sorted_models[:5]:
                    insights.append(f"- **{model}**: {count} appearances in winning combinations")
                insights.append("")
        
        return "\n".join(insights)
    
    def create_visualizations(self, output_dir: str = "analysis_plots"):
        """Create visualization plots for the analysis."""
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Team combination win rates
        combo_df = self.analyze_team_combinations()
        top_combos = combo_df.nlargest(15, 'blue_win_rate')
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_combos)), top_combos['blue_win_rate'])
        plt.yticks(range(len(top_combos)), 
                  [f"{row['blue_hint_giver']}+{row['blue_guesser']} vs {row['red_hint_giver']}+{row['red_guesser']}" 
                   for _, row in top_combos.iterrows()])
        plt.xlabel('Blue Team Win Rate')
        plt.title('Top 15 Team Combinations by Blue Win Rate')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/team_combination_win_rates.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Model performance by role
        hint_givers = self.find_best_hint_givers(10)
        guessers = self.find_best_guessers(10)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Hint givers
        ax1.barh(range(len(hint_givers)), hint_givers['avg_win_rate'])
        ax1.set_yticks(range(len(hint_givers)))
        ax1.set_yticklabels([f"{row['model']} ({row['role']})" for _, row in hint_givers.iterrows()])
        ax1.set_xlabel('Average Win Rate')
        ax1.set_title('Best Hint Givers')
        
        # Guessers
        ax2.barh(range(len(guessers)), guessers['avg_win_rate'])
        ax2.set_yticks(range(len(guessers)))
        ax2.set_yticklabels([f"{row['model']} ({row['role']})" for _, row in guessers.iterrows()])
        ax2.set_xlabel('Average Win Rate')
        ax2.set_title('Best Guessers')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_performance_by_role.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/")

def main():
    """Main analysis function."""
    
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_benchmark_results.py <results_file.json>")
        print("Example: python analyze_benchmark_results.py benchmark_results/comprehensive_20241201_143022_comprehensive.json")
        return
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"Results file not found: {results_file}")
        return
    
    try:
        # Load and analyze results
        analyzer = BenchmarkAnalyzer(results_file)
        
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        
        # Generate insights report
        insights = analyzer.generate_insights_report()
        print(insights)
        
        # Save insights to file
        insights_file = Path(results_file).parent / f"{Path(results_file).stem}_insights.md"
        with open(insights_file, 'w') as f:
            f.write(insights)
        print(f"\nInsights report saved to: {insights_file}")
        
        # Create visualizations
        output_dir = Path(results_file).parent / "analysis_plots"
        analyzer.create_visualizations(str(output_dir))
        
        print(f"\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
