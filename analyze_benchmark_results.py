"""
Benchmark Results Analysis Tool

This script analyzes the results from the comprehensive benchmark to provide
detailed insights about model performance, team dominance, and strategic insights.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from scipy import stats
import warnings

from model_config import get_model_display_name, BAMLModel

# Model display names - should match the actual models used in benchmarks
# Maps internal model identifiers (e.g., "GPT4oMini", "OpenRouterDevstral") to display names
MODEL_DISPLAY_NAMES = {
    # OpenAI GPT-4o Series
    "GPT4o": "GPT-4o",
    "GPT4oMini": "GPT-4o Mini",
    "GPT4o_20240806": "GPT-4o (2024-08-06)",
    "GPT4oMini_20240718": "GPT-4o Mini (2024-07-18)",
    # OpenAI GPT-4 Turbo Series
    "GPT4Turbo": "GPT-4 Turbo",
    "GPT4TurboPreview": "GPT-4 Turbo Preview",
    "GPT4_0125Preview": "GPT-4 (0125-preview)",
    "GPT4_1106Preview": "GPT-4 (1106-preview)",
    # OpenAI GPT-4 Base Series
    "GPT4": "GPT-4",
    "GPT4_32k": "GPT-4 32K",
    "GPT4_0613": "GPT-4 (0613)",
    # OpenAI GPT-3.5 Series
    "GPT35Turbo": "GPT-3.5 Turbo",
    "GPT35Turbo16k": "GPT-3.5 Turbo 16K",
    "GPT35TurboInstruct": "GPT-3.5 Turbo Instruct",
    # OpenAI Reasoning Models
    "O1": "o1",
    "O1Mini": "o1-mini",
    "O1Preview": "o1-preview",
    "O3": "o3",
    "O3Mini": "o3-mini",
    "O4Mini": "o4-mini",
    # OpenAI GPT-5 Series (unverified)
    "GPT5": "GPT-5",
    "GPT5Mini": "GPT-5 Mini",
    "GPT5Nano": "GPT-5 Nano",
    "GPT5Chat": "GPT-5 Chat",
    "GPT5Pro": "GPT-5 Pro",
    # OpenAI GPT-4.1 Series (unverified)
    "GPT41": "GPT-4.1",
    "GPT41Mini": "GPT-4.1 Mini",
    "GPT41Nano": "GPT-4.1 Nano",
    # Anthropic Claude Series
    "ClaudeSonnet45": "Claude Sonnet 4.5",
    "ClaudeHaiku45": "Claude Haiku 4.5",
    "ClaudeOpus41": "Claude Opus 4.1",
    "ClaudeSonnet4": "Claude Sonnet 4",
    "ClaudeOpus4": "Claude Opus 4",
    "ClaudeSonnet37": "Claude Sonnet 3.7",
    "ClaudeHaiku35": "Claude Haiku 3.5",
    "ClaudeHaiku3": "Claude 3 Haiku",
    # Google Gemini Series
    "Gemini25Pro": "Gemini 2.5 Pro",
    "Gemini25Flash": "Gemini 2.5 Flash",
    "Gemini25FlashLite": "Gemini 2.5 Flash Lite",
    "Gemini20Flash": "Gemini 2.0 Flash",
    "Gemini20FlashLite": "Gemini 2.0 Flash Lite",
    # DeepSeek Series
    "DeepSeekChat": "DeepSeek Chat",
    "DeepSeekReasoner": "DeepSeek Reasoner",
    # Meta Llama
    "Llama": "Llama 3 70B",
    # xAI Grok Series
    "Grok4": "Grok 4",
    "Grok4FastReasoning": "Grok 4 Fast Reasoning",
    "Grok4FastNonReasoning": "Grok 4 Fast",
    "Grok3": "Grok 3",
    "Grok3Fast": "Grok 3 Fast",
    "Grok3Mini": "Grok 3 Mini",
    "Grok3MiniFast": "Grok 3 Mini Fast",
    # OpenRouter (Free models)
    "OpenRouterDevstral": "Devstral",
    "OpenRouterMimoV2Flash": "MIMO V2 Flash",
    "OpenRouterNemotronNano": "Nemotron Nano",
    "OpenRouterDeepSeekR1TChimera": "DeepSeek R1T Chimera",
    "OpenRouterDeepSeekR1T2Chimera": "DeepSeek R1T2 Chimera",
    "OpenRouterGLM45Air": "GLM 4.5 Air",
    "OpenRouterLlama33_70B": "Llama 3.3 70B",
    "OpenRouterOLMo3_32B": "OLMo 3.1 32B",
}

def clean_model_name(name: str) -> str:
    """Convert internal model names to human-readable display names.

    Looks up the name in MODEL_DISPLAY_NAMES first, then falls back to
    basic cleanup (removing 'OpenRouter' prefix).

    E.g., 'GPT4oMini' -> 'GPT-4o Mini'
          'OpenRouterDevstral' -> 'Devstral'
          'ClaudeHaiku35' -> 'Claude Haiku 3.5'
    """
    if not name:
        return name
    # First try exact lookup in display names
    if name in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[name]
    # Fallback: remove OpenRouter prefix
    if name.startswith('OpenRouter'):
        return name[len('OpenRouter'):]
    return name

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

            # Get actual model names from game data
            models = game.get('models', {})

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
                    # Use actual model name from game data, with role suffix
                    actual_model = models.get(f'{team}_hint_giver', f'{team}_hint_giver')
                    model_key = f"{clean_model_name(actual_model)} (hint_giver)"
                    roles_in_game.add((model_key, team))
                    model_stats[model_key]['hints_given'] += 1

                    # Check if hint was successful (led to at least one correct guess)
                    correct_guesses = sum(1 for g in guesses if g.get('correct', False))
                    if correct_guesses > 0:
                        model_stats[model_key]['successful_hints'] += 1

                # Count guesses
                for guess in guesses:
                    # Use actual model name from game data, with role suffix
                    actual_model = models.get(f'{team}_guesser', f'{team}_guesser')
                    model_key = f"{clean_model_name(actual_model)} (guesser)"
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
                'blue_hint_giver': clean_model_name(MODEL_DISPLAY_NAMES.get(combo_stats['blue_hint_giver'], combo_stats['blue_hint_giver'])),
                'blue_guesser': clean_model_name(MODEL_DISPLAY_NAMES.get(combo_stats['blue_guesser'], combo_stats['blue_guesser'])),
                'red_hint_giver': clean_model_name(MODEL_DISPLAY_NAMES.get(combo_stats['red_hint_giver'], combo_stats['red_hint_giver'])),
                'red_guesser': clean_model_name(MODEL_DISPLAY_NAMES.get(combo_stats['red_guesser'], combo_stats['red_guesser'])),
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

    # =========================================================================
    # STATISTICAL CONFIDENCE & SIGNIFICANCE
    # =========================================================================

    def calculate_confidence_intervals(self, confidence: float = 0.95) -> pd.DataFrame:
        """Calculate Wilson score confidence intervals for win rates.

        Wilson score intervals are preferred over normal approximation for:
        - Small sample sizes
        - Proportions near 0 or 1
        """
        results = []

        for key, perf in self.model_performance.items():
            n = perf.get('games_played', 0)
            wins = perf.get('wins', 0)

            if n > 0:
                p = wins / n
                z = stats.norm.ppf(1 - (1 - confidence) / 2)

                # Wilson score interval
                denominator = 1 + z**2 / n
                center = (p + z**2 / (2*n)) / denominator
                margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator

                results.append({
                    'model': clean_model_name(perf.get('model', key)),
                    'role': perf.get('role', 'unknown'),
                    'team': perf.get('team', 'unknown'),
                    'win_rate': p,
                    'ci_lower': max(0, center - margin),
                    'ci_upper': min(1, center + margin),
                    'ci_width': 2 * margin,
                    'sample_size': n,
                    'wins': wins
                })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('win_rate', ascending=False)
        return df

    def pairwise_significance_test(self, model_a: str, model_b: str, role: str = None) -> dict:
        """Chi-squared test for significant difference between two models.

        Returns dict with p-value and whether difference is significant at 0.05 level.
        """
        # Find performance data for both models
        perf_a = None
        perf_b = None

        for key, perf in self.model_performance.items():
            model_name = clean_model_name(perf.get('model', ''))
            perf_role = perf.get('role', '')

            if model_name == model_a and (role is None or perf_role == role):
                if perf_a is None:
                    perf_a = {'wins': 0, 'games': 0}
                perf_a['wins'] += perf.get('wins', 0)
                perf_a['games'] += perf.get('games_played', 0)

            if model_name == model_b and (role is None or perf_role == role):
                if perf_b is None:
                    perf_b = {'wins': 0, 'games': 0}
                perf_b['wins'] += perf.get('wins', 0)
                perf_b['games'] += perf.get('games_played', 0)

        if perf_a is None or perf_b is None or perf_a['games'] == 0 or perf_b['games'] == 0:
            return {'error': 'Insufficient data for comparison', 'p_value': None, 'significant': False}

        # Create contingency table: [[wins_a, losses_a], [wins_b, losses_b]]
        contingency = [
            [perf_a['wins'], perf_a['games'] - perf_a['wins']],
            [perf_b['wins'], perf_b['games'] - perf_b['wins']]
        ]

        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            return {
                'model_a': model_a,
                'model_b': model_b,
                'win_rate_a': perf_a['wins'] / perf_a['games'],
                'win_rate_b': perf_b['wins'] / perf_b['games'],
                'chi2': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'sample_size_a': perf_a['games'],
                'sample_size_b': perf_b['games']
            }
        except Exception as e:
            return {'error': str(e), 'p_value': None, 'significant': False}

    def run_all_pairwise_tests(self) -> pd.DataFrame:
        """Run significance tests between all model pairs."""
        models = set()
        for perf in self.model_performance.values():
            models.add(clean_model_name(perf.get('model', '')))

        models = sorted([m for m in models if m])
        results = []

        for i, model_a in enumerate(models):
            for model_b in models[i+1:]:
                result = self.pairwise_significance_test(model_a, model_b)
                if 'error' not in result:
                    results.append(result)

        return pd.DataFrame(results)

    # =========================================================================
    # HINT GIVER DEEP ANALYSIS
    # =========================================================================

    def analyze_hint_efficiency(self) -> pd.DataFrame:
        """Deep analysis of hint giver effectiveness.

        Metrics:
        - avg_hint_count: Average number promised per hint
        - guess_yield: Correct guesses per turn (ROI)
        - efficiency: Correct guesses / promised (delivery rate)
        - risk_profile: Aggressive vs conservative hinting style
        """
        results = []

        for key, perf in self.model_performance.items():
            if perf.get('role') == 'hint_giver' and perf.get('hints_given', 0) > 0:
                hints_given = perf['hints_given']
                hint_count_total = perf.get('hint_count_total', 0)
                correct_guesses = perf.get('correct_guesses', 0)
                successful_hints = perf.get('successful_hints', 0)
                wrong_guesses = perf.get('wrong_guesses', 0)

                avg_hint_count = hint_count_total / hints_given if hints_given > 0 else 0
                guess_yield = correct_guesses / hints_given if hints_given > 0 else 0
                efficiency = correct_guesses / hint_count_total if hint_count_total > 0 else 0
                hint_success_rate = successful_hints / hints_given if hints_given > 0 else 0

                # Risk profile classification
                if avg_hint_count > 2.5:
                    risk_profile = 'aggressive'
                elif avg_hint_count < 1.5:
                    risk_profile = 'conservative'
                else:
                    risk_profile = 'balanced'

                # Calculate ambiguity score (wrong guesses caused by hints)
                ambiguity_rate = wrong_guesses / hints_given if hints_given > 0 else 0

                results.append({
                    'model': clean_model_name(perf.get('model', key)),
                    'team': perf.get('team', 'unknown'),
                    'hints_given': hints_given,
                    'avg_hint_count': round(avg_hint_count, 2),
                    'guess_yield': round(guess_yield, 2),
                    'efficiency': round(efficiency, 3),
                    'hint_success_rate': round(hint_success_rate, 3),
                    'risk_profile': risk_profile,
                    'overcommit_rate': round(1 - efficiency, 3),
                    'ambiguity_rate': round(ambiguity_rate, 2),
                    'win_rate': perf.get('wins', 0) / perf.get('games_played', 1)
                })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('efficiency', ascending=False)
        return df

    # =========================================================================
    # GUESSER DEEP ANALYSIS
    # =========================================================================

    def analyze_guesser_performance(self) -> pd.DataFrame:
        """Comprehensive guesser analysis.

        Metrics:
        - first_guess_accuracy: Critical initial decision success rate
        - overall_accuracy: Total correct / total guesses
        - bomb_rate: Catastrophic failure rate
        - invalid_rate: Rule-following ability
        - risk_adjusted_accuracy: Penalizes bomb hits heavily
        """
        results = []

        for key, perf in self.model_performance.items():
            if perf.get('role') == 'guesser' and perf.get('guesses_made', 0) > 0:
                guesses_made = perf['guesses_made']
                correct_guesses = perf.get('correct_guesses', 0)
                bomb_hits = perf.get('bomb_hits', 0)
                first_attempts = perf.get('first_guess_attempts', 0)
                first_correct = perf.get('first_guess_correct', 0)
                turns_played = perf.get('turns_played', 1)
                empty_turns = perf.get('empty_turns', 0)

                # Invalid guesses
                invalid_offboard = perf.get('invalid_offboard', 0)
                invalid_revealed = perf.get('invalid_revealed', 0)
                invalid_other = perf.get('invalid_other', 0)
                total_invalid = invalid_offboard + invalid_revealed + invalid_other

                first_guess_accuracy = first_correct / first_attempts if first_attempts > 0 else 0
                overall_accuracy = correct_guesses / guesses_made
                bomb_rate = bomb_hits / guesses_made
                invalid_rate = total_invalid / guesses_made if guesses_made > 0 else 0
                guesses_per_turn = guesses_made / turns_played if turns_played > 0 else 0
                empty_turn_rate = empty_turns / turns_played if turns_played > 0 else 0

                # Risk-adjusted accuracy: heavily penalize bomb hits
                risk_adjusted = (correct_guesses - 3 * bomb_hits) / guesses_made

                results.append({
                    'model': clean_model_name(perf.get('model', key)),
                    'team': perf.get('team', 'unknown'),
                    'games_played': perf.get('games_played', 0),
                    'first_guess_accuracy': round(first_guess_accuracy, 3),
                    'overall_accuracy': round(overall_accuracy, 3),
                    'bomb_rate': round(bomb_rate, 4),
                    'bomb_hits': bomb_hits,
                    'invalid_rate': round(invalid_rate, 4),
                    'invalid_breakdown': {
                        'offboard': invalid_offboard,
                        'revealed': invalid_revealed,
                        'other': invalid_other
                    },
                    'guesses_per_turn': round(guesses_per_turn, 2),
                    'empty_turn_rate': round(empty_turn_rate, 3),
                    'risk_adjusted_accuracy': round(risk_adjusted, 3),
                    'win_rate': perf.get('wins', 0) / perf.get('games_played', 1)
                })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('risk_adjusted_accuracy', ascending=False)
        return df

    # =========================================================================
    # HEAD-TO-HEAD MATCHUP ANALYSIS
    # =========================================================================

    def create_matchup_matrix(self, by_role: str = 'hint_giver') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create model vs model head-to-head win rate matrix.

        Args:
            by_role: 'hint_giver' or 'guesser' - which role to compare

        Returns:
            Tuple of (win_rate_matrix, games_count_matrix)
        """
        models = set()
        for combo in self.team_combinations.values():
            if by_role == 'hint_giver':
                models.add(clean_model_name(combo.get('blue_hint_giver', '')))
                models.add(clean_model_name(combo.get('red_hint_giver', '')))
            else:
                models.add(clean_model_name(combo.get('blue_guesser', '')))
                models.add(clean_model_name(combo.get('red_guesser', '')))

        models = sorted([m for m in models if m])

        # Initialize matrices
        win_counts = pd.DataFrame(0, index=models, columns=models, dtype=float)
        game_counts = pd.DataFrame(0, index=models, columns=models, dtype=int)

        for combo in self.team_combinations.values():
            if by_role == 'hint_giver':
                blue_model = clean_model_name(combo.get('blue_hint_giver', ''))
                red_model = clean_model_name(combo.get('red_hint_giver', ''))
            else:
                blue_model = clean_model_name(combo.get('blue_guesser', ''))
                red_model = clean_model_name(combo.get('red_guesser', ''))

            games = combo.get('games_played', 0)
            blue_wins = combo.get('blue_wins', 0)
            red_wins = combo.get('red_wins', 0)

            if blue_model and red_model and games > 0:
                win_counts.loc[blue_model, red_model] += blue_wins
                win_counts.loc[red_model, blue_model] += red_wins
                game_counts.loc[blue_model, red_model] += games
                game_counts.loc[red_model, blue_model] += games

        # Convert to win rates
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            win_rate_matrix = win_counts / game_counts.replace(0, np.nan)

        return win_rate_matrix, game_counts

    # =========================================================================
    # ELO RATING SYSTEM
    # =========================================================================

    def calculate_elo_ratings(self, k_factor: float = 32, initial_rating: float = 1500) -> pd.DataFrame:
        """Calculate Elo ratings for all models based on game outcomes.

        Each model gets separate ratings for hint_giver and guesser roles.
        Team rating is the average of hint_giver and guesser ratings.

        Uses game-level model info if available (new format), otherwise falls back
        to aggregate data from team_combinations (for backward compatibility).
        """
        ratings = defaultdict(lambda: initial_rating)
        rating_history = defaultdict(list)

        # Check if games have model information (new format)
        games_have_models = any(game.get('models') for game in self.games)

        if games_have_models:
            # New format: process individual games with model info
            for game in self.games:
                winner = game.get('winner')
                if not winner:
                    continue

                models = game.get('models', {})
                if not models:
                    continue

                blue_hg = models.get('blue_hint_giver')
                blue_g = models.get('blue_guesser')
                red_hg = models.get('red_hint_giver')
                red_g = models.get('red_guesser')

                # Skip if any model is missing
                if not all([blue_hg, blue_g, red_hg, red_g]):
                    continue

                # Calculate team ratings (average of members)
                blue_rating = (ratings[f"{blue_hg}_hg"] + ratings[f"{blue_g}_g"]) / 2
                red_rating = (ratings[f"{red_hg}_hg"] + ratings[f"{red_g}_g"]) / 2

                # Expected scores
                expected_blue = 1 / (1 + 10**((red_rating - blue_rating) / 400))
                expected_red = 1 - expected_blue

                # Actual scores
                actual_blue = 1 if winner == 'blue' else 0
                actual_red = 1 - actual_blue

                # Update ratings
                blue_delta = k_factor * (actual_blue - expected_blue)
                red_delta = k_factor * (actual_red - expected_red)

                ratings[f"{blue_hg}_hg"] += blue_delta / 2
                ratings[f"{blue_g}_g"] += blue_delta / 2
                ratings[f"{red_hg}_hg"] += red_delta / 2
                ratings[f"{red_g}_g"] += red_delta / 2

                # Track history
                for model_role, rating in ratings.items():
                    rating_history[model_role].append(rating)
        else:
            # Fallback: use aggregate data from team_combinations
            # This is less accurate (game order not preserved) but works with old data
            for combo_key, combo in self.team_combinations.items():
                if not (combo.get('blue_hint_giver') and combo.get('blue_guesser') and
                        combo.get('red_hint_giver') and combo.get('red_guesser')):
                    continue

                blue_hg = combo['blue_hint_giver']
                blue_g = combo['blue_guesser']
                red_hg = combo['red_hint_giver']
                red_g = combo['red_guesser']

                blue_wins = combo.get('blue_wins', 0)
                red_wins = combo.get('red_wins', 0)

                # Process blue wins
                for _ in range(blue_wins):
                    blue_rating = (ratings[f"{blue_hg}_hg"] + ratings[f"{blue_g}_g"]) / 2
                    red_rating = (ratings[f"{red_hg}_hg"] + ratings[f"{red_g}_g"]) / 2
                    expected_blue = 1 / (1 + 10**((red_rating - blue_rating) / 400))
                    blue_delta = k_factor * (1 - expected_blue)
                    red_delta = k_factor * (0 - (1 - expected_blue))
                    ratings[f"{blue_hg}_hg"] += blue_delta / 2
                    ratings[f"{blue_g}_g"] += blue_delta / 2
                    ratings[f"{red_hg}_hg"] += red_delta / 2
                    ratings[f"{red_g}_g"] += red_delta / 2
                    for model_role, rating in ratings.items():
                        rating_history[model_role].append(rating)

                # Process red wins
                for _ in range(red_wins):
                    blue_rating = (ratings[f"{blue_hg}_hg"] + ratings[f"{blue_g}_g"]) / 2
                    red_rating = (ratings[f"{red_hg}_hg"] + ratings[f"{red_g}_g"]) / 2
                    expected_blue = 1 / (1 + 10**((red_rating - blue_rating) / 400))
                    blue_delta = k_factor * (0 - expected_blue)
                    red_delta = k_factor * (1 - (1 - expected_blue))
                    ratings[f"{blue_hg}_hg"] += blue_delta / 2
                    ratings[f"{blue_g}_g"] += blue_delta / 2
                    ratings[f"{red_hg}_hg"] += red_delta / 2
                    ratings[f"{red_g}_g"] += red_delta / 2
                    for model_role, rating in ratings.items():
                        rating_history[model_role].append(rating)

        # Compile results
        results = []
        model_ratings = defaultdict(lambda: {'hint_giver': None, 'guesser': None})

        for key, rating in ratings.items():
            if key.endswith('_hg'):
                model = clean_model_name(key[:-3])
                model_ratings[model]['hint_giver'] = rating
            elif key.endswith('_g'):
                model = clean_model_name(key[:-2])
                model_ratings[model]['guesser'] = rating

        for model, role_ratings in model_ratings.items():
            hg_rating = role_ratings['hint_giver'] or initial_rating
            g_rating = role_ratings['guesser'] or initial_rating

            results.append({
                'model': model,
                'elo_hint_giver': round(hg_rating),
                'elo_guesser': round(g_rating),
                'elo_combined': round((hg_rating + g_rating) / 2),
                'elo_best_role': 'hint_giver' if hg_rating > g_rating else 'guesser'
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('elo_combined', ascending=False)
        return df

    # =========================================================================
    # GAME FLOW & MOMENTUM ANALYSIS
    # =========================================================================

    def analyze_game_momentum(self) -> pd.DataFrame:
        """Analyze comeback rates, lead preservation, and game dynamics."""
        results = []

        for game in self.games:
            turns = game.get('snapshot', {}).get('turn_history', [])
            if not turns:
                continue

            scores = {'blue': 0, 'red': 0}
            lead_changes = 0
            prev_leader = None
            max_blue_lead = 0
            max_red_lead = 0
            blue_score_history = [0]
            red_score_history = [0]

            for turn in turns:
                team = turn.get('team', '').lower()
                correct = sum(1 for g in turn.get('guesses', []) if g.get('correct', False))

                if team in scores:
                    scores[team] += correct

                blue_score_history.append(scores['blue'])
                red_score_history.append(scores['red'])

                # Track leads
                lead = scores['blue'] - scores['red']
                max_blue_lead = max(max_blue_lead, lead)
                max_red_lead = max(max_red_lead, -lead)

                # Track lead changes
                if scores['blue'] > scores['red']:
                    leader = 'blue'
                elif scores['red'] > scores['blue']:
                    leader = 'red'
                else:
                    leader = None

                if leader and leader != prev_leader and prev_leader is not None:
                    lead_changes += 1
                prev_leader = leader

            winner = game.get('winner')
            was_comeback = False
            deficit_overcome = 0

            if winner == 'blue' and max_red_lead > 0:
                was_comeback = True
                deficit_overcome = max_red_lead
            elif winner == 'red' and max_blue_lead > 0:
                was_comeback = True
                deficit_overcome = max_blue_lead

            results.append({
                'game_id': game.get('game_id', ''),
                'winner': winner,
                'total_turns': game.get('total_turns', len(turns)),
                'lead_changes': lead_changes,
                'was_comeback': was_comeback,
                'deficit_overcome': deficit_overcome,
                'max_blue_lead': max_blue_lead,
                'max_red_lead': max_red_lead,
                'final_blue_score': scores['blue'],
                'final_red_score': scores['red'],
                'competitiveness': lead_changes / max(len(turns), 1)  # Higher = more back-and-forth
            })

        return pd.DataFrame(results)

    def get_momentum_summary(self) -> dict:
        """Get summary statistics of game momentum patterns."""
        momentum_df = self.analyze_game_momentum()

        if momentum_df.empty:
            return {}

        total_games = len(momentum_df)
        comebacks = momentum_df['was_comeback'].sum()

        return {
            'total_games_analyzed': total_games,
            'comeback_rate': comebacks / total_games if total_games > 0 else 0,
            'avg_lead_changes': momentum_df['lead_changes'].mean(),
            'max_lead_changes': momentum_df['lead_changes'].max(),
            'avg_deficit_overcome': momentum_df[momentum_df['was_comeback']]['deficit_overcome'].mean() if comebacks > 0 else 0,
            'avg_competitiveness': momentum_df['competitiveness'].mean(),
            'avg_game_length': momentum_df['total_turns'].mean(),
            'blue_win_rate': (momentum_df['winner'] == 'blue').mean()
        }

    # =========================================================================
    # ROLE VERSATILITY ANALYSIS
    # =========================================================================

    def calculate_role_versatility(self) -> pd.DataFrame:
        """Analyze how well each model performs in both roles.

        Versatility score: 1 = equally good at both roles, 0 = only good at one
        """
        model_roles = defaultdict(lambda: {
            'hint_giver': {'wins': 0, 'games': 0},
            'guesser': {'wins': 0, 'games': 0}
        })

        for key, perf in self.model_performance.items():
            model = clean_model_name(perf.get('model', ''))
            role = perf.get('role', '')

            if model and role in ['hint_giver', 'guesser']:
                model_roles[model][role]['wins'] += perf.get('wins', 0)
                model_roles[model][role]['games'] += perf.get('games_played', 0)

        results = []
        for model, roles in model_roles.items():
            hg_games = roles['hint_giver']['games']
            g_games = roles['guesser']['games']

            hg_rate = roles['hint_giver']['wins'] / hg_games if hg_games > 0 else 0
            g_rate = roles['guesser']['wins'] / g_games if g_games > 0 else 0

            # Versatility: 1 - normalized difference
            if hg_rate + g_rate > 0:
                versatility = 1 - abs(hg_rate - g_rate) / max(hg_rate, g_rate, 0.001)
            else:
                versatility = 0

            # Combined strength (average performance)
            total_wins = roles['hint_giver']['wins'] + roles['guesser']['wins']
            total_games = hg_games + g_games
            combined_win_rate = total_wins / total_games if total_games > 0 else 0

            results.append({
                'model': model,
                'hint_giver_win_rate': round(hg_rate, 3),
                'hint_giver_games': hg_games,
                'guesser_win_rate': round(g_rate, 3),
                'guesser_games': g_games,
                'versatility_score': round(versatility, 3),
                'best_role': 'hint_giver' if hg_rate > g_rate else 'guesser' if g_rate > hg_rate else 'equal',
                'role_gap': round(abs(hg_rate - g_rate), 3),
                'combined_win_rate': round(combined_win_rate, 3),
                'total_games': total_games
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('combined_win_rate', ascending=False)
        return df

    # =========================================================================
    # ERROR PATTERN ANALYSIS
    # =========================================================================

    def analyze_error_patterns(self) -> dict:
        """Deep dive into failure modes and error patterns."""
        error_analysis = {
            'bomb_hits_by_model': defaultdict(int),
            'bomb_contexts': [],
            'invalid_by_type': defaultdict(lambda: {'offboard': 0, 'revealed': 0, 'other': 0}),
            'wrong_guess_colors': defaultdict(lambda: {'neutral': 0, 'opponent': 0, 'bomb': 0}),
            'total_errors_by_model': defaultdict(int)
        }

        # Aggregate from model_performance
        for key, perf in self.model_performance.items():
            model = clean_model_name(perf.get('model', key))

            if perf.get('role') == 'guesser':
                error_analysis['bomb_hits_by_model'][model] += perf.get('bomb_hits', 0)
                error_analysis['invalid_by_type'][model]['offboard'] += perf.get('invalid_offboard', 0)
                error_analysis['invalid_by_type'][model]['revealed'] += perf.get('invalid_revealed', 0)
                error_analysis['invalid_by_type'][model]['other'] += perf.get('invalid_other', 0)

                total_errors = (perf.get('bomb_hits', 0) + perf.get('invalid_offboard', 0) +
                               perf.get('invalid_revealed', 0) + perf.get('invalid_other', 0))
                error_analysis['total_errors_by_model'][model] += total_errors

        # Analyze bomb hit contexts from game data
        for game in self.games:
            turns = game.get('snapshot', {}).get('turn_history', [])

            for turn in turns:
                for guess in turn.get('guesses', []):
                    if guess.get('hit_bomb'):
                        error_analysis['bomb_contexts'].append({
                            'game_id': game.get('game_id', ''),
                            'turn': turn.get('turn_number', 0),
                            'team': turn.get('team', ''),
                            'word': guess.get('word', ''),
                            'hint_word': turn.get('hint_word', ''),
                            'hint_count': turn.get('hint_count', 0)
                        })
                    elif not guess.get('correct'):
                        color = guess.get('color', 'unknown')
                        team = turn.get('team', '').lower()

                        if color == 'neutral':
                            error_analysis['wrong_guess_colors'][team]['neutral'] += 1
                        elif color == 'bomb':
                            error_analysis['wrong_guess_colors'][team]['bomb'] += 1
                        else:
                            error_analysis['wrong_guess_colors'][team]['opponent'] += 1

        return error_analysis

    def get_error_summary(self) -> pd.DataFrame:
        """Get error summary as a DataFrame."""
        errors = self.analyze_error_patterns()

        results = []
        all_models = set(errors['bomb_hits_by_model'].keys()) | set(errors['invalid_by_type'].keys())

        for model in all_models:
            bombs = errors['bomb_hits_by_model'].get(model, 0)
            invalid = errors['invalid_by_type'].get(model, {'offboard': 0, 'revealed': 0, 'other': 0})
            total = errors['total_errors_by_model'].get(model, 0)

            results.append({
                'model': model,
                'bomb_hits': bombs,
                'invalid_offboard': invalid['offboard'],
                'invalid_revealed': invalid['revealed'],
                'invalid_other': invalid['other'],
                'total_errors': total
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('total_errors', ascending=True)
        return df

    # =========================================================================
    # HINT WORD PATTERN ANALYSIS
    # =========================================================================

    def analyze_hint_patterns(self) -> dict:
        """Analyze hint word characteristics and patterns."""
        hints = []
        hint_words = []

        for game in self.games:
            for turn in game.get('snapshot', {}).get('turn_history', []):
                hint_word = turn.get('hint_word', '')
                hint_count = turn.get('hint_count', 0)
                guesses = turn.get('guesses', [])

                if hint_word and hint_count > 0:
                    correct_count = sum(1 for g in guesses if g.get('correct', False))

                    hints.append({
                        'word': hint_word.lower(),
                        'count': hint_count,
                        'team': turn.get('team', ''),
                        'turn_number': turn.get('turn_number', 0),
                        'guesses_made': len(guesses),
                        'correct_guesses': correct_count,
                        'success': correct_count > 0,
                        'perfect': correct_count >= hint_count,
                        'word_length': len(hint_word),
                        'efficiency': correct_count / hint_count if hint_count > 0 else 0
                    })
                    hint_words.append(hint_word.lower())

        if not hints:
            return {
                'total_hints': 0,
                'unique_hints': 0,
                'creativity_ratio': 0,
                'avg_hint_length': 0,
                'avg_hint_count': 0,
                'overall_success_rate': 0,
                'perfect_hint_rate': 0,
                'hint_count_distribution': {},
                'most_common_hints': [],
                'success_by_count': {}
            }

        df = pd.DataFrame(hints)
        hint_counter = Counter(hint_words)

        # Success rate by hint count
        success_by_count = df.groupby('count').agg({
            'success': 'mean',
            'perfect': 'mean',
            'efficiency': 'mean',
            'word': 'count'
        }).rename(columns={'word': 'occurrences'}).to_dict('index')

        return {
            'total_hints': len(hints),
            'unique_hints': len(set(hint_words)),
            'creativity_ratio': len(set(hint_words)) / len(hints) if hints else 0,
            'avg_hint_length': df['word_length'].mean(),
            'avg_hint_count': df['count'].mean(),
            'overall_success_rate': df['success'].mean(),
            'perfect_hint_rate': df['perfect'].mean(),
            'avg_efficiency': df['efficiency'].mean(),
            'hint_count_distribution': dict(Counter(df['count'])),
            'most_common_hints': hint_counter.most_common(15),
            'success_by_count': success_by_count
        }

    # =========================================================================
    # FIRST-MOVER (BLUE) ADVANTAGE ANALYSIS
    # =========================================================================

    def analyze_first_mover_advantage(self) -> dict:
        """Analyze Blue team's first-mover advantage (Blue always goes first)."""
        combo_df = self.analyze_team_combinations()

        if combo_df.empty:
            return {
                'overall_blue_win_rate': 0,
                'overall_red_win_rate': 0,
                'blue_advantage': 0,
                'mirror_matches': None,
                'advantage_significant': False
            }

        # Overall blue win rate
        total_blue_wins = combo_df['blue_wins'].sum()
        total_red_wins = combo_df['red_wins'].sum()
        total_games = combo_df['games_played'].sum()

        overall_blue_rate = total_blue_wins / total_games if total_games > 0 else 0
        overall_red_rate = total_red_wins / total_games if total_games > 0 else 0

        # Find mirror matches (same models on both teams)
        mirror_matches = combo_df[
            (combo_df['blue_hint_giver'] == combo_df['red_hint_giver']) &
            (combo_df['blue_guesser'] == combo_df['red_guesser'])
        ]

        mirror_blue_rate = None
        if len(mirror_matches) > 0:
            mirror_total_games = mirror_matches['games_played'].sum()
            mirror_blue_wins = mirror_matches['blue_wins'].sum()
            mirror_blue_rate = mirror_blue_wins / mirror_total_games if mirror_total_games > 0 else 0

        # Statistical test for advantage
        # Use binomial test: is blue win rate significantly > 50%?
        try:
            p_value = stats.binomtest(total_blue_wins, total_games, 0.5, alternative='greater').pvalue
            significant = p_value < 0.05
        except:
            p_value = None
            significant = overall_blue_rate > 0.55

        return {
            'overall_blue_win_rate': round(overall_blue_rate, 3),
            'overall_red_win_rate': round(overall_red_rate, 3),
            'blue_advantage': round(overall_blue_rate - 0.5, 3),
            'mirror_match_blue_rate': round(mirror_blue_rate, 3) if mirror_blue_rate is not None else None,
            'mirror_match_count': len(mirror_matches),
            'total_games': total_games,
            'p_value': round(p_value, 4) if p_value is not None else None,
            'advantage_significant': significant,
            'advantage_magnitude': 'strong' if overall_blue_rate > 0.6 else 'moderate' if overall_blue_rate > 0.55 else 'weak' if overall_blue_rate > 0.5 else 'none'
        }

    # =========================================================================
    # GAME EFFICIENCY ANALYSIS
    # =========================================================================

    def analyze_game_efficiency(self) -> pd.DataFrame:
        """Analyze which team combinations win faster."""
        results = []

        for combo_key, combo in self.team_combinations.items():
            games = combo.get('games_played', 0)
            if games == 0:
                continue

            total_turns = combo.get('total_turns', 0)
            avg_turns = total_turns / games
            blue_wins = combo.get('blue_wins', 0)
            red_wins = combo.get('red_wins', 0)

            # Efficiency: wins per turn (higher = faster wins)
            blue_efficiency = blue_wins / total_turns if total_turns > 0 else 0
            red_efficiency = red_wins / total_turns if total_turns > 0 else 0

            results.append({
                'blue_hint_giver': clean_model_name(combo.get('blue_hint_giver', '')),
                'blue_guesser': clean_model_name(combo.get('blue_guesser', '')),
                'red_hint_giver': clean_model_name(combo.get('red_hint_giver', '')),
                'red_guesser': clean_model_name(combo.get('red_guesser', '')),
                'games_played': games,
                'avg_turns': round(avg_turns, 1),
                'blue_wins': blue_wins,
                'red_wins': red_wins,
                'blue_win_rate': round(blue_wins / games, 3),
                'blue_efficiency': round(blue_efficiency, 4),
                'red_efficiency': round(red_efficiency, 4),
                'total_turns': total_turns
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('blue_efficiency', ascending=False)
        return df

    def get_efficiency_by_model(self) -> pd.DataFrame:
        """Get average game efficiency metrics grouped by model."""
        efficiency_df = self.analyze_game_efficiency()

        if efficiency_df.empty:
            return pd.DataFrame()

        # Aggregate by hint giver (names are already cleaned in analyze_game_efficiency)
        hint_giver_efficiency = []

        all_hint_givers = set(efficiency_df['blue_hint_giver'].unique()) | set(efficiency_df['red_hint_giver'].unique())

        for model in all_hint_givers:
            if not model:
                continue

            blue_rows = efficiency_df[efficiency_df['blue_hint_giver'] == model]
            red_rows = efficiency_df[efficiency_df['red_hint_giver'] == model]

            total_games = blue_rows['games_played'].sum() + red_rows['games_played'].sum()
            total_wins = blue_rows['blue_wins'].sum() + red_rows['red_wins'].sum()
            total_turns = blue_rows['total_turns'].sum() + red_rows['total_turns'].sum()

            avg_turns_to_win = total_turns / total_wins if total_wins > 0 else float('inf')

            hint_giver_efficiency.append({
                'model': model,
                'role': 'hint_giver',
                'total_games': total_games,
                'total_wins': total_wins,
                'win_rate': round(total_wins / total_games, 3) if total_games > 0 else 0,
                'avg_turns_per_game': round(total_turns / total_games, 1) if total_games > 0 else 0,
                'avg_turns_to_win': round(avg_turns_to_win, 1) if avg_turns_to_win != float('inf') else None
            })

        return pd.DataFrame(hint_giver_efficiency).sort_values('win_rate', ascending=False)

    def generate_insights_report(self) -> str:
        """Generate a comprehensive insights report with all advanced metrics."""

        insights = []

        # Basic statistics
        total_games = len(self.games)
        total_combinations = len(self.team_combinations)

        insights.append("# Comprehensive Codenames Benchmark Insights")
        insights.append(f"Generated from {total_games} games across {total_combinations} team combinations")
        insights.append("")

        # =====================================================================
        # EXECUTIVE SUMMARY
        # =====================================================================
        insights.append("## Executive Summary")
        insights.append("")

        # First mover advantage
        fma = self.analyze_first_mover_advantage()
        insights.append(f"- **Blue (First-Mover) Win Rate**: {fma['overall_blue_win_rate']:.1%} "
                       f"({'statistically significant' if fma['advantage_significant'] else 'not significant'}, "
                       f"p={fma['p_value']:.4f})" if fma['p_value'] else f"({fma['advantage_magnitude']} advantage)")

        # Momentum summary
        momentum = self.get_momentum_summary()
        if momentum:
            insights.append(f"- **Comeback Rate**: {momentum['comeback_rate']:.1%} of games had comebacks")
            insights.append(f"- **Average Game Length**: {momentum['avg_game_length']:.1f} turns")
        insights.append("")

        # =====================================================================
        # ELO RANKINGS
        # =====================================================================
        insights.append("## Elo Rankings")
        insights.append("*Skill-based rating system accounting for opponent strength*")
        insights.append("")

        elo_df = self.calculate_elo_ratings()
        if not elo_df.empty:
            for i, (_, row) in enumerate(elo_df.head(10).iterrows(), 1):
                insights.append(f"{i}. **{row['model']}**: {row['elo_combined']} combined "
                               f"(HG: {row['elo_hint_giver']}, G: {row['elo_guesser']}, best: {row['elo_best_role']})")
        insights.append("")

        # =====================================================================
        # ROLE VERSATILITY
        # =====================================================================
        insights.append("## Role Versatility Analysis")
        insights.append("*Which models perform well in both roles?*")
        insights.append("")

        versatility_df = self.calculate_role_versatility()
        if not versatility_df.empty:
            for _, row in versatility_df.head(5).iterrows():
                insights.append(f"- **{row['model']}**: {row['combined_win_rate']:.1%} combined win rate "
                               f"(HG: {row['hint_giver_win_rate']:.1%}, G: {row['guesser_win_rate']:.1%}, "
                               f"versatility: {row['versatility_score']:.2f}, best: {row['best_role']})")
        insights.append("")

        # =====================================================================
        # STATISTICAL CONFIDENCE
        # =====================================================================
        insights.append("## Statistical Confidence (95% Wilson CI)")
        insights.append("*Win rates with confidence intervals - wider CI = less certainty*")
        insights.append("")

        ci_df = self.calculate_confidence_intervals()
        if not ci_df.empty:
            # Group by model and show aggregated
            for _, row in ci_df.head(10).iterrows():
                ci_width_pct = (row['ci_upper'] - row['ci_lower']) * 100
                insights.append(f"- **{row['model']}** ({row['role']}, {row['team']}): "
                               f"{row['win_rate']:.1%} [{row['ci_lower']:.1%} - {row['ci_upper']:.1%}] "
                               f"(n={row['sample_size']}, CI width: {ci_width_pct:.1f}pp)")
        insights.append("")

        # =====================================================================
        # HINT GIVER EFFICIENCY
        # =====================================================================
        insights.append("## Hint Giver Efficiency Analysis")
        insights.append("*Deep metrics on hint quality and strategy*")
        insights.append("")

        hint_eff = self.analyze_hint_efficiency()
        if not hint_eff.empty:
            insights.append("| Model | Team | Avg Hint | Yield | Efficiency | Risk Profile | Win Rate |")
            insights.append("|-------|------|----------|-------|------------|--------------|----------|")
            for _, row in hint_eff.iterrows():
                insights.append(f"| {row['model']} | {row['team']} | {row['avg_hint_count']:.1f} | "
                               f"{row['guess_yield']:.2f} | {row['efficiency']:.1%} | {row['risk_profile']} | "
                               f"{row['win_rate']:.1%} |")
        insights.append("")

        # =====================================================================
        # GUESSER PERFORMANCE
        # =====================================================================
        insights.append("## Guesser Performance Analysis")
        insights.append("*Critical metrics for guesser evaluation*")
        insights.append("")

        guesser_perf = self.analyze_guesser_performance()
        if not guesser_perf.empty:
            insights.append("| Model | Team | 1st Guess | Overall | Bomb Rate | Risk-Adj | Win Rate |")
            insights.append("|-------|------|-----------|---------|-----------|----------|----------|")
            for _, row in guesser_perf.iterrows():
                insights.append(f"| {row['model']} | {row['team']} | {row['first_guess_accuracy']:.1%} | "
                               f"{row['overall_accuracy']:.1%} | {row['bomb_rate']:.2%} | "
                               f"{row['risk_adjusted_accuracy']:.1%} | {row['win_rate']:.1%} |")
        insights.append("")

        # =====================================================================
        # ERROR ANALYSIS
        # =====================================================================
        insights.append("## Error Analysis")
        insights.append("*Failure modes and catastrophic errors*")
        insights.append("")

        error_df = self.get_error_summary()
        error_patterns = self.analyze_error_patterns()

        if not error_df.empty:
            insights.append("### Errors by Model (Guessers)")
            for _, row in error_df.iterrows():
                if row['total_errors'] > 0:
                    insights.append(f"- **{row['model']}**: {row['total_errors']} total errors "
                                   f"(bombs: {row['bomb_hits']}, invalid: {row['invalid_offboard'] + row['invalid_revealed'] + row['invalid_other']})")

        if error_patterns['bomb_contexts']:
            insights.append("")
            insights.append("### Recent Bomb Hits (Context)")
            for ctx in error_patterns['bomb_contexts'][:5]:
                insights.append(f"- Turn {ctx['turn']}: '{ctx['word']}' guessed after hint '{ctx['hint_word']}' ({ctx['hint_count']})")
        insights.append("")

        # =====================================================================
        # HINT WORD PATTERNS
        # =====================================================================
        insights.append("## Hint Word Analysis")
        insights.append("")

        hint_patterns = self.analyze_hint_patterns()
        if hint_patterns['total_hints'] > 0:
            insights.append(f"- **Total Hints Given**: {hint_patterns['total_hints']}")
            insights.append(f"- **Unique Hints**: {hint_patterns['unique_hints']} ({hint_patterns['creativity_ratio']:.1%} creativity)")
            insights.append(f"- **Average Hint Count**: {hint_patterns['avg_hint_count']:.2f}")
            insights.append(f"- **Overall Success Rate**: {hint_patterns['overall_success_rate']:.1%}")
            insights.append(f"- **Perfect Hint Rate**: {hint_patterns['perfect_hint_rate']:.1%}")
            insights.append("")

            if hint_patterns['most_common_hints']:
                insights.append("### Most Common Hints")
                for word, count in hint_patterns['most_common_hints'][:10]:
                    insights.append(f"- '{word}': {count} times")
            insights.append("")

            if hint_patterns['success_by_count']:
                insights.append("### Success Rate by Hint Count")
                for count, stats in sorted(hint_patterns['success_by_count'].items()):
                    insights.append(f"- **{count}**: {stats['success']:.1%} success, {stats['efficiency']:.1%} efficiency ({stats['occurrences']} hints)")
        insights.append("")

        # =====================================================================
        # GAME DYNAMICS
        # =====================================================================
        insights.append("## Game Dynamics")
        insights.append("")

        if momentum:
            insights.append(f"- **Average Lead Changes**: {momentum['avg_lead_changes']:.1f} per game")
            insights.append(f"- **Max Lead Changes**: {momentum['max_lead_changes']}")
            insights.append(f"- **Average Competitiveness**: {momentum['avg_competitiveness']:.2f}")
            if momentum['comeback_rate'] > 0:
                insights.append(f"- **Average Deficit Overcome**: {momentum['avg_deficit_overcome']:.1f} cards")
        insights.append("")

        # =====================================================================
        # OVERALL RANKINGS (Original metrics)
        # =====================================================================
        insights.append("## Overall Best Hint Givers")
        insights.append("*Aggregated across both Blue and Red teams*")
        insights.append("")
        best_hint_overall = self.find_best_hint_giver_overall(5)
        for i, (_, row) in enumerate(best_hint_overall.iterrows(), 1):
            insights.append(f"{i}. **{row['model']}**: {row['win_rate']:.1%} win rate ({row['total_wins']}/{row['games_played']} games)")
        insights.append("")

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

        # =====================================================================
        # GAME EFFICIENCY
        # =====================================================================
        insights.append("## Game Efficiency (Speed of Victory)")
        insights.append("")

        efficiency_df = self.get_efficiency_by_model()
        if not efficiency_df.empty:
            for _, row in efficiency_df.head(5).iterrows():
                turns_to_win = row['avg_turns_to_win']
                turns_str = f"{turns_to_win:.1f}" if turns_to_win else "N/A"
                insights.append(f"- **{row['model']}**: {row['win_rate']:.1%} win rate, "
                               f"{row['avg_turns_per_game']:.1f} avg turns/game, {turns_str} turns/win")
        insights.append("")

        # =====================================================================
        # STRATEGIC INSIGHTS
        # =====================================================================
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

        # First mover advantage insights
        insights.append("### First-Mover (Blue) Advantage Analysis")
        insights.append(f"- Blue wins {fma['overall_blue_win_rate']:.1%} of games (expected: 50%)")
        insights.append(f"- Advantage magnitude: **{fma['advantage_magnitude']}**")
        if fma['mirror_match_blue_rate'] is not None:
            insights.append(f"- In mirror matches (same models both sides): Blue wins {fma['mirror_match_blue_rate']:.1%}")
        insights.append("")

        return "\n".join(insights)
    
    def create_visualizations(self, output_dir: str = "analysis_plots"):
        """Create comprehensive visualization plots for the analysis."""

        Path(output_dir).mkdir(exist_ok=True)

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        print("Generating visualizations...")

        # =====================================================================
        # 1. Team combination win rates (original)
        # =====================================================================
        combo_df = self.analyze_team_combinations()
        if not combo_df.empty:
            top_combos = combo_df.nlargest(15, 'blue_win_rate')

            plt.figure(figsize=(14, 10))
            colors = ['#2ecc71' if rate > 0.5 else '#e74c3c' for rate in top_combos['blue_win_rate']]
            plt.barh(range(len(top_combos)), top_combos['blue_win_rate'], color=colors)
            plt.yticks(range(len(top_combos)),
                      [f"{row['blue_hint_giver'][:12]}+{row['blue_guesser'][:12]} vs {row['red_hint_giver'][:12]}+{row['red_guesser'][:12]}"
                       for _, row in top_combos.iterrows()], fontsize=8)
            plt.xlabel('Blue Team Win Rate')
            plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='50% baseline')
            plt.title('Top 15 Team Combinations by Blue Win Rate')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/team_combination_win_rates.png", dpi=300, bbox_inches='tight')
            plt.close()

        # =====================================================================
        # 2. Model performance by role (original, enhanced)
        # =====================================================================
        hint_givers = self.find_best_hint_givers(10)
        guessers = self.find_best_guessers(10)

        if not hint_givers.empty or not guessers.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            if not hint_givers.empty:
                colors1 = sns.color_palette("Blues_r", len(hint_givers))
                ax1.barh(range(len(hint_givers)), hint_givers['avg_win_rate'], color=colors1)
                ax1.set_yticks(range(len(hint_givers)))
                ax1.set_yticklabels([f"{row['model'][:15]} ({row['role'][:10]})" for _, row in hint_givers.iterrows()], fontsize=9)
                ax1.set_xlabel('Average Win Rate')
                ax1.set_title('Best Hint Givers by Win Rate')
                ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)

            if not guessers.empty:
                colors2 = sns.color_palette("Greens_r", len(guessers))
                ax2.barh(range(len(guessers)), guessers['avg_win_rate'], color=colors2)
                ax2.set_yticks(range(len(guessers)))
                ax2.set_yticklabels([f"{row['model'][:15]} ({row['role'][:10]})" for _, row in guessers.iterrows()], fontsize=9)
                ax2.set_xlabel('Average Win Rate')
                ax2.set_title('Best Guessers by Win Rate')
                ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/model_performance_by_role.png", dpi=300, bbox_inches='tight')
            plt.close()

        # =====================================================================
        # 3. Head-to-Head Matchup Heatmap
        # =====================================================================
        try:
            matchup_matrix, game_counts = self.create_matchup_matrix('hint_giver')
            if not matchup_matrix.empty and matchup_matrix.shape[0] > 1:
                plt.figure(figsize=(12, 10))
                # Truncate model names for display
                short_names = [name[:12] for name in matchup_matrix.index]
                display_matrix = matchup_matrix.copy()
                display_matrix.index = short_names
                display_matrix.columns = short_names

                mask = np.isnan(display_matrix.values)
                sns.heatmap(display_matrix, annot=True, fmt='.0%', cmap='RdYlGn',
                           center=0.5, mask=mask, square=True, linewidths=0.5,
                           cbar_kws={'label': 'Win Rate'})
                plt.title('Head-to-Head Win Rate Matrix (Hint Givers)\nRow model vs Column model')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/matchup_heatmap.png", dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"  Skipping matchup heatmap: {e}")

        # =====================================================================
        # 4. Elo Ratings Bar Chart
        # =====================================================================
        elo_df = self.calculate_elo_ratings()
        if not elo_df.empty:
            fig, ax = plt.subplots(figsize=(12, 8))

            models = elo_df['model'].head(10)
            x = np.arange(len(models))
            width = 0.35

            hg_ratings = elo_df['elo_hint_giver'].head(10)
            g_ratings = elo_df['elo_guesser'].head(10)

            bars1 = ax.bar(x - width/2, hg_ratings, width, label='Hint Giver Elo', color='#3498db')
            bars2 = ax.bar(x + width/2, g_ratings, width, label='Guesser Elo', color='#2ecc71')

            ax.axhline(y=1500, color='gray', linestyle='--', alpha=0.7, label='Starting Elo (1500)')
            ax.set_xlabel('Model')
            ax.set_ylabel('Elo Rating')
            ax.set_title('Elo Ratings by Model and Role')
            ax.set_xticks(x)
            ax.set_xticklabels([m[:12] for m in models], rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(min(1400, min(hg_ratings.min(), g_ratings.min()) - 50),
                       max(1600, max(hg_ratings.max(), g_ratings.max()) + 50))

            plt.tight_layout()
            plt.savefig(f"{output_dir}/elo_ratings.png", dpi=300, bbox_inches='tight')
            plt.close()

        # =====================================================================
        # 5. Role Versatility Scatter Plot
        # =====================================================================
        versatility_df = self.calculate_role_versatility()
        if not versatility_df.empty and len(versatility_df) > 1:
            plt.figure(figsize=(10, 8))

            scatter = plt.scatter(
                versatility_df['hint_giver_win_rate'],
                versatility_df['guesser_win_rate'],
                s=versatility_df['total_games'] * 5,
                c=versatility_df['combined_win_rate'],
                cmap='RdYlGn',
                alpha=0.7,
                edgecolors='black',
                linewidths=1
            )

            # Add model labels
            for _, row in versatility_df.iterrows():
                plt.annotate(row['model'][:10], (row['hint_giver_win_rate'], row['guesser_win_rate']),
                           fontsize=8, alpha=0.8, xytext=(5, 5), textcoords='offset points')

            plt.colorbar(scatter, label='Combined Win Rate')
            plt.xlabel('Hint Giver Win Rate')
            plt.ylabel('Guesser Win Rate')
            plt.title('Role Versatility: Hint Giver vs Guesser Performance\n(Size = Total Games)')

            # Add diagonal line (perfect versatility)
            lims = [0, 1]
            plt.plot(lims, lims, 'k--', alpha=0.3, label='Equal performance')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.legend()

            plt.tight_layout()
            plt.savefig(f"{output_dir}/role_versatility.png", dpi=300, bbox_inches='tight')
            plt.close()

        # =====================================================================
        # 6. Hint Efficiency vs Win Rate Scatter
        # =====================================================================
        hint_eff = self.analyze_hint_efficiency()
        if not hint_eff.empty and len(hint_eff) > 1:
            plt.figure(figsize=(10, 8))

            # Color by risk profile
            risk_colors = {'aggressive': '#e74c3c', 'balanced': '#f39c12', 'conservative': '#3498db'}
            colors = [risk_colors.get(p, '#95a5a6') for p in hint_eff['risk_profile']]

            plt.scatter(hint_eff['efficiency'], hint_eff['win_rate'],
                       s=hint_eff['hints_given'] * 2, c=colors, alpha=0.7,
                       edgecolors='black', linewidths=1)

            for _, row in hint_eff.iterrows():
                plt.annotate(f"{row['model'][:8]} ({row['team'][0]})",
                           (row['efficiency'], row['win_rate']),
                           fontsize=7, alpha=0.8, xytext=(3, 3), textcoords='offset points')

            # Legend for risk profiles
            for profile, color in risk_colors.items():
                plt.scatter([], [], c=color, label=profile.capitalize(), s=100)

            plt.xlabel('Hint Efficiency (Correct Guesses / Promised)')
            plt.ylabel('Win Rate')
            plt.title('Hint Giver Efficiency vs Win Rate\n(Size = Hints Given)')
            plt.legend(title='Risk Profile')
            plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/hint_efficiency_vs_winrate.png", dpi=300, bbox_inches='tight')
            plt.close()

        # =====================================================================
        # 7. Guesser Performance: First Guess vs Overall Accuracy
        # =====================================================================
        guesser_perf = self.analyze_guesser_performance()
        if not guesser_perf.empty and len(guesser_perf) > 1:
            plt.figure(figsize=(10, 8))

            # Color by bomb rate (darker = more bombs)
            bomb_rates = guesser_perf['bomb_rate']
            colors = plt.cm.Reds(bomb_rates / max(bomb_rates.max(), 0.01))

            plt.scatter(guesser_perf['first_guess_accuracy'],
                       guesser_perf['overall_accuracy'],
                       s=guesser_perf['games_played'] * 10,
                       c=colors, alpha=0.7,
                       edgecolors='black', linewidths=1)

            for _, row in guesser_perf.iterrows():
                plt.annotate(f"{row['model'][:8]} ({row['team'][0]})",
                           (row['first_guess_accuracy'], row['overall_accuracy']),
                           fontsize=7, alpha=0.8, xytext=(3, 3), textcoords='offset points')

            plt.xlabel('First Guess Accuracy')
            plt.ylabel('Overall Accuracy')
            plt.title('Guesser Performance: First Guess vs Overall\n(Size = Games, Color = Bomb Rate)')

            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(0, bomb_rates.max()))
            sm.set_array([])
            plt.colorbar(sm, ax=plt.gca(), label='Bomb Hit Rate')

            plt.tight_layout()
            plt.savefig(f"{output_dir}/guesser_accuracy.png", dpi=300, bbox_inches='tight')
            plt.close()

        # =====================================================================
        # 8. Confidence Interval Forest Plot
        # =====================================================================
        ci_df = self.calculate_confidence_intervals()
        if not ci_df.empty:
            # Take top 15 by win rate
            ci_top = ci_df.head(15)

            plt.figure(figsize=(12, 10))

            y_pos = range(len(ci_top))
            plt.hlines(y_pos, ci_top['ci_lower'], ci_top['ci_upper'], colors='#3498db', linewidth=2)
            plt.scatter(ci_top['win_rate'], y_pos, color='#e74c3c', s=100, zorder=5)

            plt.yticks(y_pos, [f"{row['model'][:12]} ({row['role'][:2]}, {row['team'][0]})"
                              for _, row in ci_top.iterrows()], fontsize=9)
            plt.xlabel('Win Rate (with 95% CI)')
            plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='50% baseline')
            plt.title('Win Rate Confidence Intervals (Wilson Score)')
            plt.legend()
            plt.xlim(0, 1)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/confidence_intervals.png", dpi=300, bbox_inches='tight')
            plt.close()

        # =====================================================================
        # 9. Hint Count Distribution
        # =====================================================================
        hint_patterns = self.analyze_hint_patterns()
        if hint_patterns['hint_count_distribution']:
            plt.figure(figsize=(10, 6))

            counts = sorted(hint_patterns['hint_count_distribution'].items())
            x_vals = [c[0] for c in counts]
            y_vals = [c[1] for c in counts]

            bars = plt.bar(x_vals, y_vals, color='#3498db', edgecolor='black')

            # Add success rate annotation if available
            if hint_patterns['success_by_count']:
                for i, (count, freq) in enumerate(counts):
                    if count in hint_patterns['success_by_count']:
                        success = hint_patterns['success_by_count'][count]['success']
                        plt.annotate(f'{success:.0%}', (count, freq),
                                   ha='center', va='bottom', fontsize=9, color='#27ae60')

            plt.xlabel('Hint Count (Number Promised)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Hint Counts\n(Green % = Success Rate)')

            plt.tight_layout()
            plt.savefig(f"{output_dir}/hint_count_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()

        # =====================================================================
        # 10. Error Analysis Stacked Bar
        # =====================================================================
        error_df = self.get_error_summary()
        if not error_df.empty and error_df['total_errors'].sum() > 0:
            # Filter to models with errors
            error_df = error_df[error_df['total_errors'] > 0].head(10)

            if not error_df.empty:
                plt.figure(figsize=(12, 6))

                models = error_df['model']
                x = np.arange(len(models))
                width = 0.6

                p1 = plt.bar(x, error_df['bomb_hits'], width, label='Bomb Hits', color='#e74c3c')
                p2 = plt.bar(x, error_df['invalid_offboard'], width, bottom=error_df['bomb_hits'],
                            label='Invalid (Offboard)', color='#f39c12')
                p3 = plt.bar(x, error_df['invalid_revealed'], width,
                            bottom=error_df['bomb_hits'] + error_df['invalid_offboard'],
                            label='Invalid (Revealed)', color='#9b59b6')
                p4 = plt.bar(x, error_df['invalid_other'], width,
                            bottom=error_df['bomb_hits'] + error_df['invalid_offboard'] + error_df['invalid_revealed'],
                            label='Invalid (Other)', color='#95a5a6')

                plt.xlabel('Model')
                plt.ylabel('Error Count')
                plt.title('Error Breakdown by Model')
                plt.xticks(x, [m[:12] for m in models], rotation=45, ha='right')
                plt.legend()

                plt.tight_layout()
                plt.savefig(f"{output_dir}/error_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()

        # =====================================================================
        # 11. Game Length Distribution
        # =====================================================================
        momentum_df = self.analyze_game_momentum()
        if not momentum_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Game length histogram
            axes[0].hist(momentum_df['total_turns'], bins=15, color='#3498db',
                        edgecolor='black', alpha=0.7)
            axes[0].axvline(momentum_df['total_turns'].mean(), color='#e74c3c',
                           linestyle='--', label=f'Mean: {momentum_df["total_turns"].mean():.1f}')
            axes[0].set_xlabel('Total Turns')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Game Length Distribution')
            axes[0].legend()

            # Lead changes distribution
            axes[1].hist(momentum_df['lead_changes'], bins=range(int(momentum_df['lead_changes'].max()) + 2),
                        color='#2ecc71', edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Number of Lead Changes')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Game Competitiveness (Lead Changes)')

            plt.tight_layout()
            plt.savefig(f"{output_dir}/game_dynamics.png", dpi=300, bbox_inches='tight')
            plt.close()

        # =====================================================================
        # 12. Summary Dashboard
        # =====================================================================
        try:
            fig = plt.figure(figsize=(16, 12))

            # Create grid
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # Blue advantage pie
            ax1 = fig.add_subplot(gs[0, 0])
            fma = self.analyze_first_mover_advantage()
            sizes = [fma['overall_blue_win_rate'], fma['overall_red_win_rate']]
            colors_pie = ['#3498db', '#e74c3c']
            ax1.pie(sizes, labels=['Blue Wins', 'Red Wins'], colors=colors_pie,
                   autopct='%1.1f%%', startangle=90)
            ax1.set_title('Win Distribution')

            # Top models by Elo
            ax2 = fig.add_subplot(gs[0, 1:])
            if not elo_df.empty:
                top_elo = elo_df.head(5)
                x = range(len(top_elo))
                ax2.barh(x, top_elo['elo_combined'], color='#9b59b6')
                ax2.set_yticks(x)
                ax2.set_yticklabels(top_elo['model'].str[:15])
                ax2.set_xlabel('Combined Elo')
                ax2.set_title('Top 5 Models by Elo Rating')
                ax2.axvline(x=1500, color='gray', linestyle='--', alpha=0.7)

            # Momentum summary
            ax3 = fig.add_subplot(gs[1, 0])
            momentum = self.get_momentum_summary()
            if momentum:
                metrics = ['Comeback\nRate', 'Avg Lead\nChanges', 'Avg Game\nLength']
                values = [momentum['comeback_rate'] * 100,
                         momentum['avg_lead_changes'],
                         momentum['avg_game_length']]
                ax3.bar(metrics, values, color=['#27ae60', '#f39c12', '#3498db'])
                ax3.set_title('Game Dynamics Summary')

            # Hint patterns
            ax4 = fig.add_subplot(gs[1, 1])
            hint_patterns = self.analyze_hint_patterns()
            if hint_patterns['total_hints'] > 0:
                metrics = ['Success\nRate', 'Perfect\nRate', 'Creativity']
                values = [hint_patterns['overall_success_rate'] * 100,
                         hint_patterns['perfect_hint_rate'] * 100,
                         hint_patterns['creativity_ratio'] * 100]
                ax4.bar(metrics, values, color=['#2ecc71', '#3498db', '#9b59b6'])
                ax4.set_ylabel('%')
                ax4.set_title('Hint Quality Metrics')

            # Error summary
            ax5 = fig.add_subplot(gs[1, 2])
            error_df = self.get_error_summary()
            if not error_df.empty:
                total_bombs = error_df['bomb_hits'].sum()
                total_invalid = (error_df['invalid_offboard'].sum() +
                               error_df['invalid_revealed'].sum() +
                               error_df['invalid_other'].sum())
                ax5.pie([total_bombs, total_invalid], labels=['Bomb Hits', 'Invalid Guesses'],
                       colors=['#e74c3c', '#f39c12'], autopct='%1.0f%%')
                ax5.set_title(f'Error Distribution (n={total_bombs + total_invalid})')

            # Versatility leaders
            ax6 = fig.add_subplot(gs[2, :])
            if not versatility_df.empty:
                top_vers = versatility_df.head(8)
                x = np.arange(len(top_vers))
                width = 0.35
                ax6.bar(x - width/2, top_vers['hint_giver_win_rate'], width,
                       label='Hint Giver', color='#3498db')
                ax6.bar(x + width/2, top_vers['guesser_win_rate'], width,
                       label='Guesser', color='#2ecc71')
                ax6.set_xticks(x)
                ax6.set_xticklabels(top_vers['model'].str[:12], rotation=45, ha='right')
                ax6.set_ylabel('Win Rate')
                ax6.set_title('Top Models: Performance by Role')
                ax6.legend()
                ax6.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

            plt.suptitle('Codenames Benchmark Dashboard', fontsize=14, fontweight='bold')
            plt.savefig(f"{output_dir}/dashboard.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  Skipping dashboard: {e}")

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
