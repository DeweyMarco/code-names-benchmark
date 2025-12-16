"""
Simple Codenames Demo - Configurable AI Model Edition

This demo runs a single game where you can easily configure which AI models
play the four roles by changing the Players class variables:
- Blue Team Hint Giver (Spymaster)
- Blue Team Guesser (Field Operative)
- Red Team Hint Giver (Spymaster)
- Red Team Guesser (Field Operative)

The game is VERBOSE so you can see every step clearly.

Setup:
1. Install dependencies: pip install -r requirements.txt
2. Set your API key(s) in .env file (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.)
3. Configure models in the Players class (lines 35-38)
4. Run: python3 demo_simple_game.py

Available Models:
OpenAI:
  - GPT-5: GPT5, GPT5_MINI, GPT5_NANO, GPT5_CHAT, GPT5_PRO
  - GPT-4.1: GPT41, GPT41_MINI, GPT41_NANO
  - Reasoning: O4_MINI, O3_MINI, O3, O1, O1_MINI, O1_PREVIEW
  - GPT-4o: GPT4O, GPT4O_MINI, GPT4O_20240806, GPT4O_MINI_20240718
  - GPT-4: GPT4_TURBO, GPT4, GPT4_32K, GPT4_0613
  - GPT-3.5: GPT35_TURBO, GPT35_TURBO_16K, GPT35_TURBO_INSTRUCT
Anthropic:
  - Claude 4.5: CLAUDE_SONNET_45, CLAUDE_HAIKU_45
  - Claude 4.1: CLAUDE_OPUS_41
  - Claude 4: CLAUDE_SONNET_4, CLAUDE_OPUS_4
  - Claude 3: CLAUDE_SONNET_37, CLAUDE_HAIKU_35, CLAUDE_HAIKU_3
Google:
  - Gemini 2.5: GEMINI_25_PRO, GEMINI_25_FLASH, GEMINI_25_FLASH_LITE
  - Gemini 2.0: GEMINI_20_FLASH, GEMINI_20_FLASH_LITE
DeepSeek: DEEPSEEK_CHAT, DEEPSEEK_REASONER
xAI Grok:
  - Grok 4: GROK4, GROK4_FAST_REASONING, GROK4_FAST_NON_REASONING
  - Grok 3: GROK3, GROK3_FAST, GROK3_MINI, GROK3_MINI_FAST
Meta: LLAMA
"""

import os
import sys
from dotenv import load_dotenv

from utils import generate_word_list
from game import Board, Team, CardColor
from agents.llm import BAMLHintGiver, BAMLGuesser, BAMLModel
from orchestrator import GameRunner
from model_config import get_model_display_name

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURE YOUR MODELS HERE - Just change these to try different AI models!
# ============================================================================
class Players:
    RED_HINT_GIVER = BAMLModel.GEMINI_25_FLASH
    RED_GUESSER = BAMLModel.GEMINI_25_FLASH
    BLUE_HINT_GIVER = BAMLModel.GEMINI_25_FLASH
    BLUE_GUESSER = BAMLModel.GEMINI_25_FLASH

# ANSI color codes
class Colors:
    BLUE = '\033[96m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


# Model to API key mapping
MODEL_TO_API_KEY = {
    # OpenAI - GPT-5 Series
    BAMLModel.GPT5: "OPENAI_API_KEY",
    BAMLModel.GPT5_MINI: "OPENAI_API_KEY",
    BAMLModel.GPT5_NANO: "OPENAI_API_KEY",
    BAMLModel.GPT5_CHAT: "OPENAI_API_KEY",
    BAMLModel.GPT5_PRO: "OPENAI_API_KEY",
    # OpenAI - GPT-4.1 Series
    BAMLModel.GPT41: "OPENAI_API_KEY",
    BAMLModel.GPT41_MINI: "OPENAI_API_KEY",
    BAMLModel.GPT41_NANO: "OPENAI_API_KEY",
    # OpenAI - Reasoning Models (o-series)
    BAMLModel.O4_MINI: "OPENAI_API_KEY",
    BAMLModel.O3_MINI: "OPENAI_API_KEY",
    BAMLModel.O3: "OPENAI_API_KEY",
    BAMLModel.O1: "OPENAI_API_KEY",
    BAMLModel.O1_MINI: "OPENAI_API_KEY",
    BAMLModel.O1_PREVIEW: "OPENAI_API_KEY",
    # OpenAI - GPT-4o Series
    BAMLModel.GPT4O: "OPENAI_API_KEY",
    BAMLModel.GPT4O_MINI: "OPENAI_API_KEY",
    BAMLModel.GPT4O_20240806: "OPENAI_API_KEY",
    BAMLModel.GPT4O_MINI_20240718: "OPENAI_API_KEY",
    # OpenAI - GPT-4 Turbo Series
    BAMLModel.GPT4_TURBO: "OPENAI_API_KEY",
    BAMLModel.GPT4_TURBO_PREVIEW: "OPENAI_API_KEY",
    BAMLModel.GPT4_0125_PREVIEW: "OPENAI_API_KEY",
    BAMLModel.GPT4_1106_PREVIEW: "OPENAI_API_KEY",
    # OpenAI - GPT-4 Base Series
    BAMLModel.GPT4: "OPENAI_API_KEY",
    BAMLModel.GPT4_32K: "OPENAI_API_KEY",
    BAMLModel.GPT4_0613: "OPENAI_API_KEY",
    # OpenAI - GPT-3.5 Series
    BAMLModel.GPT35_TURBO: "OPENAI_API_KEY",
    BAMLModel.GPT35_TURBO_16K: "OPENAI_API_KEY",
    BAMLModel.GPT35_TURBO_INSTRUCT: "OPENAI_API_KEY",
    # Anthropic - Claude 4.5 Series
    BAMLModel.CLAUDE_SONNET_45: "ANTHROPIC_API_KEY",
    BAMLModel.CLAUDE_HAIKU_45: "ANTHROPIC_API_KEY",
    # Anthropic - Claude 4.1 Series
    BAMLModel.CLAUDE_OPUS_41: "ANTHROPIC_API_KEY",
    # Anthropic - Claude 4 Series
    BAMLModel.CLAUDE_SONNET_4: "ANTHROPIC_API_KEY",
    BAMLModel.CLAUDE_OPUS_4: "ANTHROPIC_API_KEY",
    # Anthropic - Claude 3.7 Series
    BAMLModel.CLAUDE_SONNET_37: "ANTHROPIC_API_KEY",
    # Anthropic - Claude 3.5 Series
    BAMLModel.CLAUDE_HAIKU_35: "ANTHROPIC_API_KEY",
    # Anthropic - Claude 3 Series
    BAMLModel.CLAUDE_HAIKU_3: "ANTHROPIC_API_KEY",
    # Google - Gemini 2.5 Series
    BAMLModel.GEMINI_25_PRO: "GOOGLE_API_KEY",
    BAMLModel.GEMINI_25_FLASH: "GOOGLE_API_KEY",
    BAMLModel.GEMINI_25_FLASH_LITE: "GOOGLE_API_KEY",
    # Google - Gemini 2.0 Series
    BAMLModel.GEMINI_20_FLASH: "GOOGLE_API_KEY",
    BAMLModel.GEMINI_20_FLASH_LITE: "GOOGLE_API_KEY",
    # DeepSeek V3.2-Exp
    BAMLModel.DEEPSEEK_CHAT: "DEEPSEEK_API_KEY",
    BAMLModel.DEEPSEEK_REASONER: "DEEPSEEK_API_KEY",
    # xAI - Grok 4 Series
    BAMLModel.GROK4: "XAI_API_KEY",
    BAMLModel.GROK4_FAST_REASONING: "XAI_API_KEY",
    BAMLModel.GROK4_FAST_NON_REASONING: "XAI_API_KEY",
    # xAI - Grok 3 Series
    BAMLModel.GROK3: "XAI_API_KEY",
    BAMLModel.GROK3_FAST: "XAI_API_KEY",
    BAMLModel.GROK3_MINI: "XAI_API_KEY",
    BAMLModel.GROK3_MINI_FAST: "XAI_API_KEY",
    # Meta Llama (via Together AI)
    BAMLModel.LLAMA: "TOGETHER_API_KEY",
}


def check_api_keys() -> dict:
    """
    Check which API keys are available and which models are being used.
    Returns a dict with model -> (key_found, key_names) mapping.
    """
    models_in_use = {
        Players.RED_HINT_GIVER,
        Players.RED_GUESSER,
        Players.BLUE_HINT_GIVER,
        Players.BLUE_GUESSER,
    }

    results = {}
    for model in models_in_use:
        key_names = MODEL_TO_API_KEY.get(model, [])
        if isinstance(key_names, str):
            key_names = [key_names]

        # Check if any of the possible keys exist
        key_found = any(os.getenv(key) for key in key_names)
        results[model] = (key_found, key_names)

    return results


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def main():
    """Run a configurable AI Codenames game."""

    # Determine which models are being used
    models_in_use = {
        Players.RED_HINT_GIVER,
        Players.RED_GUESSER,
        Players.BLUE_HINT_GIVER,
        Players.BLUE_GUESSER,
    }

    # Create a friendly title based on models in use
    if len(models_in_use) == 1:
        model_name = get_model_display_name(list(models_in_use)[0])
        title = f"CODENAMES: {model_name.upper()} DEMO"
    else:
        title = "CODENAMES: MULTI-MODEL DEMO"

    print_section(f"{Colors.CYAN}{title}{Colors.RESET}")
    print("\nThis demo runs a complete Codenames game with configurable AI models.")
    print("\nYou'll see verbose output showing:")
    print("  • Game setup and board state")
    print("  • Each hint given by the spymasters")
    print("  • Each guess made by the field operatives")
    print("  • Results after each turn")
    print("  • Final game outcome and statistics")

    # Display configured models
    print_subsection("Configured Models")
    print(f"  {Colors.BLUE}Blue Team Hint Giver:{Colors.RESET} {get_model_display_name(Players.BLUE_HINT_GIVER)}")
    print(f"  {Colors.BLUE}Blue Team Guesser:{Colors.RESET} {get_model_display_name(Players.BLUE_GUESSER)}")
    print(f"  {Colors.RED}Red Team Hint Giver:{Colors.RESET} {get_model_display_name(Players.RED_HINT_GIVER)}")
    print(f"  {Colors.RED}Red Team Guesser:{Colors.RESET} {get_model_display_name(Players.RED_GUESSER)}")

    # Check API keys
    print_subsection("Checking API Keys")
    api_key_status = check_api_keys()

    missing_keys = []
    for model, (key_found, key_names) in api_key_status.items():
        model_name = get_model_display_name(model)
        if key_found:
            print(f"  {Colors.GREEN}[OK] {model_name} - API key found{Colors.RESET}")
        else:
            print(f"  {Colors.RED}[MISSING] {model_name} - API key not found{Colors.RESET}")
            missing_keys.extend(key_names)

    # If any keys are missing, show error and exit
    if missing_keys:
        print(f"\n{Colors.RED}ERROR: Required API key(s) not found!{Colors.RESET}")
        print("\nPlease set up the required API keys:")
        print("  1. Copy .env.example to .env (if you haven't already)")
        print("  2. Get API key(s) from:")

        unique_keys = set(missing_keys)
        if "OPENAI_API_KEY" in unique_keys:
            print("     • OpenAI: https://platform.openai.com/api-keys")
        if "ANTHROPIC_API_KEY" in unique_keys:
            print("     • Anthropic: https://console.anthropic.com/settings/keys")
        if "GOOGLE_API_KEY" in unique_keys:
            print("     • Google/Gemini: https://aistudio.google.com/app/apikey")
        if "DEEPSEEK_API_KEY" in unique_keys:
            print("     • DeepSeek: https://platform.deepseek.com/")
        if "XAI_API_KEY" in unique_keys:
            print("     • xAI (Grok): https://console.x.ai/")
        if "TOGETHER_API_KEY" in unique_keys:
            print("     • Together AI (for Llama): https://api.together.xyz/")

        print(f"\n  3. Add the key(s) to your .env file:")
        for key in sorted(unique_keys):
            print(f"     {key}=your_key_here")
        print("  4. Run this script again")
        sys.exit(1)

    # Generate word list
    print_subsection("Generating Game Board")
    words = generate_word_list(25)
    print(f"{Colors.GREEN}[OK] Generated {len(words)} random words for the board{Colors.RESET}")

    # Create board
    board = Board(words)
    print(f"{Colors.GREEN}[OK] Board created with:{Colors.RESET}")
    print(f"    • {len(board.get_words_by_color(CardColor.BLUE))} Blue team words")
    print(f"    • {len(board.get_words_by_color(CardColor.RED))} Red team words")
    print(f"    • {len(board.get_words_by_color(CardColor.NEUTRAL))} Neutral words")
    print(f"    • {len(board.get_words_by_color(CardColor.BOMB))} Bomb word (instant loss!)")

    # Display the board (as spymaster would see it)
    print_subsection("Board Layout (Spymaster View)")
    print("\nWords with their hidden colors:")
    for i, word in enumerate(board.all_words, 1):
        color = board.get_color(word)
        color_code, color_label = {
            CardColor.BLUE: (Colors.BLUE, "[BLUE]"),
            CardColor.RED: (Colors.RED, "[RED]"),
            CardColor.NEUTRAL: (Colors.WHITE, "[NEUTRAL]"),
            CardColor.BOMB: (Colors.MAGENTA, "[BOMB]")
        }[color]
        print(f"  {i:2d}. {color_code}{color_label}{Colors.RESET} {word.upper()}")

    # Create agents for all four roles
    print_subsection("Creating AI Agents")

    blue_hint_giver = BAMLHintGiver(Team.BLUE, model=Players.BLUE_HINT_GIVER)
    print(f"  {Colors.GREEN}[OK] Blue Team Hint Giver ({get_model_display_name(Players.BLUE_HINT_GIVER)}){Colors.RESET}")

    blue_guesser = BAMLGuesser(Team.BLUE, model=Players.BLUE_GUESSER)
    print(f"  {Colors.GREEN}[OK] Blue Team Guesser ({get_model_display_name(Players.BLUE_GUESSER)}){Colors.RESET}")

    red_hint_giver = BAMLHintGiver(Team.RED, model=Players.RED_HINT_GIVER)
    print(f"  {Colors.GREEN}[OK] Red Team Hint Giver ({get_model_display_name(Players.RED_HINT_GIVER)}){Colors.RESET}")

    red_guesser = BAMLGuesser(Team.RED, model=Players.RED_GUESSER)
    print(f"  {Colors.GREEN}[OK] Red Team Guesser ({get_model_display_name(Players.RED_GUESSER)}){Colors.RESET}")

    # Create game runner
    print_subsection("Initializing Game Runner")
    runner = GameRunner(
        board=board,
        blue_hint_giver=blue_hint_giver,
        blue_guesser=blue_guesser,
        red_hint_giver=red_hint_giver,
        red_guesser=red_guesser,
        max_turns=50,
        verbose=True,  # VERBOSE MODE - Shows all details!
        game_id="configurable_demo"
    )
    print(f"{Colors.GREEN}[OK] Game runner initialized{Colors.RESET}")
    print(f"{Colors.GREEN}[OK] Verbose mode: ENABLED (you'll see everything!){Colors.RESET}")
    print(f"{Colors.GREEN}[OK] Max turns: 50{Colors.RESET}")

    # Start the game!
    print_section(f"{Colors.CYAN}GAME START!{Colors.RESET}")
    print("\nBlue team goes first (they have 9 words to find)")
    print("Red team goes second (they have 8 words to find)")
    print(f"\nRemember: If anyone picks the {Colors.MAGENTA}[BOMB]{Colors.RESET}, their team loses immediately!\n")
    print("Starting game... (This may take 1-2 minutes as the AI thinks)")
    print("-" * 70)

    # Run the game
    result = runner.run()

    # Display final results
    print_section(f"{Colors.CYAN}FINAL RESULTS{Colors.RESET}")

    print(f"\n{Colors.BOLD}Game Outcome: {result.outcome.value.upper()}{Colors.RESET}")

    if result.winner:
        winner_color = Colors.BLUE if result.winner == Team.BLUE else Colors.RED
        print(f"\n{winner_color}Winner: {result.winner.value.upper()} TEAM!{Colors.RESET}")
    else:
        print(f"\n{Colors.YELLOW}Result: DRAW (max turns reached){Colors.RESET}")

    print(f"\n{Colors.CYAN}Game Statistics:{Colors.RESET}")
    print(f"  • Total Turns: {result.total_turns}")
    print(f"  • Blue Team Words Remaining: {result.final_scores[0]}")
    print(f"  • Red Team Words Remaining: {result.final_scores[1]}")

    if result.error:
        print(f"\n{Colors.YELLOW}Warning: {result.error}{Colors.RESET}")

    # Show turn-by-turn summary
    print_section(f"{Colors.CYAN}TURN-BY-TURN SUMMARY{Colors.RESET}")

    for turn_data in result.snapshot['turn_history']:
        team = turn_data['team']
        team_color = Colors.BLUE if team.lower() == 'blue' else Colors.RED
        hint_word = turn_data['hint_word']
        hint_count = turn_data['hint_count']
        guesses = turn_data['guesses']

        print(f"\n{team_color}Turn {turn_data['turn_number']} ({team.upper()} TEAM){Colors.RESET}")
        print(f"  Hint: '{hint_word.upper()}' for {hint_count} word(s)")

        if guesses:
            print(f"  Guesses:")
            for i, guess in enumerate(guesses, 1):
                guess_word = guess['word']
                correct = guess['correct']
                color = guess.get('color', 'unknown')

                if correct:
                    print(f"    {i}. {Colors.GREEN}[CORRECT]{Colors.RESET} {guess_word.upper()} ({color})")
                else:
                    print(f"    {i}. {Colors.RED}[WRONG]{Colors.RESET} {guess_word.upper()} ({color})")
        else:
            print(f"  Guesses: (none - passed)")

    # Show final board state
    print_section(f"{Colors.CYAN}FINAL BOARD STATE{Colors.RESET}")
    print("\nAll revealed words:")

    revealed = result.snapshot.get('revealed_words', [])
    unrevealed = [w for w in board.all_words if w not in revealed]

    print(f"\n{Colors.GREEN}Revealed ({len(revealed)} words):{Colors.RESET}")
    for word in revealed:
        color = board.get_color(word)
        color_code, color_label = {
            CardColor.BLUE: (Colors.BLUE, "[BLUE]"),
            CardColor.RED: (Colors.RED, "[RED]"),
            CardColor.NEUTRAL: (Colors.WHITE, "[NEUTRAL]"),
            CardColor.BOMB: (Colors.MAGENTA, "[BOMB]")
        }[color]
        print(f"  {color_code}{color_label}{Colors.RESET} {word.upper()}")

    print(f"\n{Colors.YELLOW}Unrevealed ({len(unrevealed)} words):{Colors.RESET}")
    for word in unrevealed:
        color = board.get_color(word)
        color_code, color_label = {
            CardColor.BLUE: (Colors.BLUE, "[BLUE]"),
            CardColor.RED: (Colors.RED, "[RED]"),
            CardColor.NEUTRAL: (Colors.WHITE, "[NEUTRAL]"),
            CardColor.BOMB: (Colors.MAGENTA, "[BOMB]")
        }[color]
        print(f"  {color_code}{color_label}{Colors.RESET} {word.upper()}")

    # Done!
    print_section(f"{Colors.GREEN}DEMO COMPLETE!{Colors.RESET}")
    print("\nTo run another game with different models:")
    print("  1. Edit the Players class in this file (lines 41-44)")
    print("  2. Change the BAMLModel values to any available model")
    print("  3. Run: python3 demo_simple_game.py")
    print("\nTo try other demos, check out: demo_llm_game.py")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Game interrupted by user (Ctrl+C){Colors.RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n{Colors.RED}ERROR: {e}{Colors.RESET}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()
        print("\nIf you need help, check the README.md or create an issue on GitHub.")
        sys.exit(1)
