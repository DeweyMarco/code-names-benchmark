"""
BAML-based agents for Codenames - Universal implementation for all LLM providers.

This module replaces the provider-specific agent files with a single, unified
implementation that uses BAML for prompt management and structured outputs.
"""
from typing import List, Optional
from enum import Enum

from game import Team, CardColor
from agents.base import HintGiver, Guesser, HintResponse
from baml_client.baml_client.sync_client import b  # Use sync client
from baml_py import ClientRegistry


class BAMLModel(Enum):
    """Available BAML client models - matches clients defined in baml_src/clients.baml"""

    # OpenAI - GPT-5 Series (Latest - August 2025)
    GPT5 = "GPT5"
    GPT5_MINI = "GPT5Mini"
    GPT5_NANO = "GPT5Nano"
    GPT5_CHAT = "GPT5Chat"
    GPT5_PRO = "GPT5Pro"

    # OpenAI - GPT-4.1 Series (April 2025)
    GPT41 = "GPT41"
    GPT41_MINI = "GPT41Mini"
    GPT41_NANO = "GPT41Nano"

    # OpenAI - Reasoning Models (o-series)
    O4_MINI = "O4Mini"
    O3_MINI = "O3Mini"
    O3 = "O3"
    O1 = "O1"
    O1_MINI = "O1Mini"
    O1_PREVIEW = "O1Preview"

    # OpenAI - GPT-4o Series (Multimodal)
    GPT4O = "GPT4o"
    GPT4O_MINI = "GPT4oMini"
    GPT4O_20240806 = "GPT4o_20240806"
    GPT4O_MINI_20240718 = "GPT4oMini_20240718"

    # OpenAI - GPT-4 Turbo Series
    GPT4_TURBO = "GPT4Turbo"
    GPT4_TURBO_PREVIEW = "GPT4TurboPreview"
    GPT4_0125_PREVIEW = "GPT4_0125Preview"
    GPT4_1106_PREVIEW = "GPT4_1106Preview"

    # OpenAI - GPT-4 Base Series
    GPT4 = "GPT4"
    GPT4_32K = "GPT4_32k"
    GPT4_0613 = "GPT4_0613"

    # OpenAI - GPT-3.5 Series
    GPT35_TURBO = "GPT35Turbo"
    GPT35_TURBO_16K = "GPT35Turbo16k"
    GPT35_TURBO_INSTRUCT = "GPT35TurboInstruct"

    # Anthropic - Claude 4.5 Series (Latest - October 2025)
    CLAUDE_SONNET_45 = "ClaudeSonnet45"
    CLAUDE_HAIKU_45 = "ClaudeHaiku45"

    # Anthropic - Claude 4.1 Series
    CLAUDE_OPUS_41 = "ClaudeOpus41"

    # Anthropic - Claude 4 Series (Legacy - May 2025)
    CLAUDE_SONNET_4 = "ClaudeSonnet4"
    CLAUDE_OPUS_4 = "ClaudeOpus4"

    # Anthropic - Claude 3.7 Series (Legacy - February 2025)
    CLAUDE_SONNET_37 = "ClaudeSonnet37"

    # Anthropic - Claude 3.5 Series (Legacy)
    CLAUDE_HAIKU_35 = "ClaudeHaiku35"

    # Anthropic - Claude 3 Series (Legacy)
    CLAUDE_HAIKU_3 = "ClaudeHaiku3"

    # Google - Gemini 2.5 Series (Latest - January 2025)
    GEMINI_25_PRO = "Gemini25Pro"
    GEMINI_25_FLASH = "Gemini25Flash"
    GEMINI_25_FLASH_LITE = "Gemini25FlashLite"

    # Google - Gemini 2.0 Series
    GEMINI_20_FLASH = "Gemini20Flash"
    GEMINI_20_FLASH_LITE = "Gemini20FlashLite"

    # DeepSeek V3.2-Exp
    DEEPSEEK_CHAT = "DeepSeekChat"
    DEEPSEEK_REASONER = "DeepSeekReasoner"

    # xAI - Grok 4 Series
    GROK4 = "Grok4"
    GROK4_FAST_REASONING = "Grok4FastReasoning"
    GROK4_FAST_NON_REASONING = "Grok4FastNonReasoning"

    # xAI - Grok 3 Series
    GROK3 = "Grok3"
    GROK3_FAST = "Grok3Fast"
    GROK3_MINI = "Grok3Mini"
    GROK3_MINI_FAST = "Grok3MiniFast"

    # Meta Llama (via Together AI or similar)
    LLAMA = "Llama"


class BAMLHintGiver(HintGiver):
    """
    Universal BAML-based hint giver (spymaster) that works with any LLM provider.

    This single class replaces all the provider-specific hint giver implementations
    (OpenAIHintGiver, AnthropicHintGiver, etc.) with a unified BAML-based approach.
    """

    def __init__(self, team: Team, model: BAMLModel = BAMLModel.GPT4O_MINI):
        """
        Initialize BAML hint giver.

        Args:
            team: Team this agent plays for
            model: BAML model/client to use (default: GPT4oMini)
        """
        super().__init__(team)
        self.model = model

    def give_hint(
        self,
        my_words: List[str],
        opponent_words: List[str],
        neutral_words: List[str],
        bomb_word: str,
        revealed_words: List[str],
        board_words: List[str]
    ) -> HintResponse:
        """
        Generate hint using BAML.

        This method automatically handles:
        - Prompt templating
        - Structured output parsing
        - Error handling and retries
        - Type validation
        """
        try:
            # Create ClientRegistry and set the model as primary
            cr = ClientRegistry()
            cr.set_primary(self.model.value)

            # Call BAML function with the specified client
            baml_response = b.GiveHint(
                team=self.team.value,
                my_words=my_words,
                opponent_words=opponent_words,
                neutral_words=neutral_words,
                bomb_word=bomb_word,
                revealed_words=revealed_words,
                baml_options={"client_registry": cr}
            )

            # BAML automatically returns a validated HintResponse-compatible object
            return HintResponse(
                word=baml_response.word,
                count=baml_response.count
            )

        except Exception as e:
            print(f"BAMLHintGiver Error ({self.model.value}): {e}")
            # Fallback to a safe hint
            return HintResponse(word="connection", count=1)


class BAMLGuesser(Guesser):
    """
    Universal BAML-based guesser (field operative) that works with any LLM provider.

    This single class replaces all the provider-specific guesser implementations
    (OpenAIGuesser, AnthropicGuesser, etc.) with a unified BAML-based approach.
    """

    def __init__(self, team: Team, model: BAMLModel = BAMLModel.GPT4O_MINI):
        """
        Initialize BAML guesser.

        Args:
            team: Team this agent plays for
            model: BAML model/client to use (default: GPT4oMini)
        """
        super().__init__(team)
        self.model = model

    def make_guesses(
        self,
        hint_word: str,
        hint_count: int,
        board_words: List[str],
        revealed_words: List[str]
    ) -> List[str]:
        """
        Make guesses using BAML.

        This method automatically handles:
        - Prompt templating
        - Structured output parsing
        - Error handling and retries
        - Type validation
        """
        try:
            # Create ClientRegistry and set the model as primary
            cr = ClientRegistry()
            cr.set_primary(self.model.value)

            # Call BAML function with the specified client
            baml_response = b.MakeGuesses(
                team=self.team.value,
                hint_word=hint_word,
                hint_count=hint_count,
                board_words=board_words,
                revealed_words=revealed_words,
                baml_options={"client_registry": cr}
            )

            # BAML automatically returns a validated list of guesses
            return baml_response.guesses

        except Exception as e:
            print(f"BAMLGuesser Error ({self.model.value}): {e}")
            # Fallback: pass turn
            return []

    def process_result(self, guessed_word: str, was_correct: bool, color: CardColor):
        """BAML agents don't maintain state between guesses."""
        pass


# ============================================================================
# CONVENIENCE FACTORY FUNCTIONS
# ============================================================================

def create_hint_giver(provider: str, model: Optional[str] = None, team: Team = Team.BLUE) -> BAMLHintGiver:
    """
    Factory function to create a hint giver with a specific provider/model.

    Args:
        provider: Provider name ("openai", "anthropic", "google", etc.)
        model: Specific model name (optional, uses default if not provided)
        team: Team for this agent

    Returns:
        BAMLHintGiver configured for the specified provider/model

    Examples:
        >>> hint_giver = create_hint_giver("openai", "gpt-4o", Team.BLUE)
        >>> hint_giver = create_hint_giver("anthropic", team=Team.RED)
        >>> hint_giver = create_hint_giver("google", "gemini-pro", Team.BLUE)
    """
    # Map provider/model combinations to BAMLModel enum
    model_map = {
        ("openai", "gpt-4o-mini"): BAMLModel.GPT4O_MINI,
        ("openai", "gpt-4o"): BAMLModel.GPT4O,
        ("openai", "gpt-4-turbo"): BAMLModel.GPT4_TURBO,
        ("openai", None): BAMLModel.GPT4O_MINI,
        ("anthropic", "claude-sonnet-4-5-20250929"): BAMLModel.CLAUDE_SONNET,
        ("anthropic", "claude-3-5-sonnet-20241022"): BAMLModel.CLAUDE_35_SONNET,
        ("anthropic", "claude-3-opus-20240229"): BAMLModel.CLAUDE_OPUS,
        ("anthropic", None): BAMLModel.CLAUDE_SONNET,
        # Gemini 2.5 Series
        ("google", "gemini-2.5-pro"): BAMLModel.GEMINI_25_PRO,
        ("google", "gemini-2.5-flash"): BAMLModel.GEMINI_25_FLASH,
        ("google", "gemini-2.5-flash-lite"): BAMLModel.GEMINI_25_FLASH_LITE,
        # Gemini 2.0 Series
        ("google", "gemini-2.0-flash"): BAMLModel.GEMINI_20_FLASH,
        ("google", "gemini-2.0-flash-lite"): BAMLModel.GEMINI_20_FLASH_LITE,
        ("google", None): BAMLModel.GEMINI_25_FLASH,  # Default to latest
        ("deepseek", None): BAMLModel.DEEPSEEK,
        ("grok", None): BAMLModel.GROK,
        ("llama", None): BAMLModel.LLAMA,
    }

    baml_model = model_map.get((provider.lower(), model), BAMLModel.GPT4O_MINI)
    return BAMLHintGiver(team=team, model=baml_model)


def create_guesser(provider: str, model: Optional[str] = None, team: Team = Team.BLUE) -> BAMLGuesser:
    """
    Factory function to create a guesser with a specific provider/model.

    Args:
        provider: Provider name ("openai", "anthropic", "google", etc.)
        model: Specific model name (optional, uses default if not provided)
        team: Team for this agent

    Returns:
        BAMLGuesser configured for the specified provider/model

    Examples:
        >>> guesser = create_guesser("openai", "gpt-4o", Team.BLUE)
        >>> guesser = create_guesser("anthropic", team=Team.RED)
        >>> guesser = create_guesser("google", "gemini-pro", Team.BLUE)
    """
    # Map provider/model combinations to BAMLModel enum
    model_map = {
        ("openai", "gpt-4o-mini"): BAMLModel.GPT4O_MINI,
        ("openai", "gpt-4o"): BAMLModel.GPT4O,
        ("openai", "gpt-4-turbo"): BAMLModel.GPT4_TURBO,
        ("openai", None): BAMLModel.GPT4O_MINI,
        ("anthropic", "claude-sonnet-4-5-20250929"): BAMLModel.CLAUDE_SONNET,
        ("anthropic", "claude-3-5-sonnet-20241022"): BAMLModel.CLAUDE_35_SONNET,
        ("anthropic", "claude-3-opus-20240229"): BAMLModel.CLAUDE_OPUS,
        ("anthropic", None): BAMLModel.CLAUDE_SONNET,
        # Gemini 2.5 Series
        ("google", "gemini-2.5-pro"): BAMLModel.GEMINI_25_PRO,
        ("google", "gemini-2.5-flash"): BAMLModel.GEMINI_25_FLASH,
        ("google", "gemini-2.5-flash-lite"): BAMLModel.GEMINI_25_FLASH_LITE,
        # Gemini 2.0 Series
        ("google", "gemini-2.0-flash"): BAMLModel.GEMINI_20_FLASH,
        ("google", "gemini-2.0-flash-lite"): BAMLModel.GEMINI_20_FLASH_LITE,
        ("google", None): BAMLModel.GEMINI_25_FLASH,  # Default to latest
        ("deepseek", None): BAMLModel.DEEPSEEK,
        ("grok", None): BAMLModel.GROK,
        ("llama", None): BAMLModel.LLAMA,
    }

    baml_model = model_map.get((provider.lower(), model), BAMLModel.GPT4O_MINI)
    return BAMLGuesser(team=team, model=baml_model)
