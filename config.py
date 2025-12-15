"""
Configuration constants for the Codenames benchmark framework.

This module centralizes all game constants and configuration values
to make them easily adjustable and maintainable.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class GameConfig:
    """Game configuration constants."""

    # Board configuration
    BOARD_SIZE: int = 25  # Total number of words on the board
    BLUE_WORDS: int = 9   # Blue team words (starting team)
    RED_WORDS: int = 8    # Red team words
    NEUTRAL_WORDS: int = 7  # Neutral words
    BOMB_COUNT: int = 1   # Number of bomb/assassin words

    # Game rules
    MAX_TURNS: int = 50   # Maximum turns before game ends (prevents infinite loops)
    MAX_GUESSES_PER_TURN: Optional[int] = None  # Max guesses per turn (None = hint_count + 1)
    STARTING_TEAM: str = "BLUE"  # Which team starts (BLUE or RED)

    # Validation
    MIN_HINT_COUNT: int = 1  # Minimum number for hints
    MAX_HINT_COUNT: int = 9  # Maximum number for hints (usually team words + 1)

    def validate(self) -> bool:
        """Validate that the configuration is internally consistent."""
        total = self.BLUE_WORDS + self.RED_WORDS + self.NEUTRAL_WORDS + self.BOMB_COUNT
        if total != self.BOARD_SIZE:
            raise ValueError(
                f"Word counts don't add up to board size: "
                f"{self.BLUE_WORDS} + {self.RED_WORDS} + {self.NEUTRAL_WORDS} + {self.BOMB_COUNT} = {total}, "
                f"expected {self.BOARD_SIZE}"
            )
        return True

    @classmethod
    def custom(cls, board_size: int = 25, starting_team: str = "BLUE") -> "GameConfig":
        """
        Create a custom game configuration with different board size.

        Args:
            board_size: Total number of words (must be odd for fair play)
            starting_team: Which team starts (BLUE or RED)

        Returns:
            New GameConfig instance with proportional word distributions
        """
        if board_size < 9:
            raise ValueError("Board size must be at least 9")
        if board_size % 2 == 0:
            raise ValueError("Board size should be odd for fair play")

        # Calculate proportional distribution
        # Starting team gets one extra word
        starting_words = (board_size - 1) // 3 + 1  # Roughly 1/3 + 1
        other_words = starting_words - 1
        neutral_words = board_size - starting_words - other_words - 1  # -1 for bomb

        return cls(
            BOARD_SIZE=board_size,
            BLUE_WORDS=starting_words if starting_team == "BLUE" else other_words,
            RED_WORDS=starting_words if starting_team == "RED" else other_words,
            NEUTRAL_WORDS=neutral_words,
            BOMB_COUNT=1,
            MAX_TURNS=max(50, board_size * 2),  # Scale with board size
            MAX_GUESSES_PER_TURN=None,
            STARTING_TEAM=starting_team,
            MIN_HINT_COUNT=1,
            MAX_HINT_COUNT=max(starting_words, other_words) + 1
        )


@dataclass
class LLMConfig:
    """LLM agent configuration."""

    # Temperature settings
    DEFAULT_TEMPERATURE: float = 0.7
    MIN_TEMPERATURE: float = 0.0
    MAX_TEMPERATURE: float = 2.0

    # Response settings
    MAX_TOKENS: int = 1024
    RESPONSE_TIMEOUT: int = 30  # seconds

    # Retry settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0  # seconds

    # Model defaults
    OPENAI_DEFAULT_MODEL: str = "gpt-4o-mini"  # Cost-effective default; consider gpt-5-mini or gpt-5-nano for newer features
    ANTHROPIC_DEFAULT_MODEL: str = "claude-sonnet-4-5-20250929"  # Latest Claude Sonnet 4.5 (Sept 2025)
    GEMINI_DEFAULT_MODEL: str = "gemini-2.5-flash"
    GROK_DEFAULT_MODEL: str = "grok-beta"
    DEEPSEEK_DEFAULT_MODEL: str = "deepseek-chat"
    LLAMA_DEFAULT_MODEL: str = "llama-3.3-70b"  # Meta Llama via API (Together AI, Groq, or Meta API)

    # Cost tracking (approximate USD per 1K tokens)
    # WARNING: Costs are organized into VERIFIED (confirmed pricing) and UNVERIFIED
    # (speculative/estimated pricing). Unverified costs may be inaccurate.
    MODEL_COSTS = {
        # ======================================================================
        # VERIFIED PRICING - Confirmed from official pricing pages
        # ======================================================================

        # OpenAI - GPT-4o Series - VERIFIED
        "gpt-4o": {"input": 0.0025, "output": 0.01},  # $2.50/1M input
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # $0.15/1M input, $0.60/1M output

        # OpenAI - GPT-4 and GPT-3.5 - VERIFIED
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},

        # Anthropic - Claude 3.5 Series - VERIFIED
        "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},  # $0.80/1M input, $4/1M output
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},  # $3/1M input, $15/1M output

        # Anthropic - Claude 3 Series - VERIFIED
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},  # $0.25/1M input, $1.25/1M output
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},  # Alias for Claude 3 Haiku
        "claude-3-opus": {"input": 0.015, "output": 0.075},

        # Google Gemini 2.0 - VERIFIED
        "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},  # $0.10/1M input, $0.40/1M output
        "gemini-2.0-flash-lite": {"input": 0.000075, "output": 0.0003},  # $0.075/1M input, $0.30/1M output

        # DeepSeek - VERIFIED
        "deepseek-chat": {"input": 0.00027, "output": 0.0011},  # Very cost-effective
        "deepseek-reasoner": {"input": 0.00055, "output": 0.00219},  # Reasoning mode

        # Meta Llama 3.3 Series - VERIFIED (Groq pricing)
        "llama-3.3-70b": {"input": 0.00059, "output": 0.00079},  # $0.59/1M input, $0.79/1M output
        "llama-3.3-70b-specdec": {"input": 0.00059, "output": 0.00099},  # $0.59/1M input, $0.99/1M output

        # Meta Llama 3.1 Series - VERIFIED
        "llama-3.1-405b": {"input": 0.0005, "output": 0.0015},
        "llama-3.1-70b": {"input": 0.00059, "output": 0.00079},
        "llama-3.1-8b": {"input": 0.0001, "output": 0.0001},  # Together AI Lite pricing

        # Meta Llama 3 Series - VERIFIED
        "llama-3-70b": {"input": 0.00059, "output": 0.00079},
        "llama-3-8b": {"input": 0.0001, "output": 0.0001},  # $0.10/1M (Together AI Lite)

        # ======================================================================
        # UNVERIFIED PRICING - Speculative/estimated; may be inaccurate
        # These models may not exist or pricing may differ from estimates
        # ======================================================================

        # OpenAI - GPT-5 Series (UNVERIFIED - models may not exist)
        "gpt-5": {"input": 0.00125, "output": 0.01},  # ESTIMATED
        "gpt-5-mini": {"input": 0.00025, "output": 0.002},  # ESTIMATED
        "gpt-5-nano": {"input": 0.00005, "output": 0.0004},  # ESTIMATED

        # OpenAI - GPT-4.1 Series (UNVERIFIED - models may not exist)
        "gpt-4.1": {"input": 0.002, "output": 0.008},  # ESTIMATED
        "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},  # ESTIMATED
        "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},  # ESTIMATED

        # Anthropic - Claude 4.5 Series (UNVERIFIED - check availability)
        "claude-sonnet-4-5-20250929": {"input": 0.003, "output": 0.015},  # ESTIMATED
        "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},  # ESTIMATED
        "claude-haiku-4-5-20251001": {"input": 0.001, "output": 0.005},  # ESTIMATED
        "claude-haiku-4-5": {"input": 0.001, "output": 0.005},  # ESTIMATED

        # Anthropic - Claude 4.1 Series (UNVERIFIED - check availability)
        "claude-opus-4-1-20250805": {"input": 0.015, "output": 0.075},  # ESTIMATED
        "claude-opus-4-1": {"input": 0.015, "output": 0.075},  # ESTIMATED

        # Anthropic - Claude 4 Series (UNVERIFIED - check availability)
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},  # ESTIMATED
        "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},  # ESTIMATED

        # Anthropic - Claude 3.7 Series (UNVERIFIED - check availability)
        "claude-3-7-sonnet-20250219": {"input": 0.003, "output": 0.015},  # ESTIMATED

        # Google Gemini 2.5 Series (UNVERIFIED - check availability)
        "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},  # ESTIMATED
        "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},  # ESTIMATED
        "gemini-2.5-flash-lite": {"input": 0.0001, "output": 0.0004},  # ESTIMATED

        # xAI Grok (UNVERIFIED - pricing may differ)
        "grok-beta": {"input": 0.003, "output": 0.015},  # ESTIMATED
        "grok-2": {"input": 0.003, "output": 0.015},  # ESTIMATED
        "grok-3": {"input": 0.003, "output": 0.015},  # ESTIMATED
        "grok-4": {"input": 0.003, "output": 0.015},  # ESTIMATED - model may not exist
        "grok-4-fast": {"input": 0.0002, "output": 0.0005},  # ESTIMATED - model may not exist

        # Meta Llama 4 Series (UNVERIFIED - check availability)
        "llama-4-maverick": {"input": 0.00027, "output": 0.00085},  # ESTIMATED
        "llama-4-scout": {"input": 0.00018, "output": 0.00059},  # ESTIMATED
    }


@dataclass
class OrchestratorConfig:
    """Orchestrator configuration."""

    # Logging
    VERBOSE_DEFAULT: bool = True
    LOG_TO_FILE: bool = False
    LOG_FILE_PATH: str = "game_logs/"

    # Game result storage
    SAVE_RESULTS: bool = True
    RESULTS_PATH: str = "game_results/"
    RESULT_FORMAT: str = "json"  # json, csv, or both

    # Timing
    TURN_DELAY: float = 0.0  # Delay between turns (for demos)
    GUESS_DELAY: float = 0.0  # Delay between guesses (for demos)

    # Validation
    STRICT_VALIDATION: bool = True  # Strict hint/guess validation
    ALLOW_INVALID_HINTS: bool = False  # Allow hints that are board words

    # Tournament settings
    GAMES_PER_MATCH: int = 1  # Number of games per agent pairing
    RANDOMIZE_STARTING_TEAM: bool = False  # Randomize who starts each game


@dataclass
class DataConfig:
    """Data and file configuration."""

    # Word list
    WORD_LIST_PATH: str = "utils/words.csv"
    MIN_WORDS_IN_POOL: int = 100  # Minimum words needed in word pool

    # Default demo settings
    DEMO_RANDOM_SEED: Optional[int] = None  # Set for reproducible demos

    # File paths
    ENV_FILE: str = ".env"
    ENV_EXAMPLE_FILE: str = ".env.example"

    # Cache settings
    CACHE_WORD_LIST: bool = True  # Cache word list in memory


class Config:
    """Main configuration class combining all config sections."""

    def __init__(self):
        self.game = GameConfig()
        self.llm = LLMConfig()
        self.orchestrator = OrchestratorConfig()
        self.data = DataConfig()

        # Validate on initialization
        self.validate()

    def validate(self):
        """Validate all configuration sections."""
        self.game.validate()
        return True

    @classmethod
    def default(cls) -> "Config":
        """Get default configuration."""
        return cls()

    @classmethod
    def custom_game(cls, board_size: int = 25, **kwargs) -> "Config":
        """Create config with custom game settings."""
        config = cls()
        config.game = GameConfig.custom(board_size, **kwargs)
        return config

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "game": {
                "board_size": self.game.BOARD_SIZE,
                "blue_words": self.game.BLUE_WORDS,
                "red_words": self.game.RED_WORDS,
                "neutral_words": self.game.NEUTRAL_WORDS,
                "bomb_count": self.game.BOMB_COUNT,
                "max_turns": self.game.MAX_TURNS,
                "starting_team": self.game.STARTING_TEAM,
            },
            "llm": {
                "default_temperature": self.llm.DEFAULT_TEMPERATURE,
                "max_tokens": self.llm.MAX_TOKENS,
                "timeout": self.llm.RESPONSE_TIMEOUT,
            },
            "orchestrator": {
                "verbose": self.orchestrator.VERBOSE_DEFAULT,
                "save_results": self.orchestrator.SAVE_RESULTS,
                "strict_validation": self.orchestrator.STRICT_VALIDATION,
            }
        }


# Global default configuration instance
default_config = Config.default()


# Convenience imports for backward compatibility
BOARD_SIZE = default_config.game.BOARD_SIZE
BLUE_WORDS = default_config.game.BLUE_WORDS
RED_WORDS = default_config.game.RED_WORDS
NEUTRAL_WORDS = default_config.game.NEUTRAL_WORDS
BOMB_COUNT = default_config.game.BOMB_COUNT
MAX_TURNS = default_config.game.MAX_TURNS