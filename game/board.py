"""
Board representation for Codenames game.
Simple word list with color assignments - optimized for LLM agents.
"""
from enum import Enum
from typing import List, Dict, Optional
import random

from config import GameConfig


class CardColor(Enum):
    """Represents the color/type of a card."""
    RED = "red"
    BLUE = "blue"
    NEUTRAL = "neutral"
    BOMB = "bomb"


class Board:
    """
    Simple Codenames board - just words and their colors.

    Uses configuration from GameConfig for word distribution.
    """

    def __init__(self, words: List[str], config: Optional[GameConfig] = None):
        """
        Initialize a new game board.

        Args:
            words: List of words for the board
            config: Game configuration (uses default if None)

        Raises:
            ValueError: If word count doesn't match config.BOARD_SIZE
        """
        self.config = config or GameConfig()

        if len(words) != self.config.BOARD_SIZE:
            raise ValueError(
                f"Board requires exactly {self.config.BOARD_SIZE} words, got {len(words)}"
            )

        self._word_colors: Dict[str, CardColor] = {}
        self._initialize_board(words)

    def _initialize_board(self, words: List[str]):
        """Assign colors to words based on configuration."""
        colors = (
            [CardColor.BLUE] * self.config.BLUE_WORDS +
            [CardColor.RED] * self.config.RED_WORDS +
            [CardColor.NEUTRAL] * self.config.NEUTRAL_WORDS +
            [CardColor.BOMB] * self.config.BOMB_COUNT
        )
        
        random.shuffle(colors)
        
        for word, color in zip(words, colors):
            self._word_colors[word.lower()] = color
    
    @property
    def all_words(self) -> List[str]:
        """Get all words on the board."""
        return list(self._word_colors.keys())
    
    def get_color(self, word: str) -> Optional[CardColor]:
        """
        Get the color of a word.
        
        Args:
            word: The word to look up (case-insensitive)
        
        Returns:
            CardColor if found, None otherwise
        """
        return self._word_colors.get(word.lower())
    
    def get_words_by_color(self, color: CardColor) -> List[str]:
        """
        Get all words of a specific color.
        
        Args:
            color: The color to filter by
        
        Returns:
            List of words with that color
        """
        return [word for word, c in self._word_colors.items() if c == color]
    
    def get_color_counts(self) -> Dict[CardColor, int]:
        """Get count of cards by color."""
        counts = {color: 0 for color in CardColor}
        for color in self._word_colors.values():
            counts[color] += 1
        return counts
    
    def __repr__(self):
        counts = self.get_color_counts()
        return (f"Board(blue={counts[CardColor.BLUE]}, red={counts[CardColor.RED]}, "
                f"neutral={counts[CardColor.NEUTRAL]}, bomb={counts[CardColor.BOMB]})")

