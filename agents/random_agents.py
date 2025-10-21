"""
Simple random agents for testing the game flow.
"""
import random
from typing import List

from game import Team, CardColor
from .base import HintGiver, Guesser, HintResponse


class RandomHintGiver(HintGiver):
    """Gives random hints for testing."""
    
    def give_hint(
        self,
        my_words: List[str],
        opponent_words: List[str],
        neutral_words: List[str],
        bomb_word: str,
        revealed_words: List[str],
        board_words: List[str]
    ) -> HintResponse:
        """Give a random hint."""
        if not my_words:
            return HintResponse(word="pass", count=0)
        
        # Pick a random word from our team as the "hint"
        hint_word = f"hint_{random.randint(1, 100)}"
        count = min(len(my_words), random.randint(1, 3))
        
        return HintResponse(word=hint_word, count=count)


class RandomGuesser(Guesser):
    """Makes random guesses for testing."""
    
    def make_guesses(
        self,
        hint_word: str,
        hint_count: int,
        board_words: List[str],
        revealed_words: List[str]
    ) -> List[str]:
        """Make random guesses."""
        unrevealed = [w for w in board_words if w not in revealed_words]
        
        if not unrevealed:
            return []
        
        # Guess up to hint_count words randomly
        num_guesses = min(hint_count, len(unrevealed))
        guesses = random.sample(unrevealed, num_guesses)
        
        return guesses
    
    def process_result(self, guessed_word: str, was_correct: bool, color: CardColor):
        """Random agent doesn't learn from results."""
        pass

