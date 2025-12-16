"""
Abstract base classes for Codenames agents.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
from dataclasses import dataclass

from game import Team, CardColor


@dataclass
class HintResponse:
    """Response from a hint giver."""
    word: str
    count: int
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validate the hint response.
        
        Returns:
            (is_valid, error_message)
        """
        if not self.word or not isinstance(self.word, str):
            return False, "Hint word must be a non-empty string"
        
        if not isinstance(self.count, int) or self.count < 1:
            return False, "Hint count must be a positive integer"
        
        # Check for multi-word hints
        if ' ' in self.word.strip():
            return False, "Hint must be a single word (no spaces)"
        
        return True, ""


class HintGiver(ABC):
    """
    Abstract base class for hint giver (spymaster) agents.
    
    The hint giver sees all word colors and must give a one-word hint
    plus a number indicating how many words relate to that hint.
    """
    
    def __init__(self, team: Team):
        """
        Initialize hint giver for a specific team.
        
        Args:
            team: The team this agent represents
        """
        self.team = team
    
    @abstractmethod
    def give_hint(
        self,
        my_words: List[str],
        opponent_words: List[str],
        neutral_words: List[str],
        bomb_words: List[str],
        revealed_words: List[str],
        board_words: List[str]
    ) -> HintResponse:
        """
        Generate a hint for the team.

        Args:
            my_words: List of unrevealed words belonging to this agent's team
            opponent_words: List of unrevealed opponent words
            neutral_words: List of unrevealed neutral words
            bomb_words: List of bomb words (if not revealed)
            revealed_words: List of already revealed words
            board_words: All words on the board (for reference)

        Returns:
            HintResponse with hint word and count

        Notes:
            - Hint word must be a single word (no spaces)
            - Hint word should not be any word currently on the board
            - Count indicates how many of your team's words relate to the hint
            - Avoid hints that could lead to bomb or opponent words
        """
        pass


class Guesser(ABC):
    """
    Abstract base class for guesser (field operative) agents.
    
    The guesser only sees the words on the board (not colors) and must
    guess words based on the hint from their team's hint giver.
    """
    
    def __init__(self, team: Team):
        """
        Initialize guesser for a specific team.
        
        Args:
            team: The team this agent represents
        """
        self.team = team
    
    @abstractmethod
    def make_guesses(
        self,
        hint_word: str,
        hint_count: int,
        board_words: List[str],
        revealed_words: List[str]
    ) -> List[str]:
        """
        Make guesses based on the hint.
        
        Args:
            hint_word: The hint word given by the hint giver
            hint_count: Number of words the hint relates to
            board_words: All words on the board
            revealed_words: List of already revealed words
        
        Returns:
            List of words to guess (in order of preference)
        
        Notes:
            - Can return 1 to (hint_count + 1) guesses
            - Standard strategy: guess up to hint_count words
            - Extra guess allowed (hint_count + 1) for previous hints
            - Guessing stops when wrong word is revealed
            - Must only guess from unrevealed words
            - Should return empty list to pass turn
        """
        pass
    
    @abstractmethod
    def process_result(self, guessed_word: str, was_correct: bool, color: CardColor):
        """
        Receive feedback on a guess (optional for learning/adjustment).

        Args:
            guessed_word: The word that was guessed
            was_correct: Whether it was the team's word
            color: The actual color of the word

        Notes:
            - This is called after each guess
            - Agents can use this to adjust their strategy
            - Not required to do anything (default implementation can pass)
        """
        pass

    def reset(self):
        """
        Reset agent state between games.

        Subclasses should override this to clear any accumulated state
        (e.g., guess history) when agents are reused across multiple games.
        """
        pass

