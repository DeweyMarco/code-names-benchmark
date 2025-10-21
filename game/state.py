"""
Game state management for Codenames.
"""
from enum import Enum
from typing import List, Set, Optional, Tuple
from dataclasses import dataclass, field

from .board import Board, CardColor


class Team(Enum):
    """Represents the two teams."""
    RED = "red"
    BLUE = "blue"


class GameOutcome(Enum):
    """Possible game outcomes."""
    RED_WIN = "red_win"
    BLUE_WIN = "blue_win"
    IN_PROGRESS = "in_progress"


@dataclass
class TurnResult:
    """Result of a single guess during a turn."""
    word: str
    color: CardColor
    correct: bool
    hit_bomb: bool = False
    
    def __str__(self):
        result = "💣 BOMB!" if self.hit_bomb else ("✓" if self.correct else "✗")
        return f"{self.word} ({self.color.value}) {result}"


@dataclass
class Turn:
    """Represents a complete turn (hint + guesses)."""
    team: Team
    turn_number: int
    hint_word: str
    hint_count: int
    guesses: List[TurnResult] = field(default_factory=list)
    
    def __str__(self):
        guess_str = ", ".join(str(g) for g in self.guesses)
        return f"Turn {self.turn_number} ({self.team.value}): '{self.hint_word}' ({self.hint_count}) → [{guess_str}]"


class GameState:
    """
    Manages the state of a Codenames game.
    
    Tracks revealed words, current turn, game history, and win conditions.
    """
    
    def __init__(self, board: Board):
        """Initialize game state with starting team from board config."""
        self._board = board
        self._revealed_words: Set[str] = set()
        self._turn_history: List[Turn] = []

        # Use starting team from board's config
        starting_team_name = board.config.STARTING_TEAM
        self._current_team = Team.BLUE if starting_team_name == "BLUE" else Team.RED

        self._game_outcome = GameOutcome.IN_PROGRESS
        self._current_turn: Optional[Turn] = None
        self._turn_number = 0
    
    @property
    def board(self) -> Board:
        """Get the game board."""
        return self._board
    
    @property
    def current_team(self) -> Team:
        """Get the team whose turn it is."""
        return self._current_team
    
    @property
    def revealed_words(self) -> Set[str]:
        """Get set of revealed words (immutable view)."""
        return self._revealed_words.copy()
    
    @property
    def unrevealed_words(self) -> List[str]:
        """Get list of unrevealed words."""
        return [word for word in self._board.all_words if word not in self._revealed_words]
    
    @property
    def turn_history(self) -> List[Turn]:
        """Get history of all turns."""
        return self._turn_history.copy()
    
    @property
    def game_outcome(self) -> GameOutcome:
        """Get current game outcome."""
        return self._game_outcome
    
    @property
    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self._game_outcome != GameOutcome.IN_PROGRESS
    
    @property
    def turn_number(self) -> int:
        """Get current turn number."""
        return self._turn_number
    
    def start_turn(self, hint_word: str, hint_count: int):
        """Start a new turn with a hint from the spymaster."""
        if self.is_game_over:
            raise ValueError("Cannot start turn: game is over")
        
        if self._current_turn is not None:
            raise ValueError("Cannot start turn: current turn not finished")
        
        self._turn_number += 1
        self._current_turn = Turn(
            team=self._current_team,
            turn_number=self._turn_number,
            hint_word=hint_word.lower(),
            hint_count=hint_count
        )
    
    def make_guess(self, word: str) -> TurnResult:
        """Make a guess for the current team."""
        if self._current_turn is None:
            raise ValueError("Cannot make guess: no turn in progress")
        
        word = word.lower()
        
        if word not in self._board.all_words:
            raise ValueError(f"Word '{word}' not on board")
        
        if word in self._revealed_words:
            raise ValueError(f"Word '{word}' already revealed")
        
        # Reveal word and determine result
        card_color = self._board.get_color(word)
        self._revealed_words.add(word)
        
        team_color = CardColor.BLUE if self._current_team == Team.BLUE else CardColor.RED
        correct = card_color == team_color
        hit_bomb = card_color == CardColor.BOMB
        
        result = TurnResult(
            word=word,
            color=card_color,
            correct=correct,
            hit_bomb=hit_bomb
        )
        
        self._current_turn.guesses.append(result)
        self._check_game_outcome(result)
        
        return result
    
    def end_turn(self):
        """End the current turn and switch teams."""
        if self._current_turn is None:
            raise ValueError("Cannot end turn: no turn in progress")
        
        self._turn_history.append(self._current_turn)
        self._current_turn = None
        
        if not self.is_game_over:
            self._current_team = Team.RED if self._current_team == Team.BLUE else Team.BLUE
    
    def _check_game_outcome(self, last_result: TurnResult):
        """Check if the game has ended and update outcome."""
        if last_result.hit_bomb:
            # Bomb ends game immediately, other team wins
            if self._current_team == Team.RED:
                self._game_outcome = GameOutcome.BLUE_WIN
            else:
                self._game_outcome = GameOutcome.RED_WIN
            return
        
        # Check if either team has revealed all their words
        blue_words = set(self._board.get_words_by_color(CardColor.BLUE))
        red_words = set(self._board.get_words_by_color(CardColor.RED))
        
        blue_remaining = blue_words - self._revealed_words
        red_remaining = red_words - self._revealed_words
        
        if not blue_remaining:
            self._game_outcome = GameOutcome.BLUE_WIN
        elif not red_remaining:
            self._game_outcome = GameOutcome.RED_WIN
    
    def get_remaining_words(self, team: Team) -> List[str]:
        """Get remaining unrevealed words for a team."""
        color = CardColor.BLUE if team == Team.BLUE else CardColor.RED
        team_words = set(self._board.get_words_by_color(color))
        return list(team_words - self._revealed_words)
    
    def get_team_scores(self) -> Tuple[int, int]:
        """Get remaining word counts: (blue_remaining, red_remaining)."""
        blue_remaining = len(self.get_remaining_words(Team.BLUE))
        red_remaining = len(self.get_remaining_words(Team.RED))
        return (blue_remaining, red_remaining)
    
    def print_status(self):
        """Print current game status."""
        blue_rem, red_rem = self.get_team_scores()
        
        print("\n" + "=" * 60)
        print(f"GAME STATUS - Turn {self._turn_number}")
        print("=" * 60)
        print(f"Current Team: {self._current_team.value.upper()}")
        print(f"Blue Remaining: {blue_rem} | Red Remaining: {red_rem}")
        print(f"Outcome: {self._game_outcome.value}")
        print(f"Revealed: {len(self._revealed_words)}/25 words")
        
        if self._current_turn:
            print(f"\nCurrent Turn: '{self._current_turn.hint_word}' ({self._current_turn.hint_count})")
            if self._current_turn.guesses:
                print("Guesses this turn:")
                for guess in self._current_turn.guesses:
                    print(f"  - {guess}")
        
        print("=" * 60)
    
    def get_snapshot(self) -> dict:
        """Get JSON-serializable snapshot of game state."""
        return {
            'turn_number': self._turn_number,
            'current_team': self._current_team.value,
            'revealed_words': list(self._revealed_words),
            'game_outcome': self._game_outcome.value,
            'team_scores': self.get_team_scores(),
            'turn_history': [
                {
                    'team': turn.team.value,
                    'turn_number': turn.turn_number,
                    'hint_word': turn.hint_word,
                    'hint_count': turn.hint_count,
                    'guesses': [
                        {
                            'word': g.word,
                            'color': g.color.value,
                            'correct': g.correct,
                            'hit_bomb': g.hit_bomb
                        }
                        for g in turn.guesses
                    ]
                }
                for turn in self._turn_history
            ]
        }
    
    def __repr__(self):
        blue_rem, red_rem = self.get_team_scores()
        return (f"GameState(turn={self._turn_number}, team={self._current_team.value}, "
                f"blue_remaining={blue_rem}, red_remaining={red_rem}, "
                f"outcome={self._game_outcome.value})")

