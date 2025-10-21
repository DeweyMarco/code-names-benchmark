"""
Game orchestrator that coordinates 4 agents through a complete Codenames game.
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from game import Board, GameState, Team, GameOutcome, CardColor
from agents import HintGiver, Guesser
from config import OrchestratorConfig, GameConfig


@dataclass
class GameResult:
    """Complete result of a game for analysis."""
    game_id: str
    outcome: GameOutcome
    winner: Optional[Team]
    total_turns: int
    final_scores: tuple  # (blue_remaining, red_remaining)
    snapshot: Dict[str, Any]
    blue_hint_giver_name: str
    blue_guesser_name: str
    red_hint_giver_name: str
    red_guesser_name: str
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage."""
        return {
            'game_id': self.game_id,
            'outcome': self.outcome.value if self.outcome else None,
            'winner': self.winner.value if self.winner else None,
            'total_turns': self.total_turns,
            'final_scores': {
                'blue_remaining': self.final_scores[0],
                'red_remaining': self.final_scores[1]
            },
            'agents': {
                'blue_hint_giver': self.blue_hint_giver_name,
                'blue_guesser': self.blue_guesser_name,
                'red_hint_giver': self.red_hint_giver_name,
                'red_guesser': self.red_guesser_name
            },
            'snapshot': self.snapshot,
            'error': self.error,
            'timestamp': self.timestamp.isoformat()
        }


class GameRunner:
    """
    Orchestrates a complete Codenames game with 4 agents.
    
    Handles the full game loop:
    1. Get hint from current team's hint giver
    2. Validate hint
    3. Get guesses from current team's guesser
    4. Execute guesses and provide feedback
    5. Switch teams
    6. Repeat until game over
    """
    
    def __init__(
        self,
        board: Board,
        blue_hint_giver: HintGiver,
        blue_guesser: Guesser,
        red_hint_giver: HintGiver,
        red_guesser: Guesser,
        max_turns: Optional[int] = None,
        verbose: Optional[bool] = None,
        game_id: Optional[str] = None,
        config: Optional[OrchestratorConfig] = None
    ):
        """
        Initialize game runner.

        Args:
            board: The game board
            blue_hint_giver: Blue team's hint giver
            blue_guesser: Blue team's guesser
            red_hint_giver: Red team's hint giver
            red_guesser: Red team's guesser
            max_turns: Maximum turns before declaring draw (uses config default if None)
            verbose: Print game progress (uses config default if None)
            game_id: Optional game identifier
            config: Orchestrator configuration (uses default if None)
        """
        self.board = board
        self.game = GameState(board)
        
        self.agents = {
            Team.BLUE: {
                'hint_giver': blue_hint_giver,
                'guesser': blue_guesser
            },
            Team.RED: {
                'hint_giver': red_hint_giver,
                'guesser': red_guesser
            }
        }

        # Use config for defaults
        self.config = config or OrchestratorConfig()
        self.game_config = GameConfig()  # Get game config for max_turns default

        self.max_turns = max_turns if max_turns is not None else self.game_config.MAX_TURNS
        self.verbose = verbose if verbose is not None else self.config.VERBOSE_DEFAULT
        self.game_id = game_id or f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _log(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)
    
    def _get_hint_from_agent(self, team: Team) -> tuple:
        """
        Get hint from team's hint giver.
        
        Returns:
            (hint_response, error_message)
        """
        hint_giver = self.agents[team]['hint_giver']
        
        # Prepare info for hint giver
        my_words = self.game.get_remaining_words(team)
        opponent_team = Team.RED if team == Team.BLUE else Team.BLUE
        opponent_words = self.game.get_remaining_words(opponent_team)
        
        neutral_words = [
            w for w in self.board.get_words_by_color(CardColor.NEUTRAL)
            if w not in self.game.revealed_words
        ]
        
        bomb_words = [
            w for w in self.board.get_words_by_color(CardColor.BOMB)
            if w not in self.game.revealed_words
        ]
        bomb_word = bomb_words[0] if bomb_words else ""
        
        try:
            hint = hint_giver.give_hint(
                my_words=my_words,
                opponent_words=opponent_words,
                neutral_words=neutral_words,
                bomb_word=bomb_word,
                revealed_words=list(self.game.revealed_words),
                board_words=self.board.all_words
            )
            
            # Validate hint
            is_valid, error = hint.validate()
            if not is_valid:
                return None, f"Invalid hint: {error}"
            
            # Check hint word not on board
            if hint.word.lower() in [w.lower() for w in self.board.all_words]:
                return None, f"Invalid hint: '{hint.word}' is on the board"
            
            return hint, None
            
        except Exception as e:
            return None, f"Hint giver error: {str(e)}"
    
    def _get_guesses_from_agent(self, team: Team, hint_word: str, hint_count: int) -> tuple:
        """
        Get guesses from team's guesser.
        
        Returns:
            (guesses_list, error_message)
        """
        guesser = self.agents[team]['guesser']
        
        try:
            guesses = guesser.make_guesses(
                hint_word=hint_word,
                hint_count=hint_count,
                board_words=self.board.all_words,
                revealed_words=list(self.game.revealed_words)
            )
            
            # Validate guesses
            if not isinstance(guesses, list):
                return None, "Guesses must be a list"
            
            return guesses, None
            
        except Exception as e:
            return None, f"Guesser error: {str(e)}"
    
    def _execute_turn(self, team: Team) -> Optional[str]:
        """
        Execute a complete turn for a team.
        
        Returns:
            Error message if any, None if successful
        """
        self._log(f"\n{'='*60}")
        self._log(f"TURN {self.game.turn_number + 1}: {team.value.upper()} TEAM")
        self._log(f"{'='*60}")
        
        # 1. Get hint from hint giver
        hint, error = self._get_hint_from_agent(team)
        if error:
            return error
        
        self._log(f"Hint: '{hint.word}' ({hint.count})")
        
        # 2. Start turn with hint
        try:
            self.game.start_turn(hint.word, hint.count)
        except ValueError as e:
            return f"Failed to start turn: {str(e)}"
        
        # 3. Get guesses from guesser
        guesses, error = self._get_guesses_from_agent(team, hint.word, hint.count)
        if error:
            self.game.end_turn()  # Clean up
            return error
        
        if not guesses:
            self._log("   Team passes (no guesses)")
            self.game.end_turn()
            return None
        
        self._log(f"Guesses: {guesses}")
        
        # 4. Execute guesses
        for guess_word in guesses:
            try:
                result = self.game.make_guess(guess_word)
                self._log(f"   → {result}")
                
                # Give feedback to guesser
                guesser = self.agents[team]['guesser']
                guesser.process_result(guess_word, result.correct, result.color)
                
                # Check if should stop
                if result.hit_bomb:
                    self._log("   BOMB HIT! Game over.")
                    break
                elif not result.correct:
                    self._log("   ✗ Wrong. Turn ends.")
                    break
                    
            except ValueError as e:
                self._log(f"   ✗ Invalid guess '{guess_word}': {e}")
                continue
        
        # 5. End turn
        self.game.end_turn()
        return None
    
    def run(self) -> GameResult:
        """
        Run the complete game until completion.
        
        Returns:
            GameResult with complete game data
        """
        self._log(f"\n🎮 STARTING GAME: {self.game_id}")
        self._log(f"{'='*60}")
        self._log(f"Blue: {self.agents[Team.BLUE]['hint_giver'].__class__.__name__} + "
                  f"{self.agents[Team.BLUE]['guesser'].__class__.__name__}")
        self._log(f"Red:  {self.agents[Team.RED]['hint_giver'].__class__.__name__} + "
                  f"{self.agents[Team.RED]['guesser'].__class__.__name__}")
        
        error_message = None
        
        # Main game loop
        while not self.game.is_game_over and self.game.turn_number < self.max_turns:
            current_team = self.game.current_team
            
            error = self._execute_turn(current_team)
            if error:
                error_message = error
                self._log(f"\nError: {error}")
                break
        
        # Check for max turns reached
        if self.game.turn_number >= self.max_turns and not self.game.is_game_over:
            self._log(f"\nMax turns ({self.max_turns}) reached. Game incomplete.")
        
        # Determine winner
        outcome = self.game.game_outcome
        winner = None
        if outcome == GameOutcome.BLUE_WIN:
            winner = Team.BLUE
        elif outcome == GameOutcome.RED_WIN:
            winner = Team.RED
        
        # Create result
        result = GameResult(
            game_id=self.game_id,
            outcome=outcome,
            winner=winner,
            total_turns=self.game.turn_number,
            final_scores=self.game.get_team_scores(),
            snapshot=self.game.get_snapshot(),
            blue_hint_giver_name=self.agents[Team.BLUE]['hint_giver'].__class__.__name__,
            blue_guesser_name=self.agents[Team.BLUE]['guesser'].__class__.__name__,
            red_hint_giver_name=self.agents[Team.RED]['hint_giver'].__class__.__name__,
            red_guesser_name=self.agents[Team.RED]['guesser'].__class__.__name__,
            error=error_message
        )
        
        self._log(f"\n{'='*60}")
        self._log(f"GAME COMPLETE")
        self._log(f"{'='*60}")
        self._log(f"Outcome: {outcome.value}")
        self._log(f"Winner: {winner.value if winner else 'None'}")
        self._log(f"Total turns: {self.game.turn_number}")
        self._log(f"Final scores - Blue: {result.final_scores[0]}, Red: {result.final_scores[1]}")
        
        return result

