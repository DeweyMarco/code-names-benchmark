"""
Game orchestrator that coordinates 4 agents through a complete Codenames game.
"""
import logging
import time
import random
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

from game import Board, GameState, Team, GameOutcome, CardColor
from agents import HintGiver, Guesser, HintResponse
from config import OrchestratorConfig, GameConfig, LLMConfig


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
    # Model names for tracking which LLM played each role
    blue_hint_giver_model: Optional[str] = None
    blue_guesser_model: Optional[str] = None
    red_hint_giver_model: Optional[str] = None
    red_guesser_model: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now())
    
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
            'models': {
                'blue_hint_giver': self.blue_hint_giver_model,
                'blue_guesser': self.blue_guesser_model,
                'red_hint_giver': self.red_hint_giver_model,
                'red_guesser': self.red_guesser_model
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
        config: Optional[OrchestratorConfig] = None,
        llm_config: Optional[LLMConfig] = None
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
            llm_config: LLM configuration for retry settings (uses default if None)
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
        self.game_config = board.config if hasattr(board, "config") else GameConfig()
        self.llm_config = llm_config or LLMConfig()

        self.max_turns = max_turns if max_turns is not None else self.game_config.MAX_TURNS
        self.verbose = verbose if verbose is not None else self.config.VERBOSE_DEFAULT
        self.game_id = game_id or f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _log(self, message: str):
        """Log message if verbose mode is on."""
        if self.verbose:
            logger.info(message)

    @staticmethod
    def _sanitize_for_log(text: str, max_length: int = 100) -> str:
        """
        Sanitize text for safe logging.

        Removes control characters and limits length to prevent log injection
        or flooding from potentially malicious LLM outputs.

        Args:
            text: The text to sanitize
            max_length: Maximum allowed length (default 100)

        Returns:
            Sanitized text safe for logging
        """
        if not isinstance(text, str):
            text = str(text)
        # Remove control characters (except space) and limit length
        sanitized = ''.join(c if c.isprintable() else '?' for c in text)
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."
        return sanitized

    def _is_valid_hint(self, hint_word: str, board_words: List[str]) -> Tuple[bool, str]:
        """
        Validate hint against board words per official Codenames rules.

        A hint is invalid if:
        - It exactly matches a board word
        - It contains a board word as a substring
        - It is a substring of a board word

        Args:
            hint_word: The proposed hint word
            board_words: List of all words on the board

        Returns:
            (is_valid, error_message) tuple
        """
        hint_lower = hint_word.lower()

        for board_word in board_words:
            board_lower = board_word.lower()

            # Check exact match
            if hint_lower == board_lower:
                return False, f"'{hint_word}' is on the board"

            # Check if hint contains board word
            if board_lower in hint_lower:
                return False, f"'{hint_word}' contains board word '{board_word}'"

            # Check if board word contains hint
            if hint_lower in board_lower:
                return False, f"Board word '{board_word}' contains '{hint_word}'"

        return True, ""

    def _get_hint_from_agent(self, team: Team) -> Tuple[Optional[HintResponse], Optional[str]]:
        """
        Get hint from team's hint giver.

        Returns:
            Tuple of (hint_response, error_message). One will be None.
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

        try:
            hint = hint_giver.give_hint(
                my_words=my_words,
                opponent_words=opponent_words,
                neutral_words=neutral_words,
                bomb_words=bomb_words,
                revealed_words=list(self.game.revealed_words),
                board_words=self.board.all_words
            )
            
            # Validate hint
            is_valid, error = hint.validate()
            if not is_valid:
                return None, f"Invalid hint: {error}"

            # Check hint word against board words (exact match and substrings)
            is_valid, error = self._is_valid_hint(hint.word, self.board.all_words)
            if not is_valid:
                return None, f"Invalid hint: {error}"

            return hint, None
            
        except Exception as e:
            # Check if this is a retryable error
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in [
                '503', 'service unavailable', 'overloaded', 'try again later',
                'rate limit', 'timeout', 'connection', 'network'
            ]):
                # This is a retryable error - let the retry logic handle it
                raise e
            else:
                # This is a non-retryable error
                return None, f"Hint giver error: {str(e)}"
    
    def _clean_guesses(self, guesses: List[str], max_guesses: int) -> List[str]:
        """
        Validate and clean a list of guesses.

        Removes non-strings, duplicates, and enforces max guess limit.

        Args:
            guesses: Raw list of guesses from the guesser
            max_guesses: Maximum number of guesses allowed

        Returns:
            Cleaned list of valid guesses
        """
        cleaned_guesses = []
        seen = set()
        for guess in guesses:
            if not isinstance(guess, str):
                self._log(f"   Skipping non-string guess: {self._sanitize_for_log(str(guess))}")
                self.game.record_invalid_guess(str(guess), "non_string_type")
                continue
            guess_lower = guess.lower()
            if guess_lower in seen:
                continue
            seen.add(guess_lower)
            cleaned_guesses.append(guess)
            if len(cleaned_guesses) >= max_guesses:
                break
        return cleaned_guesses

    def _execute_guesses(self, team: Team, guesses: List[str]) -> None:
        """
        Execute a list of guesses for a team.

        Processes each guess, updates game state, and provides feedback.
        Stops on bomb hit, wrong guess, or invalid guess.

        Args:
            team: The team making the guesses
            guesses: List of validated guesses to execute
        """
        board_words_lower = {w.lower() for w in self.board.all_words}
        revealed_lower = {w.lower() for w in self.game.revealed_words}

        for guess_word in guesses:
            try:
                guess_lower = guess_word.lower()
                if guess_lower not in board_words_lower:
                    self._log(f"   Invalid guess '{self._sanitize_for_log(guess_word)}': word not on board. Turn ends.")
                    self.game.record_invalid_guess(guess_word, "not_on_board")
                    break
                if guess_lower in revealed_lower:
                    self._log(f"   Invalid guess '{self._sanitize_for_log(guess_word)}': word already revealed. Turn ends.")
                    self.game.record_invalid_guess(guess_word, "already_revealed")
                    break

                result = self.game.make_guess(guess_word)
                self._log(f"   → {result}")
                revealed_lower.add(guess_lower)

                # Give feedback to guesser
                guesser = self.agents[team]['guesser']
                guesser.process_result(guess_word, result.correct, result.color)

                # Check if should stop
                if result.hit_bomb:
                    self._log("   BOMB HIT! Game over.")
                    break
                elif not result.correct:
                    self._log("   Wrong. Turn ends.")
                    break

            except ValueError as e:
                self._log(f"   Invalid guess '{self._sanitize_for_log(guess_word)}': {e}")
                self.game.record_invalid_guess(guess_word, "invalid_guess")
                break
            except Exception as e:
                self._log(f"   Unexpected error processing guess '{self._sanitize_for_log(guess_word)}': {e}")
                self.game.record_invalid_guess(guess_word, "unexpected_error")
                break

    def _get_guesses_from_agent(self, team: Team, hint_word: str, hint_count: int) -> Tuple[Optional[List[str]], Optional[str]]:
        """
        Get guesses from team's guesser.

        Returns:
            Tuple of (guesses_list, error_message). One will be None.
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
            # Check if this is a retryable error
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in [
                '503', 'service unavailable', 'overloaded', 'try again later',
                'rate limit', 'timeout', 'connection', 'network'
            ]):
                # This is a retryable error - let the retry logic handle it
                raise e
            else:
                # This is a non-retryable error
                return None, f"Guesser error: {str(e)}"
    
    def _execute_turn(self, team: Team) -> Optional[str]:
        """
        Execute a complete turn for a team with retry logic.

        If an error occurs, uses exponential backoff and retries.
        Does not pass the turn on error - retries until successful or max retries exceeded.

        Args:
            team: The team whose turn it is

        Returns:
            Error message if max retries exceeded, None if successful
        """
        max_retries = self.llm_config.MAX_RETRIES
        base_retry_delay = self.llm_config.RETRY_DELAY
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            if attempt > 0:
                # Exponential backoff with jitter: 5s, 10s, 20s + random(0-2s)
                retry_delay = base_retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
                self._log(f"\nRETRY ATTEMPT {attempt}/{max_retries}")
                self._log(f"Waiting {retry_delay:.1f} seconds before retry...")
                time.sleep(retry_delay)
            
            self._log(f"\n{'='*60}")
            self._log(f"TURN {self.game.turn_number + 1}: {team.value.upper()} TEAM")
            if attempt > 0:
                self._log(f"(Retry attempt {attempt})")
            self._log(f"{'='*60}")
            
            # 1. Get hint from hint giver
            try:
                hint, error = self._get_hint_from_agent(team)
                if error:
                    if attempt < max_retries:
                        self._log(f"Hint error: {error}")
                        next_delay = base_retry_delay * (2 ** attempt) + random.uniform(0, 2)
                        self._log(f"Will retry in {next_delay:.1f} seconds...")
                        continue
                    else:
                        return f"Hint error after {max_retries} retries: {error}"
            except Exception as e:
                if attempt < max_retries:
                    self._log(f"Hint giver exception: {str(e)}")
                    next_delay = base_retry_delay * (2 ** attempt) + random.uniform(0, 2)
                    self._log(f"Will retry in {next_delay:.1f} seconds...")
                    continue
                else:
                    return f"Hint giver exception after {max_retries} retries: {str(e)}"

            # Enforce configured hint count bounds
            min_hint = self.game_config.MIN_HINT_COUNT
            max_hint = self.game_config.MAX_HINT_COUNT
            if hint.count < min_hint or hint.count > max_hint:
                invalid_msg = (f"Hint count {hint.count} outside allowed range "
                               f"[{min_hint}, {max_hint}]")
                if attempt < max_retries:
                    self._log(f"Hint error: {invalid_msg}")
                    next_delay = base_retry_delay * (2 ** attempt) + random.uniform(0, 2)
                    self._log(f"Will retry in {next_delay:.1f} seconds...")
                    continue
                return f"Hint error after {max_retries} retries: {invalid_msg}"

            self._log(f"Hint: '{self._sanitize_for_log(hint.word)}' ({hint.count})")

            # 2. Start turn with hint
            try:
                self.game.start_turn(hint.word, hint.count)
            except ValueError as e:
                if attempt < max_retries:
                    self._log(f"Failed to start turn: {str(e)}")
                    next_delay = base_retry_delay * (2 ** attempt)
                    self._log(f"Will retry in {next_delay} seconds...")
                    continue
                else:
                    return f"Failed to start turn after {max_retries} retries: {str(e)}"
            
            # 3. Get guesses from guesser
            try:
                guesses, error = self._get_guesses_from_agent(team, hint.word, hint.count)
                if error:
                    if attempt < max_retries:
                        self.game.cancel_turn()  # Discard turn without adding to history
                        self._log(f"Guesser error: {error}")
                        next_delay = base_retry_delay * (2 ** attempt) + random.uniform(0, 2)
                        self._log(f"Will retry in {next_delay:.1f} seconds...")
                        continue
                    else:
                        self.game.end_turn()  # Final attempt, add to history
                        return f"Guesser error after {max_retries} retries: {error}"
            except Exception as e:
                if attempt < max_retries:
                    self.game.cancel_turn()  # Discard turn without adding to history
                    self._log(f"Guesser exception: {str(e)}")
                    next_delay = base_retry_delay * (2 ** attempt)
                    self._log(f"Will retry in {next_delay} seconds...")
                    continue
                else:
                    self.game.end_turn()  # Final attempt, add to history
                    return f"Guesser exception after {max_retries} retries: {str(e)}"
            
            if not guesses:
                # NOTE: Empty guesses could indicate either:
                # 1. Intentional pass by the LLM (strategic decision)
                # 2. LLM failure to generate valid guesses
                # We treat both cases the same way - end the turn gracefully
                self._log("   Team passes (no guesses)")
                self.game.end_turn()
                return None

            # Enforce guess limits and basic validation
            max_guesses = self.game_config.MAX_GUESSES_PER_TURN
            if max_guesses is None:
                max_guesses = hint.count + 1

            guesses = self._clean_guesses(guesses, max_guesses)
            if not guesses:
                self._log("   Team passes after validation (no valid guesses)")
                self.game.end_turn()
                return None

            self._log(f"Guesses: {[self._sanitize_for_log(g) for g in guesses]}")

            # 4. Execute guesses
            self._execute_guesses(team, guesses)

            # 5. End turn
            self.game.end_turn()
            return None  # Success!

    def run(self) -> GameResult:
        """
        Run the complete game until completion.
        
        Returns:
            GameResult with complete game data
        """
        self._log(f"\nSTARTING GAME: {self.game_id}")
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
            blue_hint_giver_model=self.agents[Team.BLUE]['hint_giver'].get_model_name(),
            blue_guesser_model=self.agents[Team.BLUE]['guesser'].get_model_name(),
            red_hint_giver_model=self.agents[Team.RED]['hint_giver'].get_model_name(),
            red_guesser_model=self.agents[Team.RED]['guesser'].get_model_name(),
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
