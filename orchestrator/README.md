# Game Orchestrator

Coordinates 4 agents through a complete Codenames game.

## GameRunner

Handles the full game loop: hint → guesses → validation → win checks → team switching

### Basic Usage

```python
from utils import generate_word_list
from game import Board, Team
from agents.llm import BAMLHintGiver, BAMLGuesser, BAMLModel
from orchestrator import GameRunner

# Setup
words = generate_word_list(25)
board = Board(words)

# Run game
runner = GameRunner(
    board=board,
    blue_hint_giver=BAMLHintGiver(Team.BLUE, BAMLModel.GPT4O_MINI),
    blue_guesser=BAMLGuesser(Team.BLUE, BAMLModel.GPT4O_MINI),
    red_hint_giver=BAMLHintGiver(Team.RED, BAMLModel.CLAUDE_SONNET_45),
    red_guesser=BAMLGuesser(Team.RED, BAMLModel.CLAUDE_SONNET_45),
    verbose=True,
    game_id="game_001"
)

result = runner.run()
print(f"Winner: {result.winner}, Turns: {result.total_turns}")
```

### GameResult

Returned after game completion:

```python
result.game_id           # Unique identifier
result.outcome           # BLUE_WIN, RED_WIN, IN_PROGRESS
result.winner            # Winning team
result.total_turns       # Number of turns
result.final_scores      # (blue_remaining, red_remaining)
result.snapshot          # Complete game state (JSON-serializable)
result.error             # Error message if any
result.to_dict()         # Convert to dictionary
```

## Configuration

Use `config.py` to customize:

```python
from config import OrchestratorConfig, GameConfig

# Orchestrator settings
config = OrchestratorConfig()
config.VERBOSE_DEFAULT = True
config.STRICT_VALIDATION = True
config.TURN_DELAY = 0.5  # Delay for demos

# Game settings
game_config = GameConfig()
game_config.MAX_TURNS = 100
```

**Available options:**
- Logging (verbosity, file output)
- Validation (strict mode, hints)
- Timing (delays)
- Tournament settings

## Features

**Validation:**
- Single-word hints only
- No board words as hints
- Invalid guesses caught
- Agent errors logged

**Safety:**
- Max turn limit
- Exception handling
- Always-valid state

**Feedback:**
- Guessers get results via `process_result()`
- Allows learning within game

**Data:**
- Full game history
- Agent names
- Timestamps
- Error tracking

Use `verbose=True` to see turn-by-turn progress.

