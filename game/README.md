# Codenames Game Engine

Core game mechanics optimized for LLM agents.

## Core Classes

### `Board`

Word-to-color mapping for the Codenames board.

```python
from game import Board, CardColor

words = ["apple", "banana", "car", ...]  # 25 words
board = Board(words)

# Get words by color
blue_words = board.get_words_by_color(CardColor.BLUE)
color = board.get_color("apple")
all_words = board.all_words
```

**Features:** Configurable size (default 25), dynamic color distribution, starting team gets +1 word

### `GameState`

Manages turn taking, guessing, and win conditions.

```python
from game import GameState

game = GameState(board)

# Play a turn
game.start_turn("animal", 2)
result = game.make_guess("dog")
game.end_turn()

# Check status
print(game.is_game_over)
print(game.game_outcome)
print(game.get_team_scores())
snapshot = game.get_snapshot()  # JSON-serializable
```

**Features:** Full history tracking, automatic win detection, validation, bomb detection

## Supporting Types

- **CardColor**: `RED`, `BLUE`, `NEUTRAL`, `BOMB`
- **Team**: `RED`, `BLUE`
- **GameOutcome**: `RED_WIN`, `BLUE_WIN`, `IN_PROGRESS`
- **TurnResult**: Outcome of a guess (`word`, `color`, `correct`, `hit_bomb`)
- **Turn**: Complete turn record (`team`, `hint_word`, `hint_count`, `guesses`)

## Game Flow

1. Create `Board` with 25 words
2. Initialize `GameState`
3. Loop: `start_turn()` → `make_guess()` × N → `end_turn()`
4. Check `game_outcome`

## Design Features

**LLM-Optimized:**
- Plain word lists (no grid layout) - better for LLM prompts
- Full history for analysis (hints, guesses, outcomes)
- Validation catches LLM errors (hallucinations, invalid guesses)

**Consistent:**
- Blue always starts first
- Immutable state (copies returned)
- JSON-serializable snapshots

## Configuration

Use `config.py` for custom variants:

```python
from config import GameConfig

# Custom board size
config = GameConfig.custom(board_size=49)  # Larger board
mini = GameConfig.custom(board_size=9)     # Quick games
```

See `config.py` for all options.

