# Code Review: Codenames AI Benchmark

This document contains a comprehensive review of the codebase identifying bugs, potential issues, code quality problems, and areas for improvement.

---

## Medium Severity Issues

### 5. Memory Leak in Guesser History
**Location:** `agents/llm/baml_agents.py:211, 252-256`

```python
self.guess_history = []  # Track guess results for analysis
# ...
def process_result(self, guessed_word: str, was_correct: bool, color: CardColor):
    self.guess_history.append({...})
```

**Problem:** `guess_history` is never cleared between games when agents are reused. In long benchmark runs with agent reuse, this list will grow unbounded.

---

### 7. Inconsistent Winner Value in Analysis
**Location:** `analyze_benchmark_results.py:73-74`

```python
blue_won = game.get('winner') == 'BLUE'
red_won = game.get('winner') == 'RED'
```

**Problem:** The code expects winner to be 'BLUE' or 'RED' (uppercase), but `GameResult.to_dict()` returns `self.winner.value` which uses the Team enum value ('blue' or 'red' - lowercase). The comparison will always fail.

---

### 8. Unreachable Code in analyze_model_performance
**Location:** `analyze_benchmark_results.py:109`

```python
if stats['total_games'] > 0:
```

**Problem:** `total_games` is never incremented in the `analyze_model_performance` method, so this condition will always be False and no data will be added to `analysis_data`.

---

### 9. Missing Validation for Custom Board Sizes
**Location:** `config.py:54-57`

```python
if board_size < 9:
    raise ValueError("Board size must be at least 9")
if board_size % 2 == 0:
    raise ValueError("Board size should be odd for fair play")
```

**Problem:** The custom board size validation requires odd numbers, but the default `BOARD_SIZE = 25` is used elsewhere without this check. Also, the distribution algorithm at lines 61-63 doesn't guarantee the total adds up correctly for all board sizes.

---

### 10. Inconsistent Case Handling
**Location:** Multiple files

**Problem:** Words are normalized to lowercase in `Board.__init__` but some comparisons/lookups don't use `.lower()`:
- `game_runner.py:386-387` creates `board_words_lower` and `revealed_lower` locally, but this pattern isn't consistent
- The `sort_words_to_codename_groups` function in `generate_words.py` doesn't normalize case

---

### 11. Silent Failure on Invalid Guesses
**Location:** `orchestrator/game_runner.py:365-376`

```python
for guess in guesses:
    if not isinstance(guess, str):
        self._log(f"   Skipping non-string guess: {guess}")
        continue
```

**Problem:** Non-string guesses are silently skipped without tracking. This could mask bugs in the LLM response parsing.

---

## Low Severity Issues

### 12. Duplicate Model Display Name Mappings
**Location:** `demo_simple_game.py:151-224` and `model_config.py:108-117`

**Problem:** `get_model_display_name` is defined in both files with different implementations. The one in `demo_simple_game.py` has many more mappings than `model_config.py`.

---

### 13. Unused Import
**Location:** `demo_simple_game.py:47`

```python
from game import Board, Team, CardColor
```

**Problem:** `Team` is imported but not used directly in the file (it's only used through the agents).

---

### 14. Magic Numbers
**Location:** Multiple files

**Problem:** Several magic numbers without explanation:
- `game_runner.py:273`: `random.uniform(0, 2)` - Why 0-2 seconds jitter?
- `demo_simple_game.py:488`: References "lines 41-44" but models are at lines 57-61
- `analyze_benchmark_results.py:218`: `combo_df['games_played'] >= 2` - Why 2?

---

### 15. Inconsistent Logging
**Location:** `game/state.py`, `orchestrator/game_runner.py`

**Problem:** `GameState.print_status()` uses `logger.info()` while `GameRunner._log()` uses `print()` when verbose. This inconsistency makes log management difficult.

---

### 16. Poor Error Messages
**Location:** `agents/base.py:27-28`

```python
if not isinstance(self.count, int) or self.count < 0:
    return False, "Hint count must be a non-negative integer"
```

**Problem:** The validation allows count of 0, but `config.py:28` sets `MIN_HINT_COUNT: int = 1`. These should be consistent.

---

### 17. Missing Type Hints
**Location:** `utils/generate_words.py`

**Problem:** Most functions lack return type hints:
```python
def load_words_from_csv(csv_path=None):  # Should be -> List[str]
def generate_word_list(num_words=None, csv_path=None):  # Should be -> List[str]
```

---

### 18. Orphaned Code Reference
**Location:** `demo_simple_game.py:488`

```python
print("To try other demos, check out: demo_llm_game.py")
```

**Problem:** `demo_llm_game.py` doesn't exist in the repository.

---

### 19. Incomplete Dataclass Field
**Location:** `orchestrator/game_runner.py:31`

```python
error: Optional[str] = None
timestamp: datetime = field(default_factory=datetime.now)
```

**Problem:** Using `datetime.now` as a factory function captures the time when the factory is called, not when the instance is created. Should be `lambda: datetime.now()` to be explicit.

---

### 20. Potential Confusion in Bomb Handling
**Location:** `game/state.py:192-198`

```python
if last_result.hit_bomb:
    if self._current_team == Team.RED:
        self._game_outcome = GameOutcome.BLUE_WIN
    else:
        self._game_outcome = GameOutcome.RED_WIN
```

**Problem:** The logic is correct but could be clearer. A comment explaining "the team that hit the bomb loses, so the other team wins" would help.

---

### 21. Empty Turn Handling Ambiguity
**Location:** `orchestrator/game_runner.py:352-356`

```python
if not guesses:
    self._log("   Team passes (no guesses)")
    self.game.end_turn()
    return None
```

**Problem:** If the LLM returns an empty list of guesses, the turn ends silently. This might be intentional (passing) or might indicate an LLM failure. There's no way to distinguish.

---

## Code Quality Issues

### 22. Long Methods
**Location:** `orchestrator/game_runner.py:254-423`

**Problem:** `_execute_turn` is 170 lines long with deeply nested logic. Should be refactored into smaller, focused methods.

---

### 23. Configuration Coupling
**Location:** `config.py:301-310`

```python
BOARD_SIZE = default_config.game.BOARD_SIZE
BLUE_WORDS = default_config.game.BLUE_WORDS
```

**Problem:** Module-level constants are created from a default config instance, creating tight coupling. Changes to configuration require module reload.

---

### 24. Inconsistent Naming Conventions
**Location:** Multiple files

**Problem:** Mixed naming styles:
- `_PROVIDER_MODEL_MAP` (module constant with underscore prefix)
- `BENCHMARK_MODELS` (no underscore)
- `_DEFAULT_CSV_PATH` (underscore)
- Some constants use `SCREAMING_SNAKE_CASE`, others don't

---

### 25. Missing docstrings
**Location:** `quick_benchmark.py:345`

```python
def run(self) -> QuickBenchmarkResult:
    """Run the quick benchmark."""
```

**Problem:** The docstring doesn't document the possible exceptions, return value details, or side effects.

---

### 26. Dead Code Path in Team Combination Logic
**Location:** `quick_benchmark.py:363-376`

```python
for blue_hint in BENCHMARK_MODELS:
    for red_guess in BENCHMARK_MODELS:
        for blue_guess in BENCHMARK_MODELS:
            if blue_guess != blue_hint:
                for red_hint in BENCHMARK_MODELS:
                    if (red_hint != red_guess and
                        red_hint != blue_hint and
                        red_hint != blue_guess and
                        red_guess != blue_hint and
                        red_guess != blue_guess):
```

**Problem:** The comment says "Each model can only play one role per game" but this constraint would mean each model appears exactly once. The condition checks prevent a model from appearing in multiple roles, but with 5 models and 4 roles, this leaves only valid combinations where all 4 roles are different models. This severely limits test coverage.

---

## Security Considerations

### 27. API Key Exposure Risk
**Location:** `config.py:107-197`

**Problem:** Model costs dictionary includes pricing for models that reference API keys. While keys aren't stored here, the configuration pattern could lead developers to add sensitive data.

---

### 28. No Input Sanitization for Hint Words
**Location:** `orchestrator/game_runner.py:123-156`

**Problem:** Hint words from LLMs are validated against board words but not sanitized for logging. If an LLM returns malicious content, it could appear in logs.

---

## Documentation Issues

### 29. Outdated Comments
**Location:** `baml_src/clients.baml:8, 54, 271, etc.`

```
// GPT-5 Series (Latest - August 2025)
// GPT-4.1 Series (April 2025)
// Claude 4.5 Series (Latest - October 2025)
```

**Problem:** Comments reference future dates (2025) as if they're historical releases.

---

### 30. Misleading Variable Name
**Location:** `quick_benchmark.py:41`

```python
GAMES_PER_COMBINATION = 2  # Reduced for quick results
```

**Problem:** Comment says "reduced" but there's no indication of what the original/full value would be.

---

## Summary

| Category | Count |
|----------|-------|
| High Severity | 0 |
| Medium Severity | 6 |
| Low Severity | 10 |
| Code Quality | 5 |
| Security | 2 |
| Documentation | 2 |
| **Total** | **25** |

### Priority Recommendations

1. **Medium:** Fix the winner value comparison case mismatch in analysis (issue #7).
2. **Medium:** Refactor long methods and add comprehensive error handling.
3. **Low:** Clean up duplicate code and improve documentation.
