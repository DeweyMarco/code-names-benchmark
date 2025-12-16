# Code Review: code-names-benchmark

**Reviewer:** Senior Engineer Review
**Date:** December 2025
**Status:** Issues identified for fixing

---

## Summary

This document contains bugs, errors, and issues identified during a thorough review of the Codenames AI benchmark codebase. Issues are categorized by severity.

---

## Logic Errors

### 1. Guesser retry after turn started corrupts game state
**File:** `orchestrator/game_runner.py:440-456`
**Severity:** High

When `_get_guesses_from_agent()` fails after `start_turn()` has been called, the code calls `end_turn()` and then retries. However, this adds a turn with zero guesses to the history, then tries to start a new turn on the retry. This corrupts the turn history and may cause unexpected behavior.

```python
try:
    guesses, error = self._get_guesses_from_agent(team, hint.word, hint.count)
    if error:
        self.game.end_turn()  # BUG: Turn added to history with no guesses
        if attempt < max_retries:
            continue  # Now will try start_turn again, incrementing turn_number
```

**Fix:** Don't call `end_turn()` on error during retries, or track that the turn needs to be discarded.

---

### 2. sort_words_to_codename_groups has misleading behavior
**File:** `utils/generate_words.py:65-99`
**Severity:** Medium

The function name and docstring suggest it "sorts words into codename groups" but it actually assigns colors based purely on list position (first N words are blue, next M are red, etc.). This is deterministic, not random.

The actual randomization happens in `Board._initialize_board()` which shuffles the colors. However, if someone uses `sort_words_to_codename_groups()` directly expecting random assignment, they won't get it.

**Fix:** Rename to `assign_words_to_codename_groups_by_position()` or add shuffling.

---

## Configuration Issues

### 3. Default models reference unverified/non-existent models
**Files:** Multiple locations
**Severity:** High

Several files default to models marked as "UNVERIFIED" in `baml_agents.py`:

- `demo_simple_game.py:59-62`: Uses `BAMLModel.GEMINI_25_FLASH` (unverified)
- `quick_benchmark.py:400`: Uses `BAMLModel.GPT5`, `BAMLModel.GEMINI_25_PRO`, `BAMLModel.CLAUDE_HAIKU_45` (unverified)
- `baml_agents.py:282`: Default for anthropic is `CLAUDE_HAIKU_45` (unverified)
- `baml_agents.py:290`: Default for google is `GEMINI_25_FLASH` (unverified)
- `config.py:105`: `ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"` (unverified)

These models may not exist or may fail at runtime.

**Fix:** Change defaults to verified models like `GPT4O_MINI`, `CLAUDE_HAIKU_35`, `GEMINI_20_FLASH`.

---

### 4. API key check references non-existent models
**File:** `quick_benchmark.py:136-142`
**Severity:** Medium

The `_check_api_keys()` method checks for API keys based on models that may not exist:
```python
required_keys = {
    BAMLModel.GPT5: ["OPENAI_API_KEY"],           # GPT5 doesn't exist
    BAMLModel.GEMINI_25_PRO: ["GOOGLE_API_KEY"],  # Unverified
    BAMLModel.CLAUDE_HAIKU_45: ["ANTHROPIC_API_KEY"],  # Unverified
    ...
}
```

**Fix:** Update to match `get_benchmark_models()` or use verified models.

---

## Design Issues

### 5. process_result should not be abstract
**File:** `agents/base.py:134-149`
**Severity:** Medium

`process_result` is marked as `@abstractmethod` but the docstring says "Not required to do anything (default implementation can pass)". This forces all subclasses to implement it even when not needed.

```python
@abstractmethod
def process_result(self, guessed_word: str, was_correct: bool, color: CardColor):
    """
    ...
    Notes:
        - Not required to do anything (default implementation can pass)  # Contradicts @abstractmethod
    """
    pass
```

**Fix:** Remove `@abstractmethod` and provide a default empty implementation:
```python
def process_result(self, guessed_word: str, was_correct: bool, color: CardColor):
    """Optional feedback method. Override if needed."""
    pass
```

---

### 6. RESTRICTED_TEMPERATURE_MODELS contains unverified models
**File:** `model_config.py:60-72`
**Severity:** Low

The set includes models like `GPT5`, `GPT5_MINI`, `GPT5_NANO`, etc. that don't exist. While this won't cause immediate errors, it's misleading.

**Fix:** Only include verified models or clearly document which are speculative.

---

## Potential Runtime Errors

### 7. Potential IndexError when game_results is empty
**File:** `benchmark_runner.py:248-253`
**Severity:** Medium

The code accesses `game_results[0]` without checking if the list is empty:

```python
agent_names = {
    'blue_hint_giver': game_results[0].blue_hint_giver_name,  # IndexError if empty
    ...
}
```

**Fix:** Add empty check:
```python
if not game_results:
    return BenchmarkResult(...)  # Handle empty case

agent_names = {
    'blue_hint_giver': game_results[0].blue_hint_giver_name,
    ...
}
```

---

### 8. BAMLModel enum creation may fail with invalid strings
**File:** `analyze_benchmark_results.py:512-513`
**Severity:** Medium

The code calls `BAMLModel(combo['blue_hint_giver'])` which could fail if the stored string doesn't exactly match an enum value:

```python
blue_team = f"{get_model_display_name(BAMLModel(combo['blue_hint_giver']))}"
```

**Fix:** Add try/except or validate the value first:
```python
try:
    model = BAMLModel(combo['blue_hint_giver'])
    display_name = get_model_display_name(model)
except ValueError:
    display_name = combo['blue_hint_giver']  # Fallback to raw string
```

---

## Minor Issues

### 9. Inconsistent error handling in game_runner retries
**File:** `orchestrator/game_runner.py:400-407`
**Severity:** Low

The retry delay logging is inconsistent - sometimes includes jitter (`+ random.uniform(0, 2)`) and sometimes doesn't:

```python
next_delay = base_retry_delay * (2 ** attempt) + random.uniform(0, 2)  # Line 395
self._log(f"Will retry in {next_delay:.1f} seconds...")

next_delay = base_retry_delay * (2 ** attempt)  # Line 403 - no jitter
self._log(f"Will retry in {next_delay} seconds...")  # Also no .1f formatting
```

**Fix:** Make consistent.

---

### 10. Unused import in model_config.py
**File:** `model_config.py:17`
**Severity:** Low

The file imports `BAMLModel` from `agents.llm` and defines `get_model_display_name()` that takes a `BAMLModel`, but `analyze_benchmark_results.py` also imports from `model_config` and needs `BAMLModel` - this circular-ish pattern is fine but could be cleaner.

---

### 11. Missing validation for board_size in custom config
**File:** `config.py:50-83`
**Severity:** Low

`GameConfig.custom()` validates `board_size >= 9` and `board_size % 2 == 0` but the calculation for word distribution could result in negative `neutral_words` for edge cases.

```python
neutral_words = board_size - starting_words - other_words - 1  # Could be negative
```

**Fix:** Add validation that `neutral_words >= 0`.

---

## Summary Table

| # | File | Line | Severity | Description |
|---|------|------|----------|-------------|
| 1 | game_runner.py | 440-456 | High | Retry corrupts turn history |
| 2 | generate_words.py | 65-99 | Medium | Misleading function behavior |
| 3 | Multiple | - | High | Unverified models as defaults |
| 4 | quick_benchmark.py | 136-142 | Medium | API check uses non-existent models |
| 5 | base.py | 134-149 | Medium | Abstract method should be optional |
| 6 | model_config.py | 60-72 | Low | Unverified models in set |
| 7 | benchmark_runner.py | 248-253 | Medium | Potential IndexError |
| 8 | analyze_benchmark_results.py | 512-513 | Medium | BAMLModel construction may fail |
| 9 | game_runner.py | 395-407 | Low | Inconsistent retry logging |
| 10 | model_config.py | 17 | Low | Import pattern |
| 11 | config.py | 50-83 | Low | Missing neutral_words validation |

---

## Recommended Priority

1. **Fix high severity issues (#1, #3)** - These affect reliability
2. **Fix medium severity issues** - These may cause problems in edge cases
3. **Address low severity issues** - Code quality improvements
