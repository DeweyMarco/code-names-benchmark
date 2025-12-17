# Code Review: code-names-benchmark

**Reviewer:** Senior Engineer Review
**Date:** December 2025
**Status:** Issues identified for fixing

---

## Summary

This document contains bugs, errors, and issues identified during a thorough review of the Codenames AI benchmark codebase. Issues are categorized by severity.


## Minor Issues

### 8. Inconsistent error handling in game_runner retries
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

### 9. Unused import in model_config.py
**File:** `model_config.py:17`
**Severity:** Low

The file imports `BAMLModel` from `agents.llm` and defines `get_model_display_name()` that takes a `BAMLModel`, but `analyze_benchmark_results.py` also imports from `model_config` and needs `BAMLModel` - this circular-ish pattern is fine but could be cleaner.

---

### 10. Missing validation for board_size in custom config
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
| 8 | game_runner.py | 395-407 | Low | Inconsistent retry logging |
| 9 | model_config.py | 17 | Low | Import pattern |
| 10 | config.py | 50-83 | Low | Missing neutral_words validation |

---

## Recommended Priority

1. **Address low severity issues** - Code quality improvements
