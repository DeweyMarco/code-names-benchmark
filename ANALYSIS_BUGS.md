# Bug Report: analyze_benchmark_results.py

This document outlines bugs and issues discovered during code review of the benchmark analysis script.

---

## Medium Severity Bugs

### 3. Win Rate Calculation Includes Draws

**Location:** Lines 237-238

```python
'blue_win_rate': combo_stats['blue_wins'] / combo_stats['games_played'] if combo_stats['games_played'] > 0 else 0,
```

**Problem:** `games_played` includes draws, so win rates are slightly deflated. For example, 5 wins + 3 losses + 2 draws = 5/10 = 50% instead of 5/8 = 62.5%.

**Consideration:** This may be intentional (treating draws as "not wins"), but should be documented.

---

## Minor Issues

### 4. Redundant MODEL_DISPLAY_NAMES Dictionary

**Location:** Lines 23-96 and Line 19

The file defines a large `MODEL_DISPLAY_NAMES` dictionary but also imports `get_model_display_name` from `model_config`. These likely overlap and could diverge over time, causing inconsistent model naming.

**Recommendation:** Use a single source of truth for model display names.

---

### 5. Bare `except` Clause

**Location:** Line 1176

```python
except:
    p_value = None
    significant = overall_blue_rate > 0.55
```

**Problem:** Catches all exceptions including `KeyboardInterrupt`, `SystemExit`, etc.

**Fix:**
```python
except (ValueError, TypeError, AttributeError) as e:
    p_value = None
    significant = overall_blue_rate > 0.55
```

---

## Summary Table

| Bug | Severity | Impact |
|-----|----------|--------|
| Draws in win rate denominator | Medium | Deflated win rates |
| Redundant display name dict | Minor | Maintenance burden |
| Bare except clause | Minor | Could mask errors |

---

## Recommended Priority

1. **Clean up redundant code** - Single source for display names
