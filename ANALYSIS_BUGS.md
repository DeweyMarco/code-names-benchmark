# Bug Report: analyze_benchmark_results.py

This document outlines bugs and issues discovered during code review of the benchmark analysis script.

---

## Critical Bugs

### 1. String Formatting Error Will Crash

**Location:** Lines 1292-1294

```python
insights.append(f"- **Blue (First-Mover) Win Rate**: {fma['overall_blue_win_rate']:.1%} "
               f"({'statistically significant' if fma['advantage_significant'] else 'not significant'}, "
               f"p={fma['p_value']:.4f})" if fma['p_value'] else f"({fma['advantage_magnitude']} advantage)")
```

**Problem:** The ternary operator placement is incorrect. When `p_value` is `None`, Python first evaluates `fma['p_value']:.4f`, which throws a `TypeError` before the conditional branch is reached.

**Fix:**
```python
if fma['p_value'] is not None:
    p_str = f"p={fma['p_value']:.4f}"
    sig_str = 'statistically significant' if fma['advantage_significant'] else 'not significant'
    insights.append(f"- **Blue (First-Mover) Win Rate**: {fma['overall_blue_win_rate']:.1%} ({sig_str}, {p_str})")
else:
    insights.append(f"- **Blue (First-Mover) Win Rate**: {fma['overall_blue_win_rate']:.1%} ({fma['advantage_magnitude']} advantage)")
```

**Impact:** Script crashes when `p_value` is `None`.

---

## Medium Severity Bugs

### 2. Double `clean_model_name` Call

**Location:** Lines 229-232

```python
'blue_hint_giver': clean_model_name(MODEL_DISPLAY_NAMES.get(combo_stats['blue_hint_giver'], combo_stats['blue_hint_giver'])),
```

**Problem:** `MODEL_DISPLAY_NAMES.get()` returns a display name (e.g., `"Devstral"`), then `clean_model_name()` tries to look it up again in the same dictionary (won't find it, falls through to fallback logic).

**Fix:**
```python
'blue_hint_giver': clean_model_name(combo_stats['blue_hint_giver']),
```

**Impact:** Inefficiency and potential naming inconsistency between different parts of the codebase.

---

### 3. Incorrect Versatility Score for Zero Win Rates

**Location:** Line 954

```python
versatility = 1 - abs(hg_rate - g_rate) / max(hg_rate, g_rate, 0.001)
```

**Problem:** If both `hg_rate` and `g_rate` are 0:
- `abs(0 - 0) = 0`
- `max(0, 0, 0.001) = 0.001`
- `versatility = 1 - 0/0.001 = 1.0`

This reports "perfect versatility" when the model won zero games in both roles.

**Fix:**
```python
if hg_rate == 0 and g_rate == 0:
    versatility = 0  # No wins = no demonstrated versatility
elif hg_rate + g_rate > 0:
    versatility = 1 - abs(hg_rate - g_rate) / max(hg_rate, g_rate)
else:
    versatility = 0
```

**Impact:** Misleading versatility metric for poorly performing models.

---

### 4. Win Rate Calculation Includes Draws

**Location:** Lines 237-238

```python
'blue_win_rate': combo_stats['blue_wins'] / combo_stats['games_played'] if combo_stats['games_played'] > 0 else 0,
```

**Problem:** `games_played` includes draws, so win rates are slightly deflated. For example, 5 wins + 3 losses + 2 draws = 5/10 = 50% instead of 5/8 = 62.5%.

**Consideration:** This may be intentional (treating draws as "not wins"), but should be documented.

---

## Minor Issues

### 5. Redundant MODEL_DISPLAY_NAMES Dictionary

**Location:** Lines 23-96 and Line 19

The file defines a large `MODEL_DISPLAY_NAMES` dictionary but also imports `get_model_display_name` from `model_config`. These likely overlap and could diverge over time, causing inconsistent model naming.

**Recommendation:** Use a single source of truth for model display names.

---

### 6. Bare `except` Clause

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
| p_value formatting crash | Critical | Script crashes on certain data |
| Double `clean_model_name` | Medium | Inefficiency, possible inconsistency |
| Zero win rate versatility | Medium | Misleading metric |
| Draws in win rate denominator | Medium | Deflated win rates |
| Redundant display name dict | Minor | Maintenance burden |
| Bare except clause | Minor | Could mask errors |

---

## Recommended Priority

1. **Fix p_value formatting** - Prevent crash
2. **Clean up redundant code** - Single source for display names
