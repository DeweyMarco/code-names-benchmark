# Bug Report: analyze_benchmark_results.py

This document outlines bugs and issues discovered during code review of the benchmark analysis script.

---

## Critical Bugs

### 1. ELO Rating Calculation is Completely Broken

**Location:** Lines 757-767

```python
for key, combo in self.team_combinations.items():
    if (combo.get('blue_hint_giver') and combo.get('blue_guesser') and
        combo.get('red_hint_giver') and combo.get('red_guesser')):
        combo_key = key
        blue_hg = combo['blue_hint_giver']
        blue_g = combo['blue_guesser']
        red_hg = combo['red_hint_giver']
        red_g = combo['red_guesser']
        break  # <-- BREAKS ON FIRST COMBO FOUND!
```

**Problem:** The loop breaks after finding *any* team combination with all fields populated. This means **ALL games are attributed to the first combination**, not the actual models that played each game. The Elo ratings are completely incorrect.

**Root Cause:** Games store agent class names (`"BAMLHintGiver"`, `"BAMLGuesser"`) in the `agents` field, not model names (`"OpenRouterDevstral"`). There is no reliable way to link a game back to its correct team combination.

**Impact:** All Elo ratings are meaningless.

---

### 2. `analyze_model_performance()` Loses Model Identity

**Location:** Lines 178, 189

```python
model_key = f"{team}_hint_giver"  # This is simplified
# ...
model_key = f"{team}_guesser"  # This is simplified
```

**Problem:** All blue hint givers are grouped under `"blue_hint_giver"` regardless of which model they are. Same for other roles. The function cannot distinguish between different models' performance.

**Impact:** Per-model performance analysis from game history is impossible; all models on the same team are conflated.

---

### 3. String Formatting Error Will Crash

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

### 4. Double `clean_model_name` Call

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

### 5. Incorrect Versatility Score for Zero Win Rates

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

### 6. Win Rate Calculation Includes Draws

**Location:** Lines 237-238

```python
'blue_win_rate': combo_stats['blue_wins'] / combo_stats['games_played'] if combo_stats['games_played'] > 0 else 0,
```

**Problem:** `games_played` includes draws, so win rates are slightly deflated. For example, 5 wins + 3 losses + 2 draws = 5/10 = 50% instead of 5/8 = 62.5%.

**Consideration:** This may be intentional (treating draws as "not wins"), but should be documented.

---

## Minor Issues

### 7. Redundant MODEL_DISPLAY_NAMES Dictionary

**Location:** Lines 23-96 and Line 19

The file defines a large `MODEL_DISPLAY_NAMES` dictionary but also imports `get_model_display_name` from `model_config`. These likely overlap and could diverge over time, causing inconsistent model naming.

**Recommendation:** Use a single source of truth for model display names.

---

### 8. Bare `except` Clause

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

### 9. No Game-to-Combination Linkage in Data Structure

**Location:** Game data structure

The `games` array stores agent class names in the `agents` field:
```json
"agents": {
    "blue_hint_giver": "BAMLHintGiver",
    "blue_guesser": "BAMLGuesser",
    "red_hint_giver": "BAMLHintGiver",
    "red_guesser": "BAMLGuesser"
}
```

These are agent class names, not model identifiers like `"OpenRouterDevstral"`. This makes it impossible to:
- Link individual games to their team combinations
- Calculate per-model statistics from game-level data
- Track model performance across games

**Recommendation:** Store model names in the game data, or add a `combination_key` field to each game that references `team_combinations`.

---

## Summary Table

| Bug | Severity | Impact |
|-----|----------|--------|
| ELO calculation breaks on first combo | Critical | All Elo ratings are wrong |
| `analyze_model_performance` generic keys | Critical | Per-model game stats unusable |
| p_value formatting crash | Critical | Script crashes on certain data |
| Double `clean_model_name` | Medium | Inefficiency, possible inconsistency |
| Zero win rate versatility | Medium | Misleading metric |
| Draws in win rate denominator | Medium | Deflated win rates |
| Redundant display name dict | Minor | Maintenance burden |
| Bare except clause | Minor | Could mask errors |
| No game-to-combo linkage | Minor | Limits analysis capabilities |

---

## Recommended Priority

1. **Fix the data structure** - Add model names or combination keys to game records
2. **Fix ELO calculation** - Use correct team combination per game
3. **Fix p_value formatting** - Prevent crash
4. **Fix `analyze_model_performance`** - Extract actual model names from data
5. **Clean up redundant code** - Single source for display names
