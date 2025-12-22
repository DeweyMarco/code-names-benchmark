# Bug Report: analyze_benchmark_results.py

This document outlines bugs and issues discovered during code review of the benchmark analysis script.

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
| Redundant display name dict | Minor | Maintenance burden |
| Bare except clause | Minor | Could mask errors |

---

## Recommended Priority

1. **Clean up redundant code** - Single source for display names
