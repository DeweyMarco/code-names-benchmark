# Comprehensive Codenames Benchmark Insights
Generated from 48 games across 24 team combinations

## Executive Summary

- **Blue (First-Mover) Win Rate**: 50.0% (not significant, p=0.5573)
- **Comeback Rate**: 42.6% of games had comebacks
- **Average Game Length**: 7.5 turns

## Elo Rankings
*Skill-based rating system accounting for opponent strength*

1. **Nemotron Nano**: 1505 combined (HG: 1509, G: 1500, best: hint_giver)
2. **MIMO V2 Flash**: 1505 combined (HG: 1500, G: 1509, best: guesser)
3. **Devstral**: 1495 combined (HG: 1491, G: 1500, best: guesser)
4. **OLMo 3.1 32B**: 1495 combined (HG: 1500, G: 1491, best: hint_giver)

## Role Versatility Analysis
*Which models perform well in both roles?*

- **Devstral**: 52.1% combined win rate (HG: 62.5%, G: 41.7%, versatility: 0.67, best: hint_giver)
- **MIMO V2 Flash**: 47.9% combined win rate (HG: 37.5%, G: 58.3%, versatility: 0.64, best: guesser)
- **OLMo 3.1 32B**: 45.8% combined win rate (HG: 54.2%, G: 37.5%, versatility: 0.69, best: hint_giver)
- **Nemotron Nano**: 33.3% combined win rate (HG: 25.0%, G: 41.7%, versatility: 0.60, best: guesser)

## Statistical Confidence (95% Wilson CI)
*Win rates with confidence intervals - wider CI = less certainty*

- **Devstral** (hint_giver, blue): 75.0% [46.8% - 91.1%] (n=12, CI width: 44.3pp)
- **MIMO V2 Flash** (guesser, red): 66.7% [39.1% - 86.2%] (n=12, CI width: 47.1pp)
- **MIMO V2 Flash** (hint_giver, blue): 58.3% [32.0% - 80.7%] (n=12, CI width: 48.7pp)
- **OLMo 3.1 32B** (hint_giver, red): 58.3% [32.0% - 80.7%] (n=12, CI width: 48.7pp)
- **OLMo 3.1 32B** (guesser, blue): 58.3% [32.0% - 80.7%] (n=12, CI width: 48.7pp)
- **Devstral** (hint_giver, red): 50.0% [25.4% - 74.6%] (n=12, CI width: 49.2pp)
- **MIMO V2 Flash** (guesser, blue): 50.0% [25.4% - 74.6%] (n=12, CI width: 49.2pp)
- **OLMo 3.1 32B** (hint_giver, blue): 50.0% [25.4% - 74.6%] (n=12, CI width: 49.2pp)
- **Nemotron Nano** (guesser, blue): 50.0% [25.4% - 74.6%] (n=12, CI width: 49.2pp)
- **Devstral** (guesser, blue): 41.7% [19.3% - 68.0%] (n=12, CI width: 48.7pp)

## Hint Giver Efficiency Analysis
*Deep metrics on hint quality and strategy*

| Model | Team | Avg Hint | Yield | Efficiency | Risk Profile | Win Rate |
|-------|------|----------|-------|------------|--------------|----------|
| Devstral | blue | 2.0 | 1.27 | 64.8% | balanced | 75.0% |
| OLMo 3.1 32B | blue | 2.3 | 1.45 | 63.4% | balanced | 50.0% |
| OLMo 3.1 32B | red | 2.1 | 1.22 | 56.8% | balanced | 58.3% |
| Devstral | red | 2.0 | 1.16 | 56.6% | balanced | 50.0% |
| MIMO V2 Flash | blue | 2.1 | 1.17 | 54.8% | balanced | 58.3% |
| MIMO V2 Flash | red | 2.2 | 1.12 | 50.0% | balanced | 16.7% |
| Nemotron Nano | blue | 3.4 | 1.50 | 43.9% | aggressive | 16.7% |
| Nemotron Nano | red | 3.9 | 1.33 | 34.3% | aggressive | 33.3% |

## Guesser Performance Analysis
*Critical metrics for guesser evaluation*

| Model | Team | 1st Guess | Overall | Bomb Rate | Risk-Adj | Win Rate |
|-------|------|-----------|---------|-----------|----------|----------|
| MIMO V2 Flash | blue | 87.8% | 69.0% | 1.19% | 65.5% | 50.0% |
| Devstral | blue | 89.4% | 72.7% | 3.03% | 63.6% | 41.7% |
| Devstral | red | 84.1% | 67.1% | 1.18% | 63.5% | 41.7% |
| MIMO V2 Flash | red | 90.0% | 72.9% | 3.39% | 62.7% | 66.7% |
| Nemotron Nano | blue | 80.0% | 67.3% | 3.85% | 55.8% | 50.0% |
| OLMo 3.1 32B | blue | 84.1% | 66.2% | 3.90% | 54.5% | 58.3% |
| OLMo 3.1 32B | red | 88.9% | 65.8% | 5.06% | 50.6% | 16.7% |
| Nemotron Nano | red | 76.2% | 57.7% | 6.41% | 38.5% | 33.3% |

## Error Analysis
*Failure modes and catastrophic errors*

### Errors by Model (Guessers)
- **MIMO V2 Flash**: 6 total errors (bombs: 3, invalid: 3)
- **OLMo 3.1 32B**: 7 total errors (bombs: 7, invalid: 0)
- **Devstral**: 8 total errors (bombs: 4, invalid: 4)
- **Nemotron Nano**: 11 total errors (bombs: 9, invalid: 2)

### Recent Bomb Hits (Context)
- Turn 4: 'net' guessed after hint 'fashion' (4)
- Turn 2: 'prince' guessed after hint 'royal' (3)
- Turn 1: 'amazon' guessed after hint 'music' (2)
- Turn 9: 'wake' guessed after hint 'squid' (2)
- Turn 10: 'jam' guessed after hint 'sweet' (2)

## Hint Word Analysis

- **Total Hints Given**: 352
- **Unique Hints**: 224 (63.6% creativity)
- **Average Hint Count**: 2.48
- **Overall Success Rate**: 83.8%
- **Perfect Hint Rate**: 33.0%

### Most Common Hints
- 'animal': 15 times
- 'metal': 11 times
- 'royalty': 6 times
- 'sound': 5 times
- 'water': 5 times
- 'kitchen': 5 times
- 'music': 4 times
- 'sport': 4 times
- 'bird': 4 times
- 'dark': 4 times

### Success Rate by Hint Count
- **1**: 93.3% success, 93.3% efficiency (30 hints)
- **2**: 83.4% success, 59.8% efficiency (205 hints)
- **3**: 82.5% success, 50.4% efficiency (80 hints)
- **4**: 81.2% success, 43.8% efficiency (16 hints)
- **5**: 100.0% success, 23.3% efficiency (6 hints)
- **6**: 75.0% success, 20.8% efficiency (8 hints)
- **7**: 60.0% success, 11.4% efficiency (5 hints)
- **8**: 100.0% success, 31.2% efficiency (2 hints)

## Game Dynamics

- **Average Lead Changes**: 0.4 per game
- **Max Lead Changes**: 3
- **Average Competitiveness**: 0.06
- **Average Deficit Overcome**: 1.9 cards

## Overall Best Hint Givers
*Aggregated across both Blue and Red teams*

1. **Devstral**: 62.5% win rate (15/24 games)
2. **OLMo 3.1 32B**: 54.2% win rate (13/24 games)
3. **MIMO V2 Flash**: 37.5% win rate (9/24 games)
4. **Nemotron Nano**: 25.0% win rate (6/24 games)

## Overall Best Guessers
*Aggregated across both Blue and Red teams*

1. **MIMO V2 Flash**: 58.3% win rate (14/24 games)
2. **Devstral**: 41.7% win rate (10/24 games)
3. **Nemotron Nano**: 41.7% win rate (10/24 games)
4. **OLMo 3.1 32B**: 37.5% win rate (9/24 games)

## Best Hint Givers (by Team)

- **Devstral** (Blue Hint Giver): 75.0% win rate (9/12 games)
- **OLMo 3.1 32B** (Red Hint Giver): 58.3% win rate (7/12 games)
- **MIMO V2 Flash** (Blue Hint Giver): 58.3% win rate (7/12 games)
- **Devstral** (Red Hint Giver): 50.0% win rate (6/12 games)
- **OLMo 3.1 32B** (Blue Hint Giver): 50.0% win rate (6/12 games)

## Best Guessers (by Team)

- **MIMO V2 Flash** (Red Guesser): 66.7% win rate (8/12 games)
- **OLMo 3.1 32B** (Blue Guesser): 58.3% win rate (7/12 games)
- **Nemotron Nano** (Blue Guesser): 50.0% win rate (6/12 games)
- **MIMO V2 Flash** (Blue Guesser): 50.0% win rate (6/12 games)
- **Devstral** (Blue Guesser): 41.7% win rate (5/12 games)

## Most Dominant Team Combinations

 1. **Devstral + OLMo 3.1 32B** vs **Nemotron Nano + MIMO V2 Flash**
    Blue Win Rate: 100.0% (2 games, 3.0 avg turns)
 2. **Devstral + Nemotron Nano** vs **MIMO V2 Flash + OLMo 3.1 32B**
    Blue Win Rate: 100.0% (2 games, 10.0 avg turns)
 3. **Devstral + MIMO V2 Flash** vs **OLMo 3.1 32B + Nemotron Nano**
    Blue Win Rate: 100.0% (2 games, 4.0 avg turns)
 4. **Devstral + OLMo 3.1 32B** vs **MIMO V2 Flash + Nemotron Nano**
    Blue Win Rate: 100.0% (2 games, 10.0 avg turns)
 5. **MIMO V2 Flash + Devstral** vs **Nemotron Nano + OLMo 3.1 32B**
    Blue Win Rate: 100.0% (2 games, 8.5 avg turns)
 6. **OLMo 3.1 32B + Nemotron Nano** vs **MIMO V2 Flash + Devstral**
    Blue Win Rate: 100.0% (2 games, 10.0 avg turns)
 7. **Devstral + MIMO V2 Flash** vs **Nemotron Nano + OLMo 3.1 32B**
    Blue Win Rate: 50.0% (2 games, 10.5 avg turns)
 8. **MIMO V2 Flash + OLMo 3.1 32B** vs **Nemotron Nano + Devstral**
    Blue Win Rate: 50.0% (2 games, 10.0 avg turns)
 9. **MIMO V2 Flash + Nemotron Nano** vs **OLMo 3.1 32B + Devstral**
    Blue Win Rate: 50.0% (2 games, 10.5 avg turns)
10. **MIMO V2 Flash + Nemotron Nano** vs **Devstral + OLMo 3.1 32B**
    Blue Win Rate: 50.0% (2 games, 9.0 avg turns)

## Best Model Synergies

- **Devstral + OLMo 3.1 32B**: 100.0% win rate (4 games)
- **Devstral + MIMO V2 Flash**: 75.0% win rate (4 games)
- **MIMO V2 Flash + Devstral**: 75.0% win rate (4 games)
- **Devstral + Nemotron Nano**: 50.0% win rate (4 games)
- **MIMO V2 Flash + Nemotron Nano**: 50.0% win rate (4 games)
- **MIMO V2 Flash + OLMo 3.1 32B**: 50.0% win rate (4 games)
- **OLMo 3.1 32B + Devstral**: 50.0% win rate (4 games)
- **OLMo 3.1 32B + MIMO V2 Flash**: 50.0% win rate (4 games)
- **OLMo 3.1 32B + Nemotron Nano**: 50.0% win rate (4 games)
- **Nemotron Nano + MIMO V2 Flash**: 25.0% win rate (4 games)

## Game Efficiency (Speed of Victory)

- **Devstral**: 62.5% win rate, 7.0 avg turns/game, 11.1 turns/win
- **OLMo 3.1 32B**: 54.2% win rate, 7.3 avg turns/game, 13.5 turns/win
- **MIMO V2 Flash**: 37.5% win rate, 8.4 avg turns/game, 22.3 turns/win
- **Nemotron Nano**: 25.0% win rate, 6.7 avg turns/game, 26.7 turns/win

## Strategic Insights

### Models Most Frequently in Winning Combinations:
- **Devstral**: 6 appearances in winning combinations
- **OLMo 3.1 32B**: 6 appearances in winning combinations
- **Nemotron Nano**: 6 appearances in winning combinations
- **MIMO V2 Flash**: 6 appearances in winning combinations

### First-Mover (Blue) Advantage Analysis
- Blue wins 50.0% of games (expected: 50%)
- Advantage magnitude: **none**
