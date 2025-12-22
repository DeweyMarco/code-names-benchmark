# Comprehensive Codenames Benchmark Insights
Generated from 48 games across 24 team combinations

## Executive Summary

- **Blue (First-Mover) Win Rate**: 50.0% (not significant, p=0.5573)
- **Comeback Rate**: 42.6% of games had comebacks
- **Average Game Length**: 7.5 turns

## Elo Rankings
*Skill-based rating system accounting for opponent strength*

1. **OpenRouterNemotronNano**: 1505 combined (HG: 1509, G: 1500, best: hint_giver)
2. **OpenRouterMimoV2Flash**: 1505 combined (HG: 1500, G: 1509, best: guesser)
3. **OpenRouterDevstral**: 1495 combined (HG: 1491, G: 1500, best: guesser)
4. **OpenRouterOLMo3_32B**: 1495 combined (HG: 1500, G: 1491, best: hint_giver)

## Role Versatility Analysis
*Which models perform well in both roles?*

- **OpenRouterDevstral**: 52.1% combined win rate (HG: 62.5%, G: 41.7%, versatility: 0.67, best: hint_giver)
- **OpenRouterMimoV2Flash**: 47.9% combined win rate (HG: 37.5%, G: 58.3%, versatility: 0.64, best: guesser)
- **OpenRouterOLMo3_32B**: 45.8% combined win rate (HG: 54.2%, G: 37.5%, versatility: 0.69, best: hint_giver)
- **OpenRouterNemotronNano**: 33.3% combined win rate (HG: 25.0%, G: 41.7%, versatility: 0.60, best: guesser)

## Statistical Confidence (95% Wilson CI)
*Win rates with confidence intervals - wider CI = less certainty*

- **OpenRouterDevstral** (hint_giver, blue): 75.0% [46.8% - 91.1%] (n=12, CI width: 44.3pp)
- **OpenRouterMimoV2Flash** (guesser, red): 66.7% [39.1% - 86.2%] (n=12, CI width: 47.1pp)
- **OpenRouterMimoV2Flash** (hint_giver, blue): 58.3% [32.0% - 80.7%] (n=12, CI width: 48.7pp)
- **OpenRouterOLMo3_32B** (hint_giver, red): 58.3% [32.0% - 80.7%] (n=12, CI width: 48.7pp)
- **OpenRouterOLMo3_32B** (guesser, blue): 58.3% [32.0% - 80.7%] (n=12, CI width: 48.7pp)
- **OpenRouterDevstral** (hint_giver, red): 50.0% [25.4% - 74.6%] (n=12, CI width: 49.2pp)
- **OpenRouterMimoV2Flash** (guesser, blue): 50.0% [25.4% - 74.6%] (n=12, CI width: 49.2pp)
- **OpenRouterOLMo3_32B** (hint_giver, blue): 50.0% [25.4% - 74.6%] (n=12, CI width: 49.2pp)
- **OpenRouterNemotronNano** (guesser, blue): 50.0% [25.4% - 74.6%] (n=12, CI width: 49.2pp)
- **OpenRouterDevstral** (guesser, blue): 41.7% [19.3% - 68.0%] (n=12, CI width: 48.7pp)

## Hint Giver Efficiency Analysis
*Deep metrics on hint quality and strategy*

| Model | Team | Avg Hint | Yield | Efficiency | Risk Profile | Win Rate |
|-------|------|----------|-------|------------|--------------|----------|
| OpenRouterDevstral | blue | 2.0 | 1.27 | 64.8% | balanced | 75.0% |
| OpenRouterOLMo3_32B | blue | 2.3 | 1.45 | 63.4% | balanced | 50.0% |
| OpenRouterOLMo3_32B | red | 2.1 | 1.22 | 56.8% | balanced | 58.3% |
| OpenRouterDevstral | red | 2.0 | 1.16 | 56.6% | balanced | 50.0% |
| OpenRouterMimoV2Flash | blue | 2.1 | 1.17 | 54.8% | balanced | 58.3% |
| OpenRouterMimoV2Flash | red | 2.2 | 1.12 | 50.0% | balanced | 16.7% |
| OpenRouterNemotronNano | blue | 3.4 | 1.50 | 43.9% | aggressive | 16.7% |
| OpenRouterNemotronNano | red | 3.9 | 1.33 | 34.3% | aggressive | 33.3% |

## Guesser Performance Analysis
*Critical metrics for guesser evaluation*

| Model | Team | 1st Guess | Overall | Bomb Rate | Risk-Adj | Win Rate |
|-------|------|-----------|---------|-----------|----------|----------|
| OpenRouterMimoV2Flash | blue | 87.8% | 69.0% | 1.19% | 65.5% | 50.0% |
| OpenRouterDevstral | blue | 89.4% | 72.7% | 3.03% | 63.6% | 41.7% |
| OpenRouterDevstral | red | 84.1% | 67.1% | 1.18% | 63.5% | 41.7% |
| OpenRouterMimoV2Flash | red | 90.0% | 72.9% | 3.39% | 62.7% | 66.7% |
| OpenRouterNemotronNano | blue | 80.0% | 67.3% | 3.85% | 55.8% | 50.0% |
| OpenRouterOLMo3_32B | blue | 84.1% | 66.2% | 3.90% | 54.5% | 58.3% |
| OpenRouterOLMo3_32B | red | 88.9% | 65.8% | 5.06% | 50.6% | 16.7% |
| OpenRouterNemotronNano | red | 76.2% | 57.7% | 6.41% | 38.5% | 33.3% |

## Error Analysis
*Failure modes and catastrophic errors*

### Errors by Model (Guessers)
- **OpenRouterMimoV2Flash**: 6 total errors (bombs: 3, invalid: 3)
- **OpenRouterOLMo3_32B**: 7 total errors (bombs: 7, invalid: 0)
- **OpenRouterDevstral**: 8 total errors (bombs: 4, invalid: 4)
- **OpenRouterNemotronNano**: 11 total errors (bombs: 9, invalid: 2)

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

1. **OpenRouterDevstral**: 62.5% win rate (15/24 games)
2. **OpenRouterOLMo3_32B**: 54.2% win rate (13/24 games)
3. **OpenRouterMimoV2Flash**: 37.5% win rate (9/24 games)
4. **OpenRouterNemotronNano**: 25.0% win rate (6/24 games)

## Overall Best Guessers
*Aggregated across both Blue and Red teams*

1. **OpenRouterMimoV2Flash**: 58.3% win rate (14/24 games)
2. **OpenRouterDevstral**: 41.7% win rate (10/24 games)
3. **OpenRouterNemotronNano**: 41.7% win rate (10/24 games)
4. **OpenRouterOLMo3_32B**: 37.5% win rate (9/24 games)

## Best Hint Givers (by Team)

- **OpenRouterDevstral** (Blue Hint Giver): 75.0% win rate (9/12 games)
- **OpenRouterOLMo3_32B** (Red Hint Giver): 58.3% win rate (7/12 games)
- **OpenRouterMimoV2Flash** (Blue Hint Giver): 58.3% win rate (7/12 games)
- **OpenRouterDevstral** (Red Hint Giver): 50.0% win rate (6/12 games)
- **OpenRouterOLMo3_32B** (Blue Hint Giver): 50.0% win rate (6/12 games)

## Best Guessers (by Team)

- **OpenRouterMimoV2Flash** (Red Guesser): 66.7% win rate (8/12 games)
- **OpenRouterOLMo3_32B** (Blue Guesser): 58.3% win rate (7/12 games)
- **OpenRouterMimoV2Flash** (Blue Guesser): 50.0% win rate (6/12 games)
- **OpenRouterNemotronNano** (Blue Guesser): 50.0% win rate (6/12 games)
- **OpenRouterDevstral** (Blue Guesser): 41.7% win rate (5/12 games)

## Most Dominant Team Combinations

 1. **OpenRouterDevstral + OpenRouterOLMo3_32B** vs **OpenRouterNemotronNano + OpenRouterMimoV2Flash**
    Blue Win Rate: 100.0% (2 games, 3.0 avg turns)
 2. **OpenRouterDevstral + OpenRouterNemotronNano** vs **OpenRouterMimoV2Flash + OpenRouterOLMo3_32B**
    Blue Win Rate: 100.0% (2 games, 10.0 avg turns)
 3. **OpenRouterDevstral + OpenRouterMimoV2Flash** vs **OpenRouterOLMo3_32B + OpenRouterNemotronNano**
    Blue Win Rate: 100.0% (2 games, 4.0 avg turns)
 4. **OpenRouterDevstral + OpenRouterOLMo3_32B** vs **OpenRouterMimoV2Flash + OpenRouterNemotronNano**
    Blue Win Rate: 100.0% (2 games, 10.0 avg turns)
 5. **OpenRouterMimoV2Flash + OpenRouterDevstral** vs **OpenRouterNemotronNano + OpenRouterOLMo3_32B**
    Blue Win Rate: 100.0% (2 games, 8.5 avg turns)
 6. **OpenRouterOLMo3_32B + OpenRouterNemotronNano** vs **OpenRouterMimoV2Flash + OpenRouterDevstral**
    Blue Win Rate: 100.0% (2 games, 10.0 avg turns)
 7. **OpenRouterDevstral + OpenRouterMimoV2Flash** vs **OpenRouterNemotronNano + OpenRouterOLMo3_32B**
    Blue Win Rate: 50.0% (2 games, 10.5 avg turns)
 8. **OpenRouterMimoV2Flash + OpenRouterOLMo3_32B** vs **OpenRouterNemotronNano + OpenRouterDevstral**
    Blue Win Rate: 50.0% (2 games, 10.0 avg turns)
 9. **OpenRouterMimoV2Flash + OpenRouterNemotronNano** vs **OpenRouterOLMo3_32B + OpenRouterDevstral**
    Blue Win Rate: 50.0% (2 games, 10.5 avg turns)
10. **OpenRouterMimoV2Flash + OpenRouterNemotronNano** vs **OpenRouterDevstral + OpenRouterOLMo3_32B**
    Blue Win Rate: 50.0% (2 games, 9.0 avg turns)

## Best Model Synergies

- **OpenRouterDevstral + OpenRouterOLMo3_32B**: 100.0% win rate (4 games)
- **OpenRouterDevstral + OpenRouterMimoV2Flash**: 75.0% win rate (4 games)
- **OpenRouterMimoV2Flash + OpenRouterDevstral**: 75.0% win rate (4 games)
- **OpenRouterDevstral + OpenRouterNemotronNano**: 50.0% win rate (4 games)
- **OpenRouterMimoV2Flash + OpenRouterNemotronNano**: 50.0% win rate (4 games)
- **OpenRouterMimoV2Flash + OpenRouterOLMo3_32B**: 50.0% win rate (4 games)
- **OpenRouterOLMo3_32B + OpenRouterDevstral**: 50.0% win rate (4 games)
- **OpenRouterOLMo3_32B + OpenRouterMimoV2Flash**: 50.0% win rate (4 games)
- **OpenRouterOLMo3_32B + OpenRouterNemotronNano**: 50.0% win rate (4 games)
- **OpenRouterNemotronNano + OpenRouterMimoV2Flash**: 25.0% win rate (4 games)

## Game Efficiency (Speed of Victory)

- **OpenRouterDevstral**: 62.5% win rate, 7.0 avg turns/game, 11.1 turns/win
- **OpenRouterOLMo3_32B**: 54.2% win rate, 7.3 avg turns/game, 13.5 turns/win
- **OpenRouterMimoV2Flash**: 37.5% win rate, 8.4 avg turns/game, 22.3 turns/win
- **OpenRouterNemotronNano**: 25.0% win rate, 6.7 avg turns/game, 26.7 turns/win

## Strategic Insights

### Models Most Frequently in Winning Combinations:
- **OpenRouterDevstral**: 6 appearances in winning combinations
- **OpenRouterOLMo3_32B**: 6 appearances in winning combinations
- **OpenRouterMimoV2Flash**: 6 appearances in winning combinations
- **OpenRouterNemotronNano**: 6 appearances in winning combinations

### First-Mover (Blue) Advantage Analysis
- Blue wins 50.0% of games (expected: 50%)
- Advantage magnitude: **none**
