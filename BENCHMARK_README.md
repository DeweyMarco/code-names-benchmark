# Codenames Benchmark Suite

This directory contains comprehensive benchmarking tools to test AI model performance in Codenames games.

## Overview

The benchmark suite tests the following models:
- **GPT-5** (OpenAI)
- **Gemini 2.5 Pro** (Google)
- **Claude Haiku 4.5** (Anthropic)
- **DeepSeek Reasoner** (DeepSeek)
- **Grok 4** (xAI)

Each model is tested in both roles:
- **Hint Giver (Spymaster)**: Gives hints to help their team find words
- **Guesser (Field Operative)**: Interprets hints and makes guesses

## Scripts

### 1. Quick Benchmark (`quick_benchmark.py`)
**Recommended for initial testing**

- Tests focused model combinations (~50-100 games)
- Takes 30-60 minutes to complete
- Provides quick insights into model performance

```bash
python quick_benchmark.py
```

### 2. Comprehensive Benchmark (`comprehensive_benchmark.py`)
**Full analysis of all combinations**

- Tests all possible model combinations (625 combinations)
- Takes several hours to complete
- Provides complete statistical analysis

```bash
python comprehensive_benchmark.py
```

### 3. Results Analysis (`analyze_benchmark_results.py`)
**Analyze benchmark results**

- Generates detailed performance reports
- Creates visualizations
- Identifies best model combinations

```bash
python analyze_benchmark_results.py <results_file.json>
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Keys
Create a `.env` file with your API keys:

```env
# OpenAI (for GPT-5)
OPENAI_API_KEY=your_openai_key_here

# Google (for Gemini 2.5 Pro)
GOOGLE_API_KEY=your_google_key_here

# Anthropic (for Claude Haiku 4.5)
ANTHROPIC_API_KEY=your_anthropic_key_here

# DeepSeek (for DeepSeek Reasoner)
DEEPSEEK_API_KEY=your_deepseek_key_here

# xAI (for Grok 4)
XAI_API_KEY=your_xai_key_here
```

### 3. Run Benchmark
```bash
# Quick benchmark (recommended first)
python quick_benchmark.py

# Or comprehensive benchmark
python comprehensive_benchmark.py
```

## Understanding Results

### Key Metrics

1. **Win Rate**: Percentage of games won by each team
2. **Hint Success Rate**: How often hints lead to correct guesses
3. **Guess Accuracy**: Percentage of correct guesses made
4. **Turn Efficiency**: Average turns needed to win games
5. **Model Synergies**: Which model combinations work best together

### Output Files

- `quick_benchmark_results/` or `benchmark_results/`: Contains JSON results
- `analysis_plots/`: Visualization charts
- `*_insights.md`: Detailed analysis reports

### Sample Analysis

The analysis will show:
- **Best Hint Givers**: Models that give the most effective hints
- **Best Guessers**: Models that interpret hints most accurately
- **Dominant Combinations**: Team pairings with highest win rates
- **Strategic Insights**: Which models work best together

## Example Workflow

1. **Start with Quick Benchmark**:
   ```bash
   python quick_benchmark.py
   ```

2. **Analyze Results**:
   ```bash
   python analyze_benchmark_results.py quick_benchmark_results/quick_20241201_143022_quick.json
   ```

3. **Run Full Benchmark** (if needed):
   ```bash
   python comprehensive_benchmark.py
   ```

## Customization

### Modify Models
Edit the `BENCHMARK_MODELS` list in the benchmark scripts to test different models.

### Adjust Game Count
Change `GAMES_PER_COMBINATION` to run more or fewer games per combination.

### Add Analysis
Extend the analysis functions to include additional metrics or visualizations.

## Troubleshooting

### API Key Issues
- Ensure all required API keys are set in `.env`
- Check that keys have sufficient credits/quota
- Verify keys are valid and active

### Memory Issues
- Reduce `GAMES_PER_COMBINATION` for large benchmarks
- Use the quick benchmark for initial testing

### Performance
- Individual games take 1-3 minutes each
- Quick benchmark: ~50-100 games (1-3 hours)
- Comprehensive benchmark: ~1,875 games (30-60 hours)

## Expected Results

Based on typical performance patterns:

- **GPT-5**: Strong reasoning, good at both hint giving and guessing
- **Gemini 2.5 Pro**: Excellent at understanding context and relationships
- **Claude Haiku 4.5**: Fast and efficient, good balance of skills
- **DeepSeek Reasoner**: Strong analytical capabilities
- **Grok 4**: Creative and unconventional approaches

The benchmark will reveal which models excel in specific roles and which combinations are most effective.
