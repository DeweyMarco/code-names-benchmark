# Codenames AI Benchmark

An AI benchmark where 4 language models play Codenames: two hint givers (spymasters) and two guessers (field operatives) competing as red and blue teams.

## Features

- **BAML-powered agents** - Universal LLM agents with type-safe structured outputs
- **Multiple providers** - OpenAI, Anthropic, Google Gemini, xAI Grok, DeepSeek, OpenRouter
- **Interactive playground** - Test prompts in VSCode without running full games
- **Configurable** - Adjust board size, rules, validation, and more
- **Benchmark suite** - Run multiple games and collect statistics

## Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys (choose at least one)
cp .env.example .env
# Edit .env with your API keys
```

### Run a Demo Game

```bash
# Run complete game demo with configurable AI models
# Edit the Players class in demo_simple_game.py to choose your models
python3 demo_simple_game.py
```

**Note**: The demo requires at least one API key. See the [API Keys](#api-keys) section below.

## API Keys

Get your API key from:
- **OpenRouter**: [openrouter.ai/keys](https://openrouter.ai/keys) (many free models - recommended for testing!)
- **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Anthropic**: [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
- **Google Gemini**: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- **xAI Grok**: [console.x.ai](https://console.x.ai/)
- **DeepSeek**: [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)

## Available Models (December 2025)

### OpenAI Models

```python
from agents.llm import BAMLModel

# GPT-5.2 Series (Latest - December 2025)
BAMLModel.GPT5              # GPT-5
BAMLModel.GPT5_MINI         # GPT-5 Mini
BAMLModel.GPT5_NANO         # GPT-5 Nano
BAMLModel.GPT5_CHAT         # GPT-5.2 Instant (gpt-5.2-chat-latest)
BAMLModel.GPT5_PRO          # GPT-5.2 Pro

# GPT-4.1 Series
BAMLModel.GPT41             # GPT-4.1
BAMLModel.GPT41_MINI        # GPT-4.1 Mini
BAMLModel.GPT41_NANO        # GPT-4.1 Nano

# Reasoning Models (o-series)
BAMLModel.O4_MINI           # o4-mini
BAMLModel.O3_MINI           # o3-mini
BAMLModel.O3                # o3
BAMLModel.O1                # o1
BAMLModel.O1_MINI           # o1-mini

# GPT-4o Series (Still Available)
BAMLModel.GPT4O             # GPT-4o
BAMLModel.GPT4O_MINI        # GPT-4o Mini (cost-effective)
```

### Anthropic Claude Models

```python
# Claude 4.5 Series (Latest)
BAMLModel.CLAUDE_SONNET_45  # Claude Sonnet 4.5 (1M context available)
BAMLModel.CLAUDE_HAIKU_45   # Claude Haiku 4.5 (fast, affordable)

# Claude 4.x Series
BAMLModel.CLAUDE_OPUS_41    # Claude Opus 4.1 (most capable)
BAMLModel.CLAUDE_SONNET_4   # Claude Sonnet 4
BAMLModel.CLAUDE_OPUS_4     # Claude Opus 4

# Claude 3.x Series (Legacy)
BAMLModel.CLAUDE_HAIKU_35   # Claude 3.5 Haiku
BAMLModel.CLAUDE_HAIKU_3    # Claude 3 Haiku
```

### Google Gemini Models

```python
# Gemini 3 Series (Latest - December 2025)
# Note: Add to clients.baml if needed
# gemini-3-flash-preview, gemini-3-pro

# Gemini 2.5 Series
BAMLModel.GEMINI_25_PRO         # Gemini 2.5 Pro (most capable)
BAMLModel.GEMINI_25_FLASH       # Gemini 2.5 Flash
BAMLModel.GEMINI_25_FLASH_LITE  # Gemini 2.5 Flash Lite

# Gemini 2.0 Series
BAMLModel.GEMINI_20_FLASH       # Gemini 2.0 Flash
BAMLModel.GEMINI_20_FLASH_LITE  # Gemini 2.0 Flash Lite (fastest)
```

### xAI Grok Models

```python
# Grok 4 Series
BAMLModel.GROK4                     # Grok 4
BAMLModel.GROK4_FAST_REASONING      # Grok 4 Fast Reasoning
BAMLModel.GROK4_FAST_NON_REASONING  # Grok 4 Fast

# Grok 3 Series
BAMLModel.GROK3             # Grok 3
BAMLModel.GROK3_FAST        # Grok 3 Fast
BAMLModel.GROK3_MINI        # Grok 3 Mini
BAMLModel.GROK3_MINI_FAST   # Grok 3 Mini Fast
```

### DeepSeek Models

```python
# DeepSeek V3.2 (Latest - December 2025)
BAMLModel.DEEPSEEK_CHAT      # DeepSeek V3.2 (non-thinking mode)
BAMLModel.DEEPSEEK_REASONER  # DeepSeek V3.2 (thinking mode)
```

### OpenRouter Free Models

```python
# Free models via OpenRouter (no cost!)
BAMLModel.OPENROUTER_DEVSTRAL              # Devstral
BAMLModel.OPENROUTER_MIMO_V2_FLASH         # MIMO V2 Flash
BAMLModel.OPENROUTER_NEMOTRON_NANO         # Nemotron Nano 12B
BAMLModel.OPENROUTER_DEEPSEEK_R1T_CHIMERA  # DeepSeek R1T Chimera
BAMLModel.OPENROUTER_DEEPSEEK_R1T2_CHIMERA # DeepSeek R1T2 Chimera
BAMLModel.OPENROUTER_GLM_45_AIR            # GLM 4.5 Air
BAMLModel.OPENROUTER_LLAMA_33_70B          # Llama 3.3 70B
BAMLModel.OPENROUTER_OLMO3_32B             # OLMo 3.1 32B
```

## Using BAML Agents

BAML (Boundary ML) provides universal agents that work with any LLM provider:

```python
from game import Team
from agents.llm import BAMLHintGiver, BAMLGuesser, BAMLModel

# Create agents with any model
hint_giver = BAMLHintGiver(Team.BLUE, model=BAMLModel.GPT5_MINI)
guesser = BAMLGuesser(Team.RED, model=BAMLModel.CLAUDE_SONNET_45)

# Mix and match providers!
```

### Why BAML?

- **One agent file** instead of provider-specific implementations
- **Automatic structured outputs** - no manual JSON parsing
- **Interactive playground** - test prompts in VSCode instantly
- **Type-safe** with auto-validation and retries

### Editing Prompts

1. Open `baml_src/main.baml`
2. Edit the `GiveHint` or `MakeGuesses` functions
3. Run `baml generate`
4. Changes take effect immediately!

Or use the BAML VSCode extension for an interactive playground.

## Project Structure

```
code-names-benchmark/
├── baml_src/              # BAML prompt definitions
│   ├── main.baml          # Agent prompts and schemas
│   └── clients.baml       # LLM provider configs
├── baml_client/           # Generated BAML client code
├── game/                  # Core game engine
│   ├── board.py           # Board with configurable word assignments
│   └── state.py           # Game state and turn logic
├── agents/                # Agent interfaces
│   ├── base.py            # Abstract HintGiver/Guesser classes
│   ├── random_agents.py   # Random agents for testing
│   └── llm/               # LLM-based agents
│       └── baml_agents.py # Universal BAML agents
├── orchestrator/          # Game coordination
│   └── game_runner.py     # Coordinates 4 agents through game
├── analysis/              # Benchmark analysis suite
│   ├── metrics/           # Performance metrics modules
│   ├── pipeline.py        # Analysis pipeline
│   ├── viz.py             # Visualization generation
│   └── report.py          # Report generation
├── utils/                 # Utilities
│   ├── generate_words.py  # Word list generation
│   └── words.csv          # Word pool (400+ words)
├── benchmark_results/     # Saved benchmark results
│   └── analysis_plots/    # Generated visualizations
├── config.py              # Game configuration
├── model_config.py        # Model-specific settings
├── benchmark.py           # Run multi-model benchmarks
├── analyze_benchmark_results.py  # Results analysis
└── demo_simple_game.py    # Complete game demo
```

## Configuration

Centralized configuration in `config.py`:

```python
from config import Config, GameConfig

# Standard game (25 words)
config = Config.default()

# Custom variants
custom = Config.custom_game(board_size=49)  # Larger board
mini = Config.custom_game(board_size=9)     # Quick games
```

Model-specific settings in `model_config.py`:
- Temperature configurations per model
- Benchmark model selection
- Display name mappings

## Example Usage

### Run a Complete Game

```python
from utils import generate_word_list
from game import Board, Team
from agents.llm import BAMLHintGiver, BAMLGuesser, BAMLModel
from orchestrator import GameRunner

# Setup
words = generate_word_list(25)
board = Board(words)

# Create agents - mix and match any models!
runner = GameRunner(
    board=board,
    blue_hint_giver=BAMLHintGiver(Team.BLUE, model=BAMLModel.GPT5_MINI),
    blue_guesser=BAMLGuesser(Team.BLUE, model=BAMLModel.GPT5_MINI),
    red_hint_giver=BAMLHintGiver(Team.RED, model=BAMLModel.CLAUDE_SONNET_45),
    red_guesser=BAMLGuesser(Team.RED, model=BAMLModel.CLAUDE_SONNET_45),
    verbose=True
)

result = runner.run()
print(f"Winner: {result.winner}, Turns: {result.total_turns}")
```

## Benchmarking

The benchmark suite tests model combinations to evaluate LLM performance in Codenames games.

### Running Benchmarks

```bash
# Run the benchmark (uses OpenRouter free models by default)
python benchmark.py

# Analyze the results
python analyze_benchmark_results.py benchmark_results/<result_file>.json
```

### Configure Benchmark Models

Edit `model_config.py` to change which models are tested:

```python
def get_benchmark_models() -> list:
    return [
        BAMLModel.GPT5_MINI,
        BAMLModel.CLAUDE_HAIKU_45,
        BAMLModel.GEMINI_25_FLASH,
        BAMLModel.DEEPSEEK_CHAT,
        # Add more models...
    ]
```

Or modify `GAMES_PER_COMBINATION` in `benchmark.py` to adjust games per matchup.

### Benchmark Metrics

The analysis generates:
- **Win Rate**: Percentage of games won by each team
- **Hint Success Rate**: How often hints lead to correct guesses
- **Guess Accuracy**: Percentage of correct guesses made
- **Turn Efficiency**: Average turns needed to win games
- **Model Synergies**: Which model combinations work best together

### Output Files

- `benchmark_results/*.json` - Raw benchmark data
- `benchmark_results/analysis_plots/` - Visualization charts
- `*_insights.md` - Detailed analysis reports

## Cost Management

Approximate costs per game (December 2025 pricing):

| Model | Cost/Game | Best For |
|-------|-----------|----------|
| `OPENROUTER_*` (free tier) | **$0.00** | Free experimentation! |
| `GEMINI_25_FLASH_LITE` | ~$0.001 | Fastest, cost-efficient |
| `DEEPSEEK_CHAT` | ~$0.002 | Most cost-effective |
| `DEEPSEEK_REASONER` | ~$0.002 | Cost-effective with reasoning |
| `GEMINI_25_FLASH` | ~$0.003 | Good price-performance |
| `CLAUDE_HAIKU_45` | ~$0.01 | Fast, affordable Claude |
| `GPT5_MINI` | ~$0.02 | Latest OpenAI, affordable |
| `CLAUDE_SONNET_45` | ~$0.05 | Frontier performance |
| `GPT5` | ~$0.10 | Latest GPT model |
| `CLAUDE_OPUS_41` | ~$0.30 | Most capable Claude |

**Tips:**
- Start with OpenRouter free models for zero-cost experimentation
- GPT-5.2 pricing: $1.75/1M input, $14/1M output (90% discount on cached inputs)
- Claude Sonnet 4.5: $3/1M input, $15/1M output
- Set API spending limits in provider dashboards
- Use random agents first to test setup (free)

## How Codenames Works

- Two teams (red/blue) compete to find their words first
- Each team: 1 spymaster (sees colors) + field operatives (see only words)
- Spymaster gives one-word hint + number
- Operatives guess based on hint
- Game ends when: all team words found (win), bomb hit (lose), or max turns reached

## Additional Documentation

- `game/README.md` - Detailed game mechanics
- `agents/README.md` - Agent interfaces and implementations
- `orchestrator/README.md` - Game coordination details
- `baml_src/` - Prompt definitions and LLM configs

## Troubleshooting

**"No API keys found"** - Copy `.env.example` to `.env` and add your keys

**"ModuleNotFoundError"** - Run `pip install -r requirements.txt`

**"Rate limit exceeded"** - Wait or switch providers

**Slow games** - LLMs take 2-10s per call (normal). Use `verbose=True` to see progress

**BAML errors** - Run `baml generate` to regenerate client code

## Sources

Model information current as of December 2025:
- [OpenAI GPT-5.2 Models](https://platform.openai.com/docs/models)
- [Anthropic Claude Models](https://docs.anthropic.com/en/docs/about-claude/models/overview)
- [Google Gemini Models](https://ai.google.dev/gemini-api/docs/models)
- [xAI Grok Models](https://docs.x.ai/docs/models)
- [DeepSeek V3.2](https://api-docs.deepseek.com/news/news251201)

## License

MIT License - See LICENSE file for details.
