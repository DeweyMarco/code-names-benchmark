# Codenames AI Benchmark

An AI benchmark where 4 language models play Codenames: two hint givers (spymasters) and two guessers (field operatives) competing as red and blue teams.

## Features

- **BAML-powered agents** - Universal LLM agents with type-safe structured outputs
- **Multiple providers** - OpenAI, Anthropic, Google Gemini, xAI Grok, DeepSeek, OpenRouter (free models!)
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
- **OpenRouter**: [openrouter.ai/keys](https://openrouter.ai/keys) (many free models!)
- **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Anthropic**: [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
- **Google Gemini**: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) (free tier available!)
- **xAI Grok**: [console.x.ai](https://console.x.ai/)
- **DeepSeek**: [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)

## Using BAML Agents

BAML (Boundary ML) provides universal agents that work with any LLM provider:

```python
from game import Team
from agents.llm import BAMLHintGiver, BAMLGuesser, BAMLModel

# Create agents with any model
hint_giver = BAMLHintGiver(Team.BLUE, model=BAMLModel.GPT4O_MINI)
guesser = BAMLGuesser(Team.RED, model=BAMLModel.CLAUDE_SONNET_45)

# Mix and match providers! Full list of available models:
# OpenAI GPT-5: GPT5, GPT5_MINI, GPT5_NANO, GPT5_CHAT, GPT5_PRO
# OpenAI GPT-4.1: GPT41, GPT41_MINI, GPT41_NANO
# OpenAI Reasoning: O4_MINI, O3_MINI, O3, O1, O1_MINI, O1_PREVIEW
# OpenAI GPT-4o: GPT4O, GPT4O_MINI
# OpenAI GPT-4: GPT4_TURBO, GPT4
# OpenAI GPT-3.5: GPT35_TURBO
# Anthropic Claude 4.5: CLAUDE_SONNET_45, CLAUDE_HAIKU_45
# Anthropic Claude 4: CLAUDE_OPUS_41, CLAUDE_SONNET_4, CLAUDE_OPUS_4
# Anthropic Claude 3: CLAUDE_SONNET_37, CLAUDE_HAIKU_35, CLAUDE_HAIKU_3
# Google Gemini 2.5: GEMINI_25_PRO, GEMINI_25_FLASH, GEMINI_25_FLASH_LITE
# Google Gemini 2.0: GEMINI_20_FLASH, GEMINI_20_FLASH_LITE
# DeepSeek: DEEPSEEK_CHAT, DEEPSEEK_REASONER
# xAI Grok 4: GROK4, GROK4_FAST_REASONING, GROK4_FAST_NON_REASONING
# xAI Grok 3: GROK3, GROK3_FAST, GROK3_MINI, GROK3_MINI_FAST
# Meta: LLAMA
# OpenRouter: OPENROUTER_DEVSTRAL, OPENROUTER_MIMO_V2_FLASH, OPENROUTER_DEEPSEEK_R1T_CHIMERA, OPENROUTER_DEEPSEEK_R1T2_CHIMERA, OPENROUTER_GLM_45_AIR, OPENROUTER_LLAMA_33_70B, OPENROUTER_OLMO3_32B
```

### Why BAML?

- **One agent file** instead of 7 provider-specific files
- **Automatic structured outputs** - no manual JSON parsing
- **Interactive playground** - test prompts in VSCode instantly
- **Type-safe** with auto-validation and retries
- **60% less code** overall

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
│   ├── main.baml         # Agent prompts and schemas
│   └── clients.baml      # LLM provider configs
├── game/                  # Core game engine
│   ├── board.py          # Board with configurable word assignments
│   └── state.py          # Game state and turn logic
├── agents/                # Agent interfaces
│   ├── base.py           # Abstract HintGiver/Guesser classes
│   ├── random_agents.py  # Random agents for testing
│   └── llm/              # LLM-based agents
│       └── baml_agents.py  # Universal BAML agents
├── orchestrator/          # Game coordination
│   └── game_runner.py    # Coordinates 4 agents through game
├── utils/                 # Utilities
│   ├── generate_words.py # Word list generation
│   └── words.csv         # Word pool (400+ words)
├── config.py              # Configuration management
├── benchmark_runner.py    # Run multiple games with statistics
└── demo_llm_game.py       # Complete game demo
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

**Available configs**: GameConfig (board, turns), LLMConfig (models, costs), OrchestratorConfig (logging, validation), DataConfig (paths)

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
    blue_hint_giver=BAMLHintGiver(Team.BLUE, model=BAMLModel.GPT4O_MINI),
    blue_guesser=BAMLGuesser(Team.BLUE, model=BAMLModel.GPT4O_MINI),
    red_hint_giver=BAMLHintGiver(Team.RED, model=BAMLModel.CLAUDE_SONNET_45),
    red_guesser=BAMLGuesser(Team.RED, model=BAMLModel.CLAUDE_SONNET_45),
    verbose=True
)

result = runner.run()
print(f"Winner: {result.winner}, Turns: {result.total_turns}")
```

### Run Benchmarks (Multiple Games)

```python
from benchmark_runner import BenchmarkRunner
from agents.llm import BAMLHintGiver, BAMLGuesser, BAMLModel
from game import Team

# Create benchmark with factory functions
runner = BenchmarkRunner(
    blue_hint_giver_factory=lambda: BAMLHintGiver(Team.BLUE, BAMLModel.GPT4O_MINI),
    blue_guesser_factory=lambda: BAMLGuesser(Team.BLUE, BAMLModel.GPT4O_MINI),
    red_hint_giver_factory=lambda: BAMLHintGiver(Team.RED, BAMLModel.CLAUDE_SONNET_45),
    red_guesser_factory=lambda: BAMLGuesser(Team.RED, BAMLModel.CLAUDE_SONNET_45),
    verbose=True
)

# Run 10 games
result = runner.run(num_games=10)
print(result.stats)  # Win rates, avg turns, performance metrics
```

## Cost Management

Approximate costs per game:

| Model | Cost/Game | Best For |
|-------|-----------|----------|
| `OPENROUTER_*` (free tier) | **$0.00** | Free experimentation! |
| `GEMINI_25_FLASH_LITE` | ~$0.001 | Fastest, cost-efficient |
| `DEEPSEEK_CHAT` | ~$0.002 | Most cost-effective |
| `DEEPSEEK_REASONER` | ~$0.002 | Cost-effective with reasoning |
| `GEMINI_25_FLASH` | ~$0.003 | Best price-performance |
| `CLAUDE_HAIKU_45` | ~$0.01 | Fast, affordable Claude 4.5 |
| `GPT4O_MINI` | ~$0.01 | Production baseline |
| `GEMINI_25_PRO` | ~$0.015 | Advanced reasoning |
| `CLAUDE_SONNET_45` | ~$0.05 | Latest Claude model |
| `GPT4O` | ~$0.20 | Premium performance |
| `CLAUDE_OPUS_41` | ~$0.30 | Most powerful Claude |

**Tips:**
- Start with OpenRouter free models for zero-cost experimentation
- Or try Gemini 2.5 Flash or GPT-4o-mini (cheap)
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

## License

MIT License - See LICENSE file for details.