# Codenames Agents

Agent interfaces and implementations for playing Codenames.

## Agent Interface

### `HintGiver` (Spymaster)

Sees all word colors and gives hints to guide the guesser.

```python
class HintGiver(ABC):
    @abstractmethod
    def give_hint(
        self,
        my_words: List[str],          # Your team's unrevealed words
        opponent_words: List[str],     # Opponent's unrevealed words
        neutral_words: List[str],      # Neutral unrevealed words
        bomb_words: List[str],         # Bomb words (if not revealed)
        revealed_words: List[str],     # Already guessed words
        board_words: List[str]         # All words on board
    ) -> HintResponse:                 # (word, count)
        pass
```

**Rules:** Single word (no board words), count = how many words it relates to

### `Guesser` (Field Operative)

Only sees words (not colors) and guesses based on hints.

```python
class Guesser(ABC):
    @abstractmethod
    def make_guesses(
        self,
        hint_word: str,               # The hint word
        hint_count: int,              # How many words it relates to
        board_words: List[str],       # All words on board
        revealed_words: List[str]     # Already revealed words
    ) -> List[str]:                   # Words to guess (in order)
        pass
```

**Rules:** Can guess 0 to (hint_count + 1) words. Stops at first wrong guess.

## Implementations

### Random Agents

Simple agents for testing without LLM costs:

```python
from agents.random_agents import RandomHintGiver, RandomGuesser
from game import Team

hint_giver = RandomHintGiver(Team.BLUE)
guesser = RandomGuesser(Team.BLUE)
```

### BAML Agents (Recommended)

Universal LLM agents powered by BAML - work with any provider:

```python
from agents.llm import BAMLHintGiver, BAMLGuesser, BAMLModel

# Create agents with any model
hint_giver = BAMLHintGiver(Team.BLUE, model=BAMLModel.GPT4O_MINI)
guesser = BAMLGuesser(Team.RED, model=BAMLModel.CLAUDE_SONNET_45)
# See full list of available models below
```

**Available models:**
- **OpenAI GPT-5**: `GPT5`, `GPT5_MINI`, `GPT5_NANO`, `GPT5_CHAT`, `GPT5_PRO`
- **OpenAI GPT-4.1**: `GPT41`, `GPT41_MINI`, `GPT41_NANO`
- **OpenAI Reasoning**: `O4_MINI`, `O3_MINI`, `O3`, `O1`, `O1_MINI`, `O1_PREVIEW`
- **OpenAI GPT-4o**: `GPT4O`, `GPT4O_MINI`, `GPT4O_20240806`, `GPT4O_MINI_20240718`
- **OpenAI GPT-4**: `GPT4_TURBO`, `GPT4_TURBO_PREVIEW`, `GPT4`, `GPT4_32K`, `GPT4_0613`
- **OpenAI GPT-3.5**: `GPT35_TURBO`, `GPT35_TURBO_16K`, `GPT35_TURBO_INSTRUCT`
- **Anthropic Claude 4.5**: `CLAUDE_SONNET_45`, `CLAUDE_HAIKU_45`
- **Anthropic Claude 4**: `CLAUDE_OPUS_41`, `CLAUDE_SONNET_4`, `CLAUDE_OPUS_4`
- **Anthropic Claude 3**: `CLAUDE_SONNET_37`, `CLAUDE_HAIKU_35`, `CLAUDE_HAIKU_3`
- **Google Gemini 2.5**: `GEMINI_25_PRO`, `GEMINI_25_FLASH`, `GEMINI_25_FLASH_LITE`
- **Google Gemini 2.0**: `GEMINI_20_FLASH`, `GEMINI_20_FLASH_LITE`
- **DeepSeek**: `DEEPSEEK_CHAT`, `DEEPSEEK_REASONER`
- **xAI Grok 4**: `GROK4`, `GROK4_FAST_REASONING`, `GROK4_FAST_NON_REASONING`
- **xAI Grok 3**: `GROK3`, `GROK3_FAST`, `GROK3_MINI`, `GROK3_MINI_FAST`
- **Meta**: `LLAMA`
- **OpenRouter (Free!)**: `OPENROUTER_DEVSTRAL`, `OPENROUTER_MIMO_V2_FLASH`, `OPENROUTER_NEMOTRON_NANO`, `OPENROUTER_DEEPSEEK_R1T_CHIMERA`, `OPENROUTER_DEEPSEEK_R1T2_CHIMERA`, `OPENROUTER_GLM_45_AIR`, `OPENROUTER_LLAMA_33_70B`, `OPENROUTER_QWEN3_235B`

**Benefits:**
- Type-safe structured outputs with automatic validation
- Single agent file for all providers
- Declarative prompts in `.baml` files
- Interactive testing playground
- Automatic retries on malformed JSON

### Customizing Prompts

Edit prompts in `baml_src/main.baml`:

```baml
function GiveHint(...) -> HintResponse {
  client GPT4oMini

  prompt #"
    You are playing Codenames as the {{ team | upper }} team's spymaster.

    // Add custom strategy instructions here

    {{ ctx.output_format }}
  "#
}
```

Then regenerate: `baml generate`

### Factory Functions

For easier benchmarking:

```python
from agents.llm import create_hint_giver, create_guesser

hint_giver = create_hint_giver("openai", "gpt-4o", Team.BLUE)
guesser = create_guesser("anthropic", team=Team.RED)
```

## Design Notes

**Why separate hint giver and guesser?**
- Different information access (colors vs words only)
- Tests different skills (association vs interpretation)
- Tests coordination between two LLMs

**Why `process_result()` for Guesser?**
- Optional feedback loop for learning within a game
- Can adjust strategy based on previous guess results

## More Information

See the main README.md and `baml_src/` directory for BAML configuration details.

