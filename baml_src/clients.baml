// LLM Client Configurations for Codenames Agents
// Define all the LLM providers and models used in the benchmark

// ============================================================================
// OPENAI CLIENTS
// ============================================================================

// GPT-5 Series (Latest - August 2025)
client<llm> GPT5 {
  provider openai
  options {
    model "gpt-5"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> GPT5Mini {
  provider openai
  options {
    model "gpt-5-mini"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> GPT5Nano {
  provider openai
  options {
    model "gpt-5-nano"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> GPT5Chat {
  provider openai
  options {
    model "gpt-5-chat-latest"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> GPT5Pro {
  provider openai
  options {
    model "gpt-5-pro"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

// GPT-4.1 Series (April 2025)
client<llm> GPT41 {
  provider openai
  options {
    model "gpt-4.1"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> GPT41Mini {
  provider openai
  options {
    model "gpt-4.1-mini"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> GPT41Nano {
  provider openai
  options {
    model "gpt-4.1-nano"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

// Reasoning Models (o-series)
client<llm> O4Mini {
  provider openai
  options {
    model "o4-mini"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> O3Mini {
  provider openai
  options {
    model "o3-mini"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> O3 {
  provider openai
  options {
    model "o3"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> O1 {
  provider openai
  options {
    model "o1"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> O1Mini {
  provider openai
  options {
    model "o1-mini"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> O1Preview {
  provider openai
  options {
    model "o1-preview"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

// GPT-4o Series (Multimodal)
client<llm> GPT4o {
  provider openai
  options {
    model "gpt-4o"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> GPT4oMini {
  provider openai
  options {
    model "gpt-4o-mini"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> GPT4o_20240806 {
  provider openai
  options {
    model "gpt-4o-2024-08-06"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> GPT4oMini_20240718 {
  provider openai
  options {
    model "gpt-4o-mini-2024-07-18"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

// GPT-4 Turbo Series
client<llm> GPT4Turbo {
  provider openai
  options {
    model "gpt-4-turbo"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> GPT4TurboPreview {
  provider openai
  options {
    model "gpt-4-turbo-preview"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> GPT4_0125Preview {
  provider openai
  options {
    model "gpt-4-0125-preview"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> GPT4_1106Preview {
  provider openai
  options {
    model "gpt-4-1106-preview"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

// GPT-4 Base Series
client<llm> GPT4 {
  provider openai
  options {
    model "gpt-4"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> GPT4_32k {
  provider openai
  options {
    model "gpt-4-32k"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> GPT4_0613 {
  provider openai
  options {
    model "gpt-4-0613"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

// GPT-3.5 Turbo Series
client<llm> GPT35Turbo {
  provider openai
  options {
    model "gpt-3.5-turbo"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> GPT35Turbo16k {
  provider openai
  options {
    model "gpt-3.5-turbo-16k"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

client<llm> GPT35TurboInstruct {
  provider openai
  options {
    model "gpt-3.5-turbo-instruct"
    api_key env.OPENAI_API_KEY
    temperature 0.7
  }
}

// ============================================================================
// ANTHROPIC CLIENTS (CLAUDE)
// ============================================================================

// Claude 4.5 Series (Latest - October 2025)
client<llm> ClaudeSonnet45 {
  provider anthropic
  options {
    model "claude-sonnet-4-5-20250929"
    api_key env.ANTHROPIC_API_KEY
    temperature 0.7
    max_tokens 1024
  }
}

client<llm> ClaudeHaiku45 {
  provider anthropic
  options {
    model "claude-haiku-4-5-20251001"
    api_key env.ANTHROPIC_API_KEY
    temperature 0.7
    max_tokens 1024
  }
}

// Claude 4.1 Series
client<llm> ClaudeOpus41 {
  provider anthropic
  options {
    model "claude-opus-4-1-20250805"
    api_key env.ANTHROPIC_API_KEY
    temperature 0.7
    max_tokens 1024
  }
}

// Claude 4 Series (Legacy - May 2025)
client<llm> ClaudeSonnet4 {
  provider anthropic
  options {
    model "claude-sonnet-4-20250514"
    api_key env.ANTHROPIC_API_KEY
    temperature 0.7
    max_tokens 1024
  }
}

client<llm> ClaudeOpus4 {
  provider anthropic
  options {
    model "claude-opus-4-20250514"
    api_key env.ANTHROPIC_API_KEY
    temperature 0.7
    max_tokens 1024
  }
}

// Claude 3.7 Series (Legacy - February 2025)
client<llm> ClaudeSonnet37 {
  provider anthropic
  options {
    model "claude-3-7-sonnet-20250219"
    api_key env.ANTHROPIC_API_KEY
    temperature 0.7
    max_tokens 1024
  }
}

// Claude 3.5 Series (Legacy)
client<llm> ClaudeHaiku35 {
  provider anthropic
  options {
    model "claude-3-5-haiku-20241022"
    api_key env.ANTHROPIC_API_KEY
    temperature 0.7
    max_tokens 1024
  }
}

// Claude 3 Series (Legacy)
client<llm> ClaudeHaiku3 {
  provider anthropic
  options {
    model "claude-3-haiku-20240307"
    api_key env.ANTHROPIC_API_KEY
    temperature 0.7
    max_tokens 1024
  }
}

// ============================================================================
// GOOGLE CLIENTS (GEMINI)
// ============================================================================

// Gemini 2.5 Series (Latest - January 2025)
client<llm> Gemini25Pro {
  provider google-ai
  options {
    model "gemini-2.5-pro"
    api_key env.GOOGLE_API_KEY
  }
}

client<llm> Gemini25Flash {
  provider google-ai
  options {
    model "gemini-2.5-flash"
    api_key env.GOOGLE_API_KEY
  }
}

client<llm> Gemini25FlashLite {
  provider google-ai
  options {
    model "gemini-2.5-flash-lite"
    api_key env.GOOGLE_API_KEY
  }
}

// Gemini 2.0 Series
client<llm> Gemini20Flash {
  provider google-ai
  options {
    model "gemini-2.0-flash"
    api_key env.GOOGLE_API_KEY
  }
}

client<llm> Gemini20FlashLite {
  provider google-ai
  options {
    model "gemini-2.0-flash-lite"
    api_key env.GOOGLE_API_KEY
  }
}

// ============================================================================
// CUSTOM PROVIDERS (for DeepSeek, Grok, Llama via OpenAI-compatible APIs)
// ============================================================================

// DeepSeek V3.2-Exp - Non-thinking mode (128K context)
client<llm> DeepSeekChat {
  provider openai
  options {
    model "deepseek-chat"
    api_key env.DEEPSEEK_API_KEY
    base_url "https://api.deepseek.com"
    temperature 0.7
  }
}

// DeepSeek V3.2-Exp - Thinking/reasoning mode (128K context)
client<llm> DeepSeekReasoner {
  provider openai
  options {
    model "deepseek-reasoner"
    api_key env.DEEPSEEK_API_KEY
    base_url "https://api.deepseek.com"
    temperature 0.7
  }
}

// ============================================================================
// XAI CLIENTS (GROK)
// ============================================================================

// Grok 4 - Latest flagship reasoning model (256K context)
client<llm> Grok4 {
  provider openai
  options {
    model "grok-4-0709"
    api_key env.XAI_API_KEY
    base_url "https://api.x.ai/v1"
    temperature 0.7
  }
}

// Grok 4 Fast - Fast reasoning model (2M context)
client<llm> Grok4FastReasoning {
  provider openai
  options {
    model "grok-4-fast-reasoning"
    api_key env.XAI_API_KEY
    base_url "https://api.x.ai/v1"
    temperature 0.7
  }
}

// Grok 4 Fast Non-Reasoning - Fast model without reasoning (2M context)
client<llm> Grok4FastNonReasoning {
  provider openai
  options {
    model "grok-4-fast-non-reasoning"
    api_key env.XAI_API_KEY
    base_url "https://api.x.ai/v1"
    temperature 0.7
  }
}

// Grok 3 Beta - Flagship model for enterprise tasks (131K context)
client<llm> Grok3 {
  provider openai
  options {
    model "grok-3-beta"
    api_key env.XAI_API_KEY
    base_url "https://api.x.ai/v1"
    temperature 0.7
  }
}

// Grok 3 Fast Beta - Fastest flagship model (131K context)
client<llm> Grok3Fast {
  provider openai
  options {
    model "grok-3-fast-beta"
    api_key env.XAI_API_KEY
    base_url "https://api.x.ai/v1"
    temperature 0.7
  }
}

// Grok 3 Mini Beta - Smaller model for basic tasks (32K context)
client<llm> Grok3Mini {
  provider openai
  options {
    model "grok-3-mini-beta"
    api_key env.XAI_API_KEY
    base_url "https://api.x.ai/v1"
    temperature 0.7
  }
}

// Grok 3 Mini Fast Beta - Faster mini model (32K context)
client<llm> Grok3MiniFast {
  provider openai
  options {
    model "grok-3-mini-fast-beta"
    api_key env.XAI_API_KEY
    base_url "https://api.x.ai/v1"
    temperature 0.7
  }
}

// Llama via OpenAI-compatible endpoint (e.g., via Together AI or Replicate)
client<llm> Llama {
  provider openai
  options {
    model "meta-llama/Llama-3-70b-chat-hf"
    api_key env.TOGETHER_API_KEY
    base_url "https://api.together.xyz/v1"
    temperature 0.7
  }
}
