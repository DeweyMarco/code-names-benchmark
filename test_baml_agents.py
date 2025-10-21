"""
Comprehensive test script for BAML agents.

This tests all available LLM models through BAML integration,
attempting to generate hints for each model and reporting results.
"""
import os
from dotenv import load_dotenv
from baml_client.baml_client.sync_client import b
from baml_py import ClientRegistry
from typing import List, Tuple

# Load environment variables
load_dotenv()

# Define all models to test
MODELS_TO_TEST = {
    "OpenAI": [
        # GPT-5 Series
        ("GPT5", "gpt-5", "OPENAI_API_KEY"),
        ("GPT5Mini", "gpt-5-mini", "OPENAI_API_KEY"),
        ("GPT5Nano", "gpt-5-nano", "OPENAI_API_KEY"),
        ("GPT5Chat", "gpt-5-chat-latest", "OPENAI_API_KEY"),
        ("GPT5Pro", "gpt-5-pro", "OPENAI_API_KEY"),
        # GPT-4.1 Series
        ("GPT41", "gpt-4.1", "OPENAI_API_KEY"),
        ("GPT41Mini", "gpt-4.1-mini", "OPENAI_API_KEY"),
        ("GPT41Nano", "gpt-4.1-nano", "OPENAI_API_KEY"),
        # Reasoning Models
        ("O4Mini", "o4-mini", "OPENAI_API_KEY"),
        ("O3Mini", "o3-mini", "OPENAI_API_KEY"),
        ("O3", "o3", "OPENAI_API_KEY"),
        ("O1", "o1", "OPENAI_API_KEY"),
        ("O1Mini", "o1-mini", "OPENAI_API_KEY"),
        ("O1Preview", "o1-preview", "OPENAI_API_KEY"),
        # GPT-4o Series
        ("GPT4o", "gpt-4o", "OPENAI_API_KEY"),
        ("GPT4oMini", "gpt-4o-mini", "OPENAI_API_KEY"),
        ("GPT4o_20240806", "gpt-4o-2024-08-06", "OPENAI_API_KEY"),
        ("GPT4oMini_20240718", "gpt-4o-mini-2024-07-18", "OPENAI_API_KEY"),
        # GPT-4 Turbo
        ("GPT4Turbo", "gpt-4-turbo", "OPENAI_API_KEY"),
        ("GPT4TurboPreview", "gpt-4-turbo-preview", "OPENAI_API_KEY"),
        ("GPT4_0125Preview", "gpt-4-0125-preview", "OPENAI_API_KEY"),
        ("GPT4_1106Preview", "gpt-4-1106-preview", "OPENAI_API_KEY"),
        # GPT-4 Base
        ("GPT4", "gpt-4", "OPENAI_API_KEY"),
        ("GPT4_32k", "gpt-4-32k", "OPENAI_API_KEY"),
        ("GPT4_0613", "gpt-4-0613", "OPENAI_API_KEY"),
        # GPT-3.5
        ("GPT35Turbo", "gpt-3.5-turbo", "OPENAI_API_KEY"),
        ("GPT35Turbo16k", "gpt-3.5-turbo-16k", "OPENAI_API_KEY"),
        ("GPT35TurboInstruct", "gpt-3.5-turbo-instruct", "OPENAI_API_KEY"),
    ],
    "Anthropic": [
        # Claude 4.5 Series
        ("ClaudeSonnet45", "claude-sonnet-4-5-20250929", "ANTHROPIC_API_KEY"),
        ("ClaudeHaiku45", "claude-haiku-4-5-20251001", "ANTHROPIC_API_KEY"),
        # Claude 4.1 Series
        ("ClaudeOpus41", "claude-opus-4-1-20250805", "ANTHROPIC_API_KEY"),
        # Claude 4 Series
        ("ClaudeSonnet4", "claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"),
        ("ClaudeOpus4", "claude-opus-4-20250514", "ANTHROPIC_API_KEY"),
        # Claude 3.7 Series
        ("ClaudeSonnet37", "claude-3-7-sonnet-20250219", "ANTHROPIC_API_KEY"),
        # Claude 3.5 Series
        ("ClaudeHaiku35", "claude-3-5-haiku-20241022", "ANTHROPIC_API_KEY"),
        # Claude 3 Series
        ("ClaudeHaiku3", "claude-3-haiku-20240307", "ANTHROPIC_API_KEY"),
    ],
    "Google": [
        # Gemini 2.5 Series
        ("Gemini25Pro", "gemini-2.5-pro", "GOOGLE_API_KEY"),
        ("Gemini25Flash", "gemini-2.5-flash", "GOOGLE_API_KEY"),
        ("Gemini25FlashLite", "gemini-2.5-flash-lite", "GOOGLE_API_KEY"),
        # Gemini 2.0 Series
        ("Gemini20Flash", "gemini-2.0-flash", "GOOGLE_API_KEY"),
        ("Gemini20FlashLite", "gemini-2.0-flash-lite", "GOOGLE_API_KEY"),
    ],
    "DeepSeek": [
        ("DeepSeekChat", "deepseek-chat", "DEEPSEEK_API_KEY", "https://api.deepseek.com"),
        ("DeepSeekReasoner", "deepseek-reasoner", "DEEPSEEK_API_KEY", "https://api.deepseek.com"),
    ],
    "XAI": [
        ("Grok4", "grok-4-0709", "XAI_API_KEY", "https://api.x.ai/v1"),
        ("Grok4FastReasoning", "grok-4-fast-reasoning", "XAI_API_KEY", "https://api.x.ai/v1"),
        ("Grok4FastNonReasoning", "grok-4-fast-non-reasoning", "XAI_API_KEY", "https://api.x.ai/v1"),
        ("Grok3", "grok-3-beta", "XAI_API_KEY", "https://api.x.ai/v1"),
        ("Grok3Fast", "grok-3-fast-beta", "XAI_API_KEY", "https://api.x.ai/v1"),
        ("Grok3Mini", "grok-3-mini-beta", "XAI_API_KEY", "https://api.x.ai/v1"),
        ("Grok3MiniFast", "grok-3-mini-fast-beta", "XAI_API_KEY", "https://api.x.ai/v1"),
    ],
    "Llama": [
        ("Llama", "meta-llama/Llama-3-70b-chat-hf", "TOGETHER_API_KEY", "https://api.together.xyz/v1"),
    ],
}


def test_model(client_name: str, model_name: str, api_key_env: str, base_url: str = None):
    """Test a single model through BAML."""
    # Check if API key exists
    if not os.getenv(api_key_env):
        return "SKIP", f"Missing {api_key_env}"

    try:
        # Determine provider based on model type
        if "claude" in model_name.lower():
            provider = "anthropic"
            options = {
                "model": model_name,
                "api_key": os.getenv(api_key_env),
                "temperature": 0.7,
                "max_tokens": 1024,
            }
        elif "gemini" in model_name.lower():
            provider = "google-ai"
            options = {
                "model": model_name,
                "api_key": os.getenv(api_key_env),
            }
        else:  # OpenAI-compatible
            provider = "openai"
            options = {
                "model": model_name,
                "api_key": os.getenv(api_key_env),
                "temperature": 0.7,
            }
            if base_url:
                options["base_url"] = base_url

        # Create a dynamic client
        cr = ClientRegistry()
        cr.add_llm_client(client_name, provider, options)
        cr.set_primary(client_name)

        # Test the model
        result = b.GiveHint(
            team="blue",
            my_words=["dog", "cat", "bird"],
            opponent_words=["car", "tree", "house"],
            neutral_words=["book", "phone"],
            bomb_word="explosion",
            revealed_words=[],
            baml_options={"client_registry": cr}
        )

        return "OK", f"Hint: {result.word} ({result.count})"

    except Exception as e:
        error_msg = str(e)
        if "BamlClientHttpError" in error_msg:
            # Extract just the error type
            return "FAILED", f"BamlClientHttpError(...)"
        return "ERROR", error_msg[:100]


def main():
    """Test all models."""
    print("\n" + "=" * 80)
    print("BAML Model Connectivity Test - Testing All Models")
    print("=" * 80)

    # Check all API keys
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "XAI_API_KEY": os.getenv("XAI_API_KEY"),
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
        "TOGETHER_API_KEY": os.getenv("TOGETHER_API_KEY"),
    }

    print("\nAPI Key Status:")
    for key_name, key_value in api_keys.items():
        if key_value:
            print(f"  ✓ {key_name}")
        else:
            print(f"  ✗ {key_name} (missing)")

    # Test all models
    total = 0
    passed = 0
    failed = 0
    skipped = 0

    for provider_name, models in MODELS_TO_TEST.items():
        print(f"\n{provider_name} Models:")
        print("-" * 40)

        for model_info in models:
            total += 1
            if len(model_info) == 3:
                client_name, model_name, api_key_env = model_info
                base_url = None
            else:
                client_name, model_name, api_key_env, base_url = model_info

            status, message = test_model(client_name, model_name, api_key_env, base_url)

            if status == "OK":
                print(f"✓ {client_name:25} {message}")
                passed += 1
            elif status == "SKIP":
                print(f"⊘ {client_name:25} {message}")
                skipped += 1
            else:
                print(f"✗ {client_name:25} {message}")
                failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"Total models tested: {total}")
    print(f"  ✓ Passed:  {passed}")
    print(f"  ✗ Failed:  {failed}")
    print(f"  ⊘ Skipped: {skipped}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()