"""
Model Configuration Helper

This module provides configuration for different models, including
temperature settings and other model-specific parameters.

Model availability current as of December 2025.
"""

from typing import Dict, Any
from agents.llm import BAMLModel

# Model-specific configurations
MODEL_CONFIGS = {
    # ==========================================================================
    # OpenAI Models
    # ==========================================================================

    # GPT-5.2 Series (December 2025)
    BAMLModel.GPT5: {"temperature": 0.7},
    BAMLModel.GPT5_MINI: {"temperature": 0.7},
    BAMLModel.GPT5_NANO: {"temperature": 0.7},
    BAMLModel.GPT5_CHAT: {"temperature": 0.7},
    BAMLModel.GPT5_PRO: {"temperature": 0.7},

    # GPT-4.1 Series
    BAMLModel.GPT41: {"temperature": 0.7},
    BAMLModel.GPT41_MINI: {"temperature": 0.7},
    BAMLModel.GPT41_NANO: {"temperature": 0.7},

    # Reasoning models (o-series) - requires temperature 1.0
    BAMLModel.O4_MINI: {"temperature": 1.0},
    BAMLModel.O3_MINI: {"temperature": 1.0},
    BAMLModel.O3: {"temperature": 1.0},
    BAMLModel.O1: {"temperature": 1.0},
    BAMLModel.O1_MINI: {"temperature": 1.0},
    BAMLModel.O1_PREVIEW: {"temperature": 1.0},

    # GPT-4o series
    BAMLModel.GPT4O: {"temperature": 0.7},
    BAMLModel.GPT4O_MINI: {"temperature": 0.7},

    # GPT-4 series (legacy)
    BAMLModel.GPT4_TURBO: {"temperature": 0.7},
    BAMLModel.GPT4: {"temperature": 0.7},
    BAMLModel.GPT35_TURBO: {"temperature": 0.7},

    # ==========================================================================
    # Anthropic Claude Models
    # ==========================================================================

    # Claude 4.5 Series (Latest)
    BAMLModel.CLAUDE_SONNET_45: {"temperature": 0.7},
    BAMLModel.CLAUDE_HAIKU_45: {"temperature": 0.7},

    # Claude 4.x Series
    BAMLModel.CLAUDE_OPUS_41: {"temperature": 0.7},
    BAMLModel.CLAUDE_SONNET_4: {"temperature": 0.7},
    BAMLModel.CLAUDE_OPUS_4: {"temperature": 0.7},

    # Claude 3.x Series (Legacy)
    BAMLModel.CLAUDE_SONNET_37: {"temperature": 0.7},
    BAMLModel.CLAUDE_HAIKU_35: {"temperature": 0.7},
    BAMLModel.CLAUDE_HAIKU_3: {"temperature": 0.7},

    # ==========================================================================
    # Google Gemini Models
    # ==========================================================================

    # Gemini 2.5 Series
    BAMLModel.GEMINI_25_PRO: {"temperature": 0.7},
    BAMLModel.GEMINI_25_FLASH: {"temperature": 0.7},
    BAMLModel.GEMINI_25_FLASH_LITE: {"temperature": 0.7},

    # Gemini 2.0 Series
    BAMLModel.GEMINI_20_FLASH: {"temperature": 0.7},
    BAMLModel.GEMINI_20_FLASH_LITE: {"temperature": 0.7},

    # ==========================================================================
    # xAI Grok Models
    # ==========================================================================

    BAMLModel.GROK4: {"temperature": 0.7},
    BAMLModel.GROK4_FAST_REASONING: {"temperature": 0.7},
    BAMLModel.GROK4_FAST_NON_REASONING: {"temperature": 0.7},
    BAMLModel.GROK3: {"temperature": 0.7},
    BAMLModel.GROK3_FAST: {"temperature": 0.7},
    BAMLModel.GROK3_MINI: {"temperature": 0.7},
    BAMLModel.GROK3_MINI_FAST: {"temperature": 0.7},

    # ==========================================================================
    # DeepSeek Models
    # ==========================================================================

    BAMLModel.DEEPSEEK_CHAT: {"temperature": 0.7},
    BAMLModel.DEEPSEEK_REASONER: {"temperature": 0.7},

    # ==========================================================================
    # Other Models
    # ==========================================================================

    BAMLModel.LLAMA: {"temperature": 0.7},

    # OpenRouter - Free Models
    BAMLModel.OPENROUTER_DEVSTRAL: {"temperature": 0.7},
    BAMLModel.OPENROUTER_MIMO_V2_FLASH: {"temperature": 0.7},
    BAMLModel.OPENROUTER_NEMOTRON_NANO: {"temperature": 0.7},
    BAMLModel.OPENROUTER_DEEPSEEK_R1T_CHIMERA: {"temperature": 0.7},
    BAMLModel.OPENROUTER_DEEPSEEK_R1T2_CHIMERA: {"temperature": 0.7},
    BAMLModel.OPENROUTER_GLM_45_AIR: {"temperature": 0.7},
    BAMLModel.OPENROUTER_LLAMA_33_70B: {"temperature": 0.7},
    BAMLModel.OPENROUTER_OLMO3_32B: {"temperature": 0.7},
}

def get_model_config(model: BAMLModel) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    return MODEL_CONFIGS.get(model, {"temperature": 0.7})

def get_temperature(model: BAMLModel) -> float:
    """Get the appropriate temperature for a model."""
    config = get_model_config(model)
    return config.get("temperature", 0.7)

def is_temperature_restricted(model: BAMLModel) -> bool:
    """Check if a model has temperature restrictions."""
    config = get_model_config(model)
    return config.get("temperature", 0.7) == 1.0

# Models that require temperature 1.0 (reasoning models)
RESTRICTED_TEMPERATURE_MODELS = {
    BAMLModel.O4_MINI,
    BAMLModel.O3_MINI,
    BAMLModel.O3,
    BAMLModel.O1,
    BAMLModel.O1_MINI,
    BAMLModel.O1_PREVIEW,
}

def get_benchmark_models() -> list:
    """
    Get the list of models for benchmarking.

    Returns a diverse set of models from different providers that are
    confirmed available as of December 2025.
    """
    return [
        BAMLModel.OPENROUTER_DEVSTRAL,
        BAMLModel.OPENROUTER_MIMO_V2_FLASH,
        BAMLModel.OPENROUTER_NEMOTRON_NANO,
        BAMLModel.OPENROUTER_DEEPSEEK_R1T_CHIMERA,
        BAMLModel.OPENROUTER_DEEPSEEK_R1T2_CHIMERA,
        BAMLModel.OPENROUTER_GLM_45_AIR,
        BAMLModel.OPENROUTER_LLAMA_33_70B,
        BAMLModel.OPENROUTER_OLMO3_32B,
    ]

def get_model_display_name(model: BAMLModel) -> str:
    """Get display name for a model."""
    display_names = {
        # OpenAI GPT-5.2 Series (December 2025)
        BAMLModel.GPT5: "GPT-5",
        BAMLModel.GPT5_MINI: "GPT-5 Mini",
        BAMLModel.GPT5_NANO: "GPT-5 Nano",
        BAMLModel.GPT5_CHAT: "GPT-5.2 Instant",
        BAMLModel.GPT5_PRO: "GPT-5.2 Pro",
        # OpenAI GPT-4.1 Series
        BAMLModel.GPT41: "GPT-4.1",
        BAMLModel.GPT41_MINI: "GPT-4.1 Mini",
        BAMLModel.GPT41_NANO: "GPT-4.1 Nano",
        # OpenAI Reasoning Models
        BAMLModel.O4_MINI: "o4-mini",
        BAMLModel.O3_MINI: "o3-mini",
        BAMLModel.O3: "o3",
        BAMLModel.O1: "o1",
        BAMLModel.O1_MINI: "o1-mini",
        BAMLModel.O1_PREVIEW: "o1-preview",
        # OpenAI GPT-4o Series
        BAMLModel.GPT4O: "GPT-4o",
        BAMLModel.GPT4O_MINI: "GPT-4o Mini",
        BAMLModel.GPT4O_20240806: "GPT-4o (2024-08-06)",
        BAMLModel.GPT4O_MINI_20240718: "GPT-4o Mini (2024-07-18)",
        # OpenAI GPT-4 Series (Legacy)
        BAMLModel.GPT4_TURBO: "GPT-4 Turbo",
        BAMLModel.GPT4_TURBO_PREVIEW: "GPT-4 Turbo Preview",
        BAMLModel.GPT4_0125_PREVIEW: "GPT-4 (0125-preview)",
        BAMLModel.GPT4_1106_PREVIEW: "GPT-4 (1106-preview)",
        BAMLModel.GPT4: "GPT-4",
        BAMLModel.GPT4_32K: "GPT-4 32K",
        BAMLModel.GPT4_0613: "GPT-4 (0613)",
        # OpenAI GPT-3.5 Series (Legacy)
        BAMLModel.GPT35_TURBO: "GPT-3.5 Turbo",
        BAMLModel.GPT35_TURBO_16K: "GPT-3.5 Turbo 16K",
        BAMLModel.GPT35_TURBO_INSTRUCT: "GPT-3.5 Turbo Instruct",
        # Anthropic Claude 4.5 Series (Latest)
        BAMLModel.CLAUDE_SONNET_45: "Claude Sonnet 4.5",
        BAMLModel.CLAUDE_HAIKU_45: "Claude Haiku 4.5",
        # Anthropic Claude 4.x Series
        BAMLModel.CLAUDE_OPUS_41: "Claude Opus 4.1",
        BAMLModel.CLAUDE_SONNET_4: "Claude Sonnet 4",
        BAMLModel.CLAUDE_OPUS_4: "Claude Opus 4",
        # Anthropic Claude 3.x Series (Legacy)
        BAMLModel.CLAUDE_SONNET_37: "Claude Sonnet 3.7",
        BAMLModel.CLAUDE_HAIKU_35: "Claude Haiku 3.5",
        BAMLModel.CLAUDE_HAIKU_3: "Claude 3 Haiku",
        # Google Gemini 2.5 Series
        BAMLModel.GEMINI_25_PRO: "Gemini 2.5 Pro",
        BAMLModel.GEMINI_25_FLASH: "Gemini 2.5 Flash",
        BAMLModel.GEMINI_25_FLASH_LITE: "Gemini 2.5 Flash Lite",
        # Google Gemini 2.0 Series
        BAMLModel.GEMINI_20_FLASH: "Gemini 2.0 Flash",
        BAMLModel.GEMINI_20_FLASH_LITE: "Gemini 2.0 Flash Lite",
        # xAI Grok 4 Series
        BAMLModel.GROK4: "Grok 4",
        BAMLModel.GROK4_FAST_REASONING: "Grok 4 Fast Reasoning",
        BAMLModel.GROK4_FAST_NON_REASONING: "Grok 4 Fast",
        # xAI Grok 3 Series
        BAMLModel.GROK3: "Grok 3",
        BAMLModel.GROK3_FAST: "Grok 3 Fast",
        BAMLModel.GROK3_MINI: "Grok 3 Mini",
        BAMLModel.GROK3_MINI_FAST: "Grok 3 Mini Fast",
        # DeepSeek V3.2 (December 2025)
        BAMLModel.DEEPSEEK_CHAT: "DeepSeek V3.2",
        BAMLModel.DEEPSEEK_REASONER: "DeepSeek V3.2 Reasoner",
        # Meta Llama
        BAMLModel.LLAMA: "Llama 3 70B",
        # OpenRouter Free Models
        BAMLModel.OPENROUTER_DEVSTRAL: "Devstral (OpenRouter)",
        BAMLModel.OPENROUTER_MIMO_V2_FLASH: "MIMO V2 Flash (OpenRouter)",
        BAMLModel.OPENROUTER_NEMOTRON_NANO: "Nemotron Nano 12B (OpenRouter)",
        BAMLModel.OPENROUTER_DEEPSEEK_R1T_CHIMERA: "DeepSeek R1T Chimera (OpenRouter)",
        BAMLModel.OPENROUTER_DEEPSEEK_R1T2_CHIMERA: "DeepSeek R1T2 Chimera (OpenRouter)",
        BAMLModel.OPENROUTER_GLM_45_AIR: "GLM 4.5 Air (OpenRouter)",
        BAMLModel.OPENROUTER_LLAMA_33_70B: "Llama 3.3 70B (OpenRouter)",
        BAMLModel.OPENROUTER_OLMO3_32B: "OLMo 3.1 32B (OpenRouter)",
    }
    return display_names.get(model, model.value)
