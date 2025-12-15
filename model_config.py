"""
Model Configuration Helper

This module provides configuration for different models, including
temperature settings and other model-specific parameters.
"""

from typing import Dict, Any
from agents.llm import BAMLModel

# Model-specific configurations
MODEL_CONFIGS = {
    # GPT-5 series - requires temperature 1.0
    BAMLModel.GPT5: {"temperature": 1.0},
    BAMLModel.GPT5_MINI: {"temperature": 1.0},
    BAMLModel.GPT5_NANO: {"temperature": 1.0},
    BAMLModel.GPT5_CHAT: {"temperature": 1.0},
    BAMLModel.GPT5_PRO: {"temperature": 1.0},
    
    # Reasoning models (o-series) - requires temperature 1.0
    BAMLModel.O4_MINI: {"temperature": 1.0},
    BAMLModel.O3_MINI: {"temperature": 1.0},
    BAMLModel.O3: {"temperature": 1.0},
    BAMLModel.O1: {"temperature": 1.0},
    BAMLModel.O1_MINI: {"temperature": 1.0},
    BAMLModel.O1_PREVIEW: {"temperature": 1.0},
    
    # GPT-4 series - supports custom temperature
    BAMLModel.GPT4O: {"temperature": 0.7},
    BAMLModel.GPT4O_MINI: {"temperature": 0.7},
    BAMLModel.GPT4_TURBO: {"temperature": 0.7},
    BAMLModel.GPT4: {"temperature": 0.7},
    BAMLModel.GPT35_TURBO: {"temperature": 0.7},
    
    # Anthropic models - supports custom temperature
    BAMLModel.CLAUDE_SONNET_45: {"temperature": 0.7},
    BAMLModel.CLAUDE_HAIKU_45: {"temperature": 0.7},
    BAMLModel.CLAUDE_OPUS_41: {"temperature": 0.7},
    BAMLModel.CLAUDE_SONNET_4: {"temperature": 0.7},
    BAMLModel.CLAUDE_OPUS_4: {"temperature": 0.7},
    BAMLModel.CLAUDE_SONNET_37: {"temperature": 0.7},
    BAMLModel.CLAUDE_HAIKU_35: {"temperature": 0.7},
    BAMLModel.CLAUDE_HAIKU_3: {"temperature": 0.7},
    
    # Google models - supports custom temperature
    BAMLModel.GEMINI_25_PRO: {"temperature": 0.7},
    BAMLModel.GEMINI_25_FLASH: {"temperature": 0.7},
    BAMLModel.GEMINI_25_FLASH_LITE: {"temperature": 0.7},
    BAMLModel.GEMINI_20_FLASH: {"temperature": 0.7},
    BAMLModel.GEMINI_20_FLASH_LITE: {"temperature": 0.7},
    
    # DeepSeek models - supports custom temperature
    BAMLModel.DEEPSEEK_CHAT: {"temperature": 0.7},
    BAMLModel.DEEPSEEK_REASONER: {"temperature": 0.7},
    
    # xAI Grok models - supports custom temperature
    BAMLModel.GROK4: {"temperature": 0.7},
    BAMLModel.GROK4_FAST_REASONING: {"temperature": 0.7},
    BAMLModel.GROK4_FAST_NON_REASONING: {"temperature": 0.7},
    BAMLModel.GROK3: {"temperature": 0.7},
    BAMLModel.GROK3_FAST: {"temperature": 0.7},
    BAMLModel.GROK3_MINI: {"temperature": 0.7},
    BAMLModel.GROK3_MINI_FAST: {"temperature": 0.7},
    
    # Llama models - supports custom temperature
    BAMLModel.LLAMA: {"temperature": 0.7},
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

# Models that require temperature 1.0
RESTRICTED_TEMPERATURE_MODELS = {
    BAMLModel.GPT5,
    BAMLModel.GPT5_MINI,
    BAMLModel.GPT5_NANO,
    BAMLModel.GPT5_CHAT,
    BAMLModel.GPT5_PRO,
    BAMLModel.O4_MINI,
    BAMLModel.O3_MINI,
    BAMLModel.O3,
    BAMLModel.O1,
    BAMLModel.O1_MINI,
    BAMLModel.O1_PREVIEW,
}

def get_benchmark_models() -> list:
    """Get the list of models for benchmarking."""
    return [
        BAMLModel.GPT5,
        BAMLModel.GEMINI_25_PRO,
        BAMLModel.CLAUDE_HAIKU_45,
        BAMLModel.DEEPSEEK_REASONER,
        BAMLModel.GROK4,
    ]

def get_model_display_name(model: BAMLModel) -> str:
    """Get display name for a model."""
    display_names = {
        BAMLModel.GPT5: "GPT-5",
        BAMLModel.GEMINI_25_PRO: "Gemini 2.5 Pro",
        BAMLModel.CLAUDE_HAIKU_45: "Claude Haiku 4.5",
        BAMLModel.DEEPSEEK_REASONER: "DeepSeek Reasoner",
        BAMLModel.GROK4: "Grok 4",
    }
    return display_names.get(model, model.value)
