"""
LLM-based agents for Codenames.

All agents now use BAML for structured outputs and prompt management.
"""
from .baml_agents import (
    BAMLHintGiver,
    BAMLGuesser,
    BAMLModel,
    create_hint_giver,
    create_guesser
)

__all__ = [
    'BAMLHintGiver',
    'BAMLGuesser',
    'BAMLModel',
    'create_hint_giver',
    'create_guesser',
]

