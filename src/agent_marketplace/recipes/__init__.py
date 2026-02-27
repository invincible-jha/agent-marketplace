"""Agent composition recipe registry.

Provides pre-built composition patterns that describe how agent capabilities
can be combined into multi-step pipelines.

Classes
-------
AgentRecipe
    A named, versioned description of how to compose capabilities.
RecipeStep
    A single step within a recipe referencing a capability by category or tag.
RecipeRegistry
    In-memory registry for loading, querying, and resolving recipes.
"""
from __future__ import annotations

from agent_marketplace.recipes.recipe_registry import (
    AgentRecipe,
    RecipeRegistry,
    RecipeStep,
)

__all__ = [
    "AgentRecipe",
    "RecipeStep",
    "RecipeRegistry",
]
