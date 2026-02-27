"""Tests for agent_marketplace.recipes.recipe_registry."""
from __future__ import annotations

import threading

import pytest

from agent_marketplace.schema.capability import AgentCapability, CapabilityCategory
from agent_marketplace.schema.provider import ProviderInfo
from agent_marketplace.recipes.recipe_registry import (
    AgentRecipe,
    RecipeRegistry,
    RecipeStep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cap(
    name: str = "test-cap",
    category: CapabilityCategory = CapabilityCategory.ANALYSIS,
    tags: list[str] | None = None,
) -> AgentCapability:
    return AgentCapability(
        name=name,
        version="1.0",
        description="A test capability.",
        category=category,
        input_types=["text/plain"],
        output_type="application/json",
        trust_level=0.8,
        tags=tags or ["test"],
        supported_frameworks=["langchain"],
        provider=ProviderInfo(name="Test Provider"),
    )


def _step(
    step_name: str = "analyse",
    description: str = "Analyse stuff.",
    required_category: str | None = CapabilityCategory.ANALYSIS.value,
    required_tags: tuple[str, ...] = (),
    optional_tags: tuple[str, ...] = (),
    input_from: tuple[str, ...] = (),
    output_key: str = "result",
) -> RecipeStep:
    return RecipeStep(
        step_name=step_name,
        description=description,
        required_category=required_category,
        required_tags=required_tags,
        optional_tags=optional_tags,
        input_from=input_from,
        output_key=output_key,
    )


def _recipe(
    name: str = "test-recipe",
    version: str = "1.0.0",
    steps: tuple[RecipeStep, ...] | None = None,
    tags: tuple[str, ...] = ("test",),
) -> AgentRecipe:
    return AgentRecipe(
        name=name,
        version=version,
        description="A test recipe.",
        steps=steps or (_step(),),
        tags=tags,
    )


# ===========================================================================
# RecipeStep
# ===========================================================================


class TestRecipeStep:
    def test_frozen(self) -> None:
        step = _step()
        with pytest.raises(Exception):
            step.step_name = "other"  # type: ignore[misc]

    def test_matches_by_category(self) -> None:
        step = _step(required_category=CapabilityCategory.ANALYSIS.value)
        cap = _cap(category=CapabilityCategory.ANALYSIS)
        assert step.matches(cap)

    def test_does_not_match_wrong_category(self) -> None:
        step = _step(required_category=CapabilityCategory.GENERATION.value)
        cap = _cap(category=CapabilityCategory.ANALYSIS)
        assert not step.matches(cap)

    def test_matches_no_category_constraint(self) -> None:
        step = _step(required_category=None)
        cap = _cap(category=CapabilityCategory.ANALYSIS)
        assert step.matches(cap)

    def test_matches_with_required_tags(self) -> None:
        step = _step(
            required_category=CapabilityCategory.ANALYSIS.value,
            required_tags=("statistics",),
        )
        cap = _cap(category=CapabilityCategory.ANALYSIS, tags=["statistics", "data"])
        assert step.matches(cap)

    def test_fails_missing_required_tag(self) -> None:
        step = _step(
            required_category=CapabilityCategory.ANALYSIS.value,
            required_tags=("statistics",),
        )
        cap = _cap(category=CapabilityCategory.ANALYSIS, tags=["data"])
        assert not step.matches(cap)

    def test_case_insensitive_tag_match(self) -> None:
        step = _step(
            required_category=None,
            required_tags=("Statistics",),
        )
        cap = _cap(tags=["statistics"])
        assert step.matches(cap)

    def test_case_insensitive_category_match(self) -> None:
        step = _step(required_category="ANALYSIS")
        cap = _cap(category=CapabilityCategory.ANALYSIS)
        assert step.matches(cap)

    def test_multiple_required_tags_all_must_be_present(self) -> None:
        step = _step(required_category=None, required_tags=("a", "b", "c"))
        cap_all = _cap(tags=["a", "b", "c"])
        cap_partial = _cap(tags=["a", "b"])
        assert step.matches(cap_all)
        assert not step.matches(cap_partial)

    def test_to_dict_keys(self) -> None:
        step = _step()
        d = step.to_dict()
        for key in ("step_name", "description", "required_category", "required_tags",
                    "optional_tags", "input_from", "output_key", "metadata"):
            assert key in d

    def test_to_dict_serialisable_types(self) -> None:
        import json
        step = _step(required_tags=("a",), input_from=("prev",))
        d = step.to_dict()
        # Should not raise
        json.dumps(d)


# ===========================================================================
# AgentRecipe
# ===========================================================================


class TestAgentRecipe:
    def test_frozen(self) -> None:
        recipe = _recipe()
        with pytest.raises(Exception):
            recipe.name = "other"  # type: ignore[misc]

    def test_step_count(self) -> None:
        recipe = _recipe(steps=(_step("s1"), _step("s2"), _step("s3")))
        assert recipe.step_count == 3

    def test_step_names(self) -> None:
        recipe = _recipe(steps=(_step("extract"), _step("analyse")))
        assert recipe.step_names == ["extract", "analyse"]

    def test_get_step_found(self) -> None:
        step = _step("extract")
        recipe = _recipe(steps=(step, _step("analyse")))
        result = recipe.get_step("extract")
        assert result is not None
        assert result.step_name == "extract"

    def test_get_step_not_found(self) -> None:
        recipe = _recipe()
        assert recipe.get_step("nonexistent") is None

    def test_to_dict_keys(self) -> None:
        recipe = _recipe()
        d = recipe.to_dict()
        for key in ("name", "version", "description", "steps", "tags",
                    "author", "step_count"):
            assert key in d

    def test_to_dict_steps_is_list(self) -> None:
        recipe = _recipe()
        d = recipe.to_dict()
        assert isinstance(d["steps"], list)

    def test_resolve_returns_matching_caps(self) -> None:
        step = _step(required_category=CapabilityCategory.ANALYSIS.value)
        recipe = _recipe(steps=(step,))
        analysis_cap = _cap(category=CapabilityCategory.ANALYSIS)
        gen_cap = _cap(category=CapabilityCategory.GENERATION, tags=["gen"])
        result = recipe.resolve([analysis_cap, gen_cap])
        assert analysis_cap in result["analyse"]
        assert gen_cap not in result["analyse"]

    def test_resolve_empty_capabilities(self) -> None:
        recipe = _recipe()
        result = recipe.resolve([])
        assert result["analyse"] == []

    def test_resolve_returns_all_steps(self) -> None:
        recipe = _recipe(steps=(_step("s1"), _step("s2")))
        result = recipe.resolve([])
        assert set(result.keys()) == {"s1", "s2"}

    def test_resolve_no_category_matches_all(self) -> None:
        step = _step(required_category=None, required_tags=())
        recipe = _recipe(steps=(step,))
        caps = [
            _cap(category=CapabilityCategory.ANALYSIS),
            _cap(category=CapabilityCategory.GENERATION, tags=["gen"]),
        ]
        result = recipe.resolve(caps)
        assert len(result["analyse"]) == 2


# ===========================================================================
# RecipeRegistry — construction
# ===========================================================================


class TestRecipeRegistryInit:
    def test_empty_registry(self) -> None:
        registry = RecipeRegistry()
        assert registry.recipe_count == 0

    def test_init_with_recipes(self) -> None:
        recipes = [_recipe("r1"), _recipe("r2")]
        registry = RecipeRegistry(recipes)
        assert registry.recipe_count == 2

    def test_with_builtin_recipes_count(self) -> None:
        registry = RecipeRegistry.with_builtin_recipes()
        # Expect exactly 10 built-in recipes
        assert registry.recipe_count == 10

    def test_with_builtin_recipes_names(self) -> None:
        registry = RecipeRegistry.with_builtin_recipes()
        names = registry.list_names()
        expected = {
            "research", "rag", "code-review", "data-pipeline",
            "document-qa", "content-generation", "summarise-and-act",
            "multi-hop-reasoning", "eval-loop", "classify-and-route",
        }
        assert expected.issubset(set(names))


# ===========================================================================
# RecipeRegistry — CRUD
# ===========================================================================


class TestRecipeRegistryCRUD:
    def test_register_and_get(self) -> None:
        registry = RecipeRegistry()
        recipe = _recipe("my-recipe")
        registry.register(recipe)
        retrieved = registry.get("my-recipe")
        assert retrieved.name == "my-recipe"

    def test_get_missing_raises_key_error(self) -> None:
        registry = RecipeRegistry()
        with pytest.raises(KeyError, match="nonexistent"):
            registry.get("nonexistent")

    def test_get_or_none_present(self) -> None:
        registry = RecipeRegistry([_recipe("r1")])
        assert registry.get_or_none("r1") is not None

    def test_get_or_none_absent(self) -> None:
        registry = RecipeRegistry()
        assert registry.get_or_none("absent") is None

    def test_register_replaces_existing(self) -> None:
        registry = RecipeRegistry([_recipe("r1", version="1.0.0")])
        registry.register(_recipe("r1", version="2.0.0"))
        assert registry.get("r1").version == "2.0.0"
        assert registry.recipe_count == 1

    def test_remove_existing(self) -> None:
        registry = RecipeRegistry([_recipe("r1")])
        removed = registry.remove("r1")
        assert removed is True
        assert registry.recipe_count == 0

    def test_remove_nonexistent(self) -> None:
        registry = RecipeRegistry()
        assert registry.remove("nonexistent") is False

    def test_list_names_sorted(self) -> None:
        registry = RecipeRegistry([_recipe("z-recipe"), _recipe("a-recipe"), _recipe("m-recipe")])
        names = registry.list_names()
        assert names == sorted(names)

    def test_list_recipes_sorted(self) -> None:
        registry = RecipeRegistry([_recipe("z"), _recipe("a")])
        recipes = registry.list_recipes()
        assert [r.name for r in recipes] == sorted(r.name for r in recipes)

    def test_recipe_count_updates(self) -> None:
        registry = RecipeRegistry()
        assert registry.recipe_count == 0
        registry.register(_recipe("r1"))
        assert registry.recipe_count == 1
        registry.register(_recipe("r2"))
        assert registry.recipe_count == 2
        registry.remove("r1")
        assert registry.recipe_count == 1


# ===========================================================================
# RecipeRegistry — search_by_tag
# ===========================================================================


class TestSearchByTag:
    def test_finds_matching_recipes(self) -> None:
        r1 = _recipe("r1", tags=("data", "etl"))
        r2 = _recipe("r2", tags=("generation",))
        registry = RecipeRegistry([r1, r2])
        results = registry.search_by_tag("data")
        assert len(results) == 1
        assert results[0].name == "r1"

    def test_case_insensitive(self) -> None:
        recipe = _recipe("r1", tags=("Data",))
        registry = RecipeRegistry([recipe])
        assert len(registry.search_by_tag("data")) == 1

    def test_no_match_returns_empty(self) -> None:
        registry = RecipeRegistry([_recipe("r1", tags=("foo",))])
        assert registry.search_by_tag("bar") == []

    def test_multiple_recipes_match(self) -> None:
        r1 = _recipe("r1", tags=("rag", "retrieval"))
        r2 = _recipe("r2", tags=("rag", "grounding"))
        registry = RecipeRegistry([r1, r2])
        results = registry.search_by_tag("rag")
        assert len(results) == 2

    def test_builtin_research_tag(self) -> None:
        registry = RecipeRegistry.with_builtin_recipes()
        results = registry.search_by_tag("research")
        assert any(r.name == "research" for r in results)


# ===========================================================================
# RecipeRegistry — resolve_all
# ===========================================================================


class TestResolveAll:
    def test_resolve_all_returns_all_recipes(self) -> None:
        r1 = _recipe("r1", steps=(_step("s1", required_category=CapabilityCategory.ANALYSIS.value),))
        r2 = _recipe("r2", steps=(_step("s2", required_category=CapabilityCategory.GENERATION.value),))
        registry = RecipeRegistry([r1, r2])
        analysis_cap = _cap(category=CapabilityCategory.ANALYSIS)
        resolved = registry.resolve_all([analysis_cap])
        assert "r1" in resolved
        assert "r2" in resolved

    def test_resolve_all_empty_registry(self) -> None:
        registry = RecipeRegistry()
        assert registry.resolve_all([]) == {}

    def test_resolve_all_correct_matching(self) -> None:
        step = _step("analyse", required_category=CapabilityCategory.ANALYSIS.value)
        recipe = _recipe("my-recipe", steps=(step,))
        registry = RecipeRegistry([recipe])
        cap = _cap(category=CapabilityCategory.ANALYSIS)
        resolved = registry.resolve_all([cap])
        assert cap in resolved["my-recipe"]["analyse"]


# ===========================================================================
# Built-in recipes structural checks
# ===========================================================================


class TestBuiltinRecipes:
    def _registry(self) -> RecipeRegistry:
        return RecipeRegistry.with_builtin_recipes()

    def test_research_has_three_steps(self) -> None:
        recipe = self._registry().get("research")
        assert recipe.step_count == 3

    def test_rag_has_two_steps(self) -> None:
        recipe = self._registry().get("rag")
        assert recipe.step_count == 2

    def test_code_review_has_three_steps(self) -> None:
        recipe = self._registry().get("code-review")
        assert recipe.step_count == 3

    def test_data_pipeline_has_three_steps(self) -> None:
        recipe = self._registry().get("data-pipeline")
        assert recipe.step_count == 3

    def test_document_qa_has_three_steps(self) -> None:
        recipe = self._registry().get("document-qa")
        assert recipe.step_count == 3

    def test_content_generation_has_two_steps(self) -> None:
        recipe = self._registry().get("content-generation")
        assert recipe.step_count == 2

    def test_summarise_and_act_has_three_steps(self) -> None:
        recipe = self._registry().get("summarise-and-act")
        assert recipe.step_count == 3

    def test_multi_hop_has_four_steps(self) -> None:
        recipe = self._registry().get("multi-hop-reasoning")
        assert recipe.step_count == 4

    def test_eval_loop_has_three_steps(self) -> None:
        recipe = self._registry().get("eval-loop")
        assert recipe.step_count == 3

    def test_classify_and_route_has_two_steps(self) -> None:
        recipe = self._registry().get("classify-and-route")
        assert recipe.step_count == 2

    def test_all_recipes_have_descriptions(self) -> None:
        registry = self._registry()
        for recipe in registry.list_recipes():
            assert recipe.description.strip(), f"Recipe {recipe.name!r} has no description"

    def test_all_steps_have_descriptions(self) -> None:
        registry = self._registry()
        for recipe in registry.list_recipes():
            for step in recipe.steps:
                assert step.description.strip(), (
                    f"Step {step.step_name!r} in recipe {recipe.name!r} has no description"
                )

    def test_all_recipes_have_tags(self) -> None:
        registry = self._registry()
        for recipe in registry.list_recipes():
            assert len(recipe.tags) >= 1, f"Recipe {recipe.name!r} has no tags"

    def test_research_resolve_with_analysis_cap(self) -> None:
        registry = self._registry()
        recipe = registry.get("research")
        analysis_cap = _cap(category=CapabilityCategory.ANALYSIS)
        resolved = recipe.resolve([analysis_cap])
        # The "analyse" step requires ANALYSIS category
        assert analysis_cap in resolved["analyse"]
        # The "search" step requires RESEARCH category — should not match ANALYSIS cap
        assert analysis_cap not in resolved["search"]

    def test_rag_resolve_retrieval_step(self) -> None:
        registry = self._registry()
        recipe = registry.get("rag")
        research_cap = _cap(category=CapabilityCategory.RESEARCH, tags=["retrieval"])
        gen_cap = _cap(category=CapabilityCategory.GENERATION, tags=["generation"])
        resolved = recipe.resolve([research_cap, gen_cap])
        assert research_cap in resolved["retrieve"]
        assert gen_cap in resolved["generate"]

    def test_all_recipes_serialisable(self) -> None:
        import json
        registry = self._registry()
        for recipe in registry.list_recipes():
            d = recipe.to_dict()
            json.dumps(d)  # Should not raise


# ===========================================================================
# Thread safety
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_register_and_list(self) -> None:
        registry = RecipeRegistry()
        errors: list[Exception] = []

        def register_many(start: int) -> None:
            try:
                for i in range(start, start + 10):
                    registry.register(_recipe(f"recipe-{i}"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=register_many, args=(i * 10,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert registry.recipe_count == 50

    def test_concurrent_get_and_remove(self) -> None:
        registry = RecipeRegistry([_recipe(f"r{i}") for i in range(20)])
        errors: list[Exception] = []

        def read_all() -> None:
            try:
                for _ in range(50):
                    registry.list_names()
            except Exception as exc:
                errors.append(exc)

        def remove_some() -> None:
            try:
                for i in range(0, 10):
                    registry.remove(f"r{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=read_all),
            threading.Thread(target=remove_some),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
