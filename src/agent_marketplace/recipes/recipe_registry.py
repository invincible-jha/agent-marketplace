"""Agent composition recipe registry.

Design
------
:class:`AgentRecipe` describes a named, versioned multi-step pipeline of agent
capabilities.  Each :class:`RecipeStep` in the pipeline references a capability
by category and/or required tags, making recipes capability-agnostic — they
describe *what* to compose, not *which* specific registered capability to use.

:class:`RecipeRegistry` stores recipes in memory, supports lookup by name, and
can resolve which registered :class:`AgentCapability` objects satisfy each step
using a simple match function.

Built-in recipes cover the most common agentic composition patterns:
``research``, ``rag``, ``code-review``, ``data-pipeline``, ``document-qa``,
``content-generation``, ``summarise-and-act``, ``multi-hop-reasoning``,
``eval-loop``, ``classify-and-route``.

Usage
-----
::

    from agent_marketplace.recipes import RecipeRegistry

    registry = RecipeRegistry.with_builtin_recipes()
    recipe = registry.get("research")
    for step in recipe.steps:
        print(step.step_name, step.required_category)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent_marketplace.schema.capability import AgentCapability, CapabilityCategory


# ---------------------------------------------------------------------------
# RecipeStep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecipeStep:
    """A single step in a composition recipe.

    Parameters
    ----------
    step_name:
        Short human-readable name for this step (e.g. ``"retrieval"``).
    description:
        What this step accomplishes in the pipeline.
    required_category:
        The :class:`CapabilityCategory` value that must be satisfied.
        If None, matching falls back to ``required_tags`` only.
    required_tags:
        All tags that a candidate capability must possess.
    optional_tags:
        Tags that are preferred but not mandatory.
    input_from:
        Name(s) of preceding steps whose output feeds into this step.
    output_key:
        Key name under which this step's output is stored for downstream steps.
    metadata:
        Arbitrary extra annotations (e.g. retry_limit, timeout_ms).
    """

    step_name: str
    description: str = ""
    required_category: str | None = None
    required_tags: tuple[str, ...] = ()
    optional_tags: tuple[str, ...] = ()
    input_from: tuple[str, ...] = ()
    output_key: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def matches(self, capability: AgentCapability) -> bool:
        """Return True when *capability* satisfies this step's requirements.

        A capability satisfies the step when:

        1. If ``required_category`` is set: the capability's category value
           matches (case-insensitive).
        2. All ``required_tags`` are present in the capability's tag set.

        Parameters
        ----------
        capability:
            The capability to test.

        Returns
        -------
        bool
        """
        if self.required_category is not None:
            if capability.category.value.lower() != self.required_category.lower():
                return False

        cap_tags = {t.lower() for t in capability.tags}
        for required_tag in self.required_tags:
            if required_tag.lower() not in cap_tags:
                return False

        return True

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary."""
        return {
            "step_name": self.step_name,
            "description": self.description,
            "required_category": self.required_category,
            "required_tags": list(self.required_tags),
            "optional_tags": list(self.optional_tags),
            "input_from": list(self.input_from),
            "output_key": self.output_key,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# AgentRecipe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentRecipe:
    """A named, versioned composition pattern for agent capabilities.

    Parameters
    ----------
    name:
        Unique identifier for this recipe (e.g. ``"research"``).
    version:
        Semantic version string (e.g. ``"1.0.0"``).
    description:
        Human-readable summary of what this recipe accomplishes.
    steps:
        Ordered list of :class:`RecipeStep` objects defining the pipeline.
    tags:
        Categorisation labels for the recipe itself.
    author:
        Creator or maintainer of the recipe.
    metadata:
        Arbitrary extra annotations.
    """

    name: str
    version: str
    description: str
    steps: tuple[RecipeStep, ...]
    tags: tuple[str, ...] = ()
    author: str = "aumos-ai"
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        """Number of steps in this recipe."""
        return len(self.steps)

    @property
    def step_names(self) -> list[str]:
        """Ordered list of step names."""
        return [s.step_name for s in self.steps]

    def get_step(self, step_name: str) -> RecipeStep | None:
        """Return the step with the given name, or None if not found.

        Parameters
        ----------
        step_name:
            The name to look up.

        Returns
        -------
        RecipeStep | None
        """
        for step in self.steps:
            if step.step_name == step_name:
                return step
        return None

    def resolve(
        self, capabilities: list[AgentCapability]
    ) -> dict[str, list[AgentCapability]]:
        """Return candidates for each step from *capabilities*.

        For each step, filters *capabilities* to those that satisfy
        ``step.matches()``.

        Parameters
        ----------
        capabilities:
            Pool of candidate :class:`AgentCapability` objects.

        Returns
        -------
        dict[str, list[AgentCapability]]
            Mapping of step_name → matching capabilities.
        """
        result: dict[str, list[AgentCapability]] = {}
        for step in self.steps:
            result[step.step_name] = [c for c in capabilities if step.matches(c)]
        return result

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "tags": list(self.tags),
            "author": self.author,
            "step_count": self.step_count,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# RecipeRegistry
# ---------------------------------------------------------------------------


class RecipeRegistry:
    """In-memory store for :class:`AgentRecipe` objects.

    Parameters
    ----------
    recipes:
        Optional initial list of recipes to load.

    Example
    -------
    ::

        registry = RecipeRegistry.with_builtin_recipes()
        recipe = registry.get("research")
        resolved = recipe.resolve(my_capabilities)
    """

    def __init__(self, recipes: list[AgentRecipe] | None = None) -> None:
        self._lock = threading.Lock()
        self._recipes: dict[str, AgentRecipe] = {}
        for recipe in (recipes or []):
            self._register(recipe)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, recipe: AgentRecipe) -> None:
        """Add or replace a recipe in the registry.

        Parameters
        ----------
        recipe:
            The recipe to register.
        """
        with self._lock:
            self._register(recipe)

    def get(self, name: str) -> AgentRecipe:
        """Return the recipe with the given name.

        Parameters
        ----------
        name:
            Recipe name (case-sensitive).

        Returns
        -------
        AgentRecipe

        Raises
        ------
        KeyError
            If no recipe with *name* is registered.
        """
        with self._lock:
            try:
                return self._recipes[name]
            except KeyError:
                raise KeyError(f"No recipe registered with name {name!r}.")

    def get_or_none(self, name: str) -> AgentRecipe | None:
        """Return the recipe or None if not found.

        Parameters
        ----------
        name:
            Recipe name.

        Returns
        -------
        AgentRecipe | None
        """
        with self._lock:
            return self._recipes.get(name)

    def remove(self, name: str) -> bool:
        """Remove a recipe by name.

        Parameters
        ----------
        name:
            Name of the recipe to remove.

        Returns
        -------
        bool
            True if the recipe was present and removed; False otherwise.
        """
        with self._lock:
            if name in self._recipes:
                del self._recipes[name]
                return True
            return False

    def list_names(self) -> list[str]:
        """Return sorted list of registered recipe names.

        Returns
        -------
        list[str]
        """
        with self._lock:
            return sorted(self._recipes.keys())

    def list_recipes(self) -> list[AgentRecipe]:
        """Return all registered recipes sorted by name.

        Returns
        -------
        list[AgentRecipe]
        """
        with self._lock:
            return sorted(self._recipes.values(), key=lambda r: r.name)

    def search_by_tag(self, tag: str) -> list[AgentRecipe]:
        """Return recipes that include *tag* in their tags.

        Parameters
        ----------
        tag:
            Tag to search for (case-insensitive).

        Returns
        -------
        list[AgentRecipe]
        """
        tag_lower = tag.lower()
        with self._lock:
            return [
                r for r in self._recipes.values()
                if any(t.lower() == tag_lower for t in r.tags)
            ]

    def resolve_all(
        self, capabilities: list[AgentCapability]
    ) -> dict[str, dict[str, list[AgentCapability]]]:
        """Resolve all recipes against *capabilities*.

        Parameters
        ----------
        capabilities:
            Pool of candidate capabilities.

        Returns
        -------
        dict[str, dict[str, list[AgentCapability]]]
            Outer key is recipe name; inner mapping is step_name →
            matching capabilities.
        """
        with self._lock:
            recipes = list(self._recipes.values())
        return {r.name: r.resolve(capabilities) for r in recipes}

    @property
    def recipe_count(self) -> int:
        """Number of registered recipes."""
        with self._lock:
            return len(self._recipes)

    # ------------------------------------------------------------------
    # Factory: built-in recipes
    # ------------------------------------------------------------------

    @classmethod
    def with_builtin_recipes(cls) -> "RecipeRegistry":
        """Return a registry pre-loaded with all built-in recipes.

        Built-in recipes:

        - ``research``             — web search + analysis + summarise
        - ``rag``                  — retrieval + generation
        - ``code-review``          — extraction + analysis + evaluation
        - ``data-pipeline``        — extraction + transformation + analysis
        - ``document-qa``          — extraction + reasoning + generation
        - ``content-generation``   — research + generation
        - ``summarise-and-act``    — extraction + reasoning + automation
        - ``multi-hop-reasoning``  — research + reasoning + analysis
        - ``eval-loop``            — generation + evaluation + analysis
        - ``classify-and-route``   — analysis + automation

        Returns
        -------
        RecipeRegistry
        """
        registry = cls()
        for recipe in _BUILTIN_RECIPES:
            registry.register(recipe)
        return registry

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register(self, recipe: AgentRecipe) -> None:
        """Register a recipe without acquiring the lock (caller must hold it)."""
        self._recipes[recipe.name] = recipe


# ---------------------------------------------------------------------------
# Built-in recipe definitions
# ---------------------------------------------------------------------------

_BUILTIN_RECIPES: list[AgentRecipe] = [
    # 1. Research pipeline
    AgentRecipe(
        name="research",
        version="1.0.0",
        description=(
            "Multi-step research pipeline: search the web, analyse findings, "
            "and produce a structured summary."
        ),
        steps=(
            RecipeStep(
                step_name="search",
                description="Retrieve relevant documents or web pages.",
                required_category=CapabilityCategory.RESEARCH.value,
                required_tags=(),
                output_key="search_results",
            ),
            RecipeStep(
                step_name="analyse",
                description="Analyse and extract key insights from search results.",
                required_category=CapabilityCategory.ANALYSIS.value,
                required_tags=(),
                input_from=("search",),
                output_key="insights",
            ),
            RecipeStep(
                step_name="summarise",
                description="Generate a human-readable summary of insights.",
                required_category=CapabilityCategory.GENERATION.value,
                required_tags=(),
                input_from=("analyse",),
                output_key="summary",
            ),
        ),
        tags=("research", "information-retrieval", "summarisation"),
    ),

    # 2. Retrieval-Augmented Generation (RAG)
    AgentRecipe(
        name="rag",
        version="1.0.0",
        description=(
            "Retrieval-augmented generation: retrieve relevant context chunks "
            "and use them to ground a language model response."
        ),
        steps=(
            RecipeStep(
                step_name="retrieve",
                description="Retrieve top-k context chunks from a vector store or corpus.",
                required_category=CapabilityCategory.RESEARCH.value,
                required_tags=(),
                output_key="context_chunks",
            ),
            RecipeStep(
                step_name="generate",
                description="Generate a grounded response conditioned on retrieved context.",
                required_category=CapabilityCategory.GENERATION.value,
                required_tags=(),
                input_from=("retrieve",),
                output_key="response",
            ),
        ),
        tags=("rag", "retrieval", "generation", "grounding"),
    ),

    # 3. Code Review
    AgentRecipe(
        name="code-review",
        version="1.0.0",
        description=(
            "Automated code review: extract code structure, analyse for issues, "
            "and evaluate quality against defined standards."
        ),
        steps=(
            RecipeStep(
                step_name="extract",
                description="Extract code structure, dependencies, and metadata.",
                required_category=CapabilityCategory.EXTRACTION.value,
                required_tags=(),
                output_key="code_structure",
            ),
            RecipeStep(
                step_name="analyse",
                description="Analyse extracted structure for bugs, smells, and anti-patterns.",
                required_category=CapabilityCategory.ANALYSIS.value,
                required_tags=(),
                input_from=("extract",),
                output_key="findings",
            ),
            RecipeStep(
                step_name="evaluate",
                description="Score overall code quality and compliance.",
                required_category=CapabilityCategory.EVALUATION.value,
                required_tags=(),
                input_from=("analyse",),
                output_key="quality_score",
            ),
        ),
        tags=("code", "review", "quality", "static-analysis"),
    ),

    # 4. Data Pipeline
    AgentRecipe(
        name="data-pipeline",
        version="1.0.0",
        description=(
            "Extract raw data, apply transformations, and produce analytical insights."
        ),
        steps=(
            RecipeStep(
                step_name="extract",
                description="Extract raw data from a source (file, API, database).",
                required_category=CapabilityCategory.EXTRACTION.value,
                required_tags=(),
                output_key="raw_data",
            ),
            RecipeStep(
                step_name="transform",
                description="Clean, normalise, and reshape the raw data.",
                required_category=CapabilityCategory.TRANSFORMATION.value,
                required_tags=(),
                input_from=("extract",),
                output_key="transformed_data",
            ),
            RecipeStep(
                step_name="analyse",
                description="Compute aggregates, trends, and statistical insights.",
                required_category=CapabilityCategory.ANALYSIS.value,
                required_tags=(),
                input_from=("transform",),
                output_key="analytics",
            ),
        ),
        tags=("data", "etl", "pipeline", "analytics"),
    ),

    # 5. Document Q&A
    AgentRecipe(
        name="document-qa",
        version="1.0.0",
        description=(
            "Extract content from a document, reason over it, and generate a "
            "precise answer to a user question."
        ),
        steps=(
            RecipeStep(
                step_name="extract",
                description="Extract text and structure from the document.",
                required_category=CapabilityCategory.EXTRACTION.value,
                required_tags=(),
                output_key="document_text",
            ),
            RecipeStep(
                step_name="reason",
                description="Apply multi-step reasoning over the extracted content.",
                required_category=CapabilityCategory.REASONING.value,
                required_tags=(),
                input_from=("extract",),
                output_key="reasoning_trace",
            ),
            RecipeStep(
                step_name="answer",
                description="Generate a concise, grounded answer.",
                required_category=CapabilityCategory.GENERATION.value,
                required_tags=(),
                input_from=("reason",),
                output_key="answer",
            ),
        ),
        tags=("document", "qa", "question-answering", "reasoning"),
    ),

    # 6. Content Generation
    AgentRecipe(
        name="content-generation",
        version="1.0.0",
        description=(
            "Research a topic and produce high-quality, factually grounded content."
        ),
        steps=(
            RecipeStep(
                step_name="research",
                description="Gather facts, references, and background on the topic.",
                required_category=CapabilityCategory.RESEARCH.value,
                required_tags=(),
                output_key="research_material",
            ),
            RecipeStep(
                step_name="generate",
                description="Draft original content grounded in the research material.",
                required_category=CapabilityCategory.GENERATION.value,
                required_tags=(),
                input_from=("research",),
                output_key="draft",
            ),
        ),
        tags=("content", "writing", "generation", "research"),
    ),

    # 7. Summarise and Act
    AgentRecipe(
        name="summarise-and-act",
        version="1.0.0",
        description=(
            "Extract key information from a document, reason about the best action, "
            "then automate the chosen action."
        ),
        steps=(
            RecipeStep(
                step_name="extract",
                description="Extract key entities and actions from the input.",
                required_category=CapabilityCategory.EXTRACTION.value,
                required_tags=(),
                output_key="extracted_entities",
            ),
            RecipeStep(
                step_name="reason",
                description="Determine the optimal action given the extracted information.",
                required_category=CapabilityCategory.REASONING.value,
                required_tags=(),
                input_from=("extract",),
                output_key="action_plan",
            ),
            RecipeStep(
                step_name="automate",
                description="Execute the chosen action via an automation capability.",
                required_category=CapabilityCategory.AUTOMATION.value,
                required_tags=(),
                input_from=("reason",),
                output_key="action_result",
            ),
        ),
        tags=("automation", "action", "reasoning", "extraction"),
    ),

    # 8. Multi-hop Reasoning
    AgentRecipe(
        name="multi-hop-reasoning",
        version="1.0.0",
        description=(
            "Answer complex questions requiring multiple retrieval + reasoning hops."
        ),
        steps=(
            RecipeStep(
                step_name="hop-1-retrieve",
                description="First retrieval: gather seed documents.",
                required_category=CapabilityCategory.RESEARCH.value,
                required_tags=(),
                output_key="hop1_docs",
            ),
            RecipeStep(
                step_name="hop-1-reason",
                description="First reasoning pass to identify what is still unknown.",
                required_category=CapabilityCategory.REASONING.value,
                required_tags=(),
                input_from=("hop-1-retrieve",),
                output_key="follow_up_queries",
            ),
            RecipeStep(
                step_name="hop-2-retrieve",
                description="Second retrieval: gather follow-up documents.",
                required_category=CapabilityCategory.RESEARCH.value,
                required_tags=(),
                input_from=("hop-1-reason",),
                output_key="hop2_docs",
            ),
            RecipeStep(
                step_name="synthesise",
                description="Synthesise findings from both hops into a final answer.",
                required_category=CapabilityCategory.ANALYSIS.value,
                required_tags=(),
                input_from=("hop-1-reason", "hop-2-retrieve"),
                output_key="final_answer",
            ),
        ),
        tags=("multi-hop", "reasoning", "retrieval", "complex-qa"),
    ),

    # 9. Evaluation Loop
    AgentRecipe(
        name="eval-loop",
        version="1.0.0",
        description=(
            "Generate a candidate output, evaluate it, and analyse the evaluation "
            "to inform improvements."
        ),
        steps=(
            RecipeStep(
                step_name="generate",
                description="Produce a candidate output (text, code, plan, etc.).",
                required_category=CapabilityCategory.GENERATION.value,
                required_tags=(),
                output_key="candidate",
            ),
            RecipeStep(
                step_name="evaluate",
                description="Score and critique the candidate output.",
                required_category=CapabilityCategory.EVALUATION.value,
                required_tags=(),
                input_from=("generate",),
                output_key="evaluation",
            ),
            RecipeStep(
                step_name="analyse",
                description="Analyse evaluation results to extract improvement signals.",
                required_category=CapabilityCategory.ANALYSIS.value,
                required_tags=(),
                input_from=("evaluate",),
                output_key="improvement_signals",
            ),
        ),
        tags=("evaluation", "quality", "generation", "feedback"),
    ),

    # 10. Classify and Route
    AgentRecipe(
        name="classify-and-route",
        version="1.0.0",
        description=(
            "Analyse an incoming request to classify its intent, then route it "
            "to the appropriate automated handler."
        ),
        steps=(
            RecipeStep(
                step_name="classify",
                description="Classify the intent and type of the incoming request.",
                required_category=CapabilityCategory.ANALYSIS.value,
                required_tags=(),
                output_key="classification",
            ),
            RecipeStep(
                step_name="route",
                description="Route the classified request to the correct automation.",
                required_category=CapabilityCategory.AUTOMATION.value,
                required_tags=(),
                input_from=("classify",),
                output_key="routing_decision",
            ),
        ),
        tags=("classification", "routing", "automation", "triage"),
    ),
]
