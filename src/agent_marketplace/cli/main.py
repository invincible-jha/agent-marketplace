"""CLI entry point for agent-marketplace.

Invoked as::

    agent-marketplace [OPTIONS] COMMAND [ARGS]...

or, during development::

    python -m agent_marketplace.cli.main

Commands
--------
- ``version``    — show version information.
- ``plugins``    — list registered plugins.
- ``register``   — register a capability from a JSON or YAML file.
- ``search``     — search capabilities by keyword and filters.
- ``info``       — display details for a single capability.
- ``review``     — submit a review for a provider.
- ``analytics``  — display marketplace analytics summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option()
def cli() -> None:
    """Agent capability registry, discovery, and semantic matching."""


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


@cli.command(name="version")
def version_command() -> None:
    """Show detailed version information."""
    from agent_marketplace import __version__

    console.print(f"[bold]agent-marketplace[/bold] v{__version__}")


# ---------------------------------------------------------------------------
# plugins
# ---------------------------------------------------------------------------


@cli.command(name="plugins")
def plugins_command() -> None:
    """List all registered plugins loaded from entry-points."""
    console.print("[bold]Registered plugins:[/bold]")
    console.print("  (No plugins registered. Install a plugin package to see entries here.)")


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


@cli.command(name="register")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    "--db",
    default=":memory:",
    show_default=True,
    help="SQLite database path (use ':memory:' for ephemeral in-memory store).",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "yaml", "auto"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="File format.  'auto' infers from the file extension.",
)
def register_command(file: str, db: str, fmt: str) -> None:
    """Register a capability from a JSON or YAML FILE.

    FILE should be a path to a JSON or YAML file whose content matches
    the AgentCapability schema.

    Examples
    --------
    \\b
      agent-marketplace register capability.yaml
      agent-marketplace register capability.json --db ./registry.db
    """
    from agent_marketplace.schema.capability import AgentCapability
    from agent_marketplace.schema.validator import SchemaValidator

    file_path = Path(file)

    # Determine format
    effective_fmt = fmt
    if effective_fmt == "auto":
        suffix = file_path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            effective_fmt = "yaml"
        else:
            effective_fmt = "json"

    raw_text = file_path.read_text(encoding="utf-8")

    try:
        if effective_fmt == "yaml":
            capability = AgentCapability.from_yaml(raw_text)
        else:
            capability = AgentCapability.from_json(raw_text)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to parse capability file:[/red] {exc}")
        sys.exit(1)

    validator = SchemaValidator()
    result = validator.validate(capability)

    if not result.valid:
        console.print("[red]Validation errors:[/red]")
        for error in result.errors:
            console.print(f"  - {error}")
        sys.exit(1)

    if result.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"  - {warning}")

    # Persist to store
    store = _open_store(db)
    try:
        store.register(capability)
    except ValueError as exc:
        console.print(f"[red]Registration failed:[/red] {exc}")
        sys.exit(1)

    console.print(
        f"[green]Registered[/green] [bold]{capability.name}[/bold] v{capability.version} "
        f"(id: {capability.capability_id})"
    )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@cli.command(name="search")
@click.argument("keyword", default="")
@click.option("--category", "-c", default="", help="Filter by capability category.")
@click.option("--tag", "-t", "tags", multiple=True, help="Filter by tag (repeatable).")
@click.option("--min-trust", default=0.0, show_default=True, type=float, help="Minimum trust level.")
@click.option("--max-cost", default=None, type=float, help="Maximum cost per call (USD).")
@click.option("--limit", "-n", default=20, show_default=True, type=int, help="Maximum results.")
@click.option("--db", default=":memory:", show_default=True, help="SQLite database path.")
@click.option("--json-output", is_flag=True, default=False, help="Output raw JSON.")
def search_command(
    keyword: str,
    category: str,
    tags: tuple[str, ...],
    min_trust: float,
    max_cost: float | None,
    limit: int,
    db: str,
    json_output: bool,
) -> None:
    """Search for capabilities matching KEYWORD and optional filters.

    Examples
    --------
    \\b
      agent-marketplace search "pdf extraction"
      agent-marketplace search --category extraction --min-trust 0.7
      agent-marketplace search --tag nlp --tag ocr --max-cost 0.01
    """
    from agent_marketplace.registry.store import SearchQuery
    from agent_marketplace.schema.capability import CapabilityCategory

    store = _open_store(db)

    cat: CapabilityCategory | None = None
    if category:
        try:
            cat = CapabilityCategory(category.lower())
        except ValueError:
            console.print(f"[red]Unknown category:[/red] {category!r}")
            sys.exit(1)

    query = SearchQuery(
        keyword=keyword,
        category=cat,
        tags=list(tags),
        min_trust=min_trust,
        max_cost=max_cost if max_cost is not None else float("inf"),
        limit=limit,
    )
    results = store.search(query)

    if json_output:
        click.echo(json.dumps([cap.to_dict() for cap in results], indent=2))
        return

    if not results:
        console.print("[yellow]No capabilities found matching your criteria.[/yellow]")
        return

    table = Table(title=f"Search results ({len(results)} found)", show_lines=False)
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Version")
    table.add_column("Category")
    table.add_column("Trust", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Provider")

    for cap in results:
        table.add_row(
            cap.capability_id[:8],
            cap.name,
            cap.version,
            cap.category.value,
            f"{cap.trust_level:.2f}",
            f"{cap.cost:.4f}" if cap.cost > 0 else "free",
            cap.provider.name,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@cli.command(name="info")
@click.argument("capability_id")
@click.option("--db", default=":memory:", show_default=True, help="SQLite database path.")
@click.option("--json-output", is_flag=True, default=False, help="Output raw JSON.")
def info_command(capability_id: str, db: str, json_output: bool) -> None:
    """Display detailed information for a capability by CAPABILITY_ID.

    Examples
    --------
    \\b
      agent-marketplace info abc123def456
      agent-marketplace info abc123def456 --json-output
    """
    store = _open_store(db)

    try:
        capability = store.get(capability_id)
    except KeyError:
        console.print(f"[red]Capability [bold]{capability_id}[/bold] not found.[/red]")
        sys.exit(1)

    if json_output:
        click.echo(capability.to_json())
        return

    console.print(f"\n[bold]{capability.name}[/bold] v{capability.version}")
    console.print(f"  ID:          {capability.capability_id}")
    console.print(f"  Description: {capability.description}")
    console.print(f"  Category:    {capability.category.value}")
    console.print(f"  Tags:        {', '.join(capability.tags) or '(none)'}")
    console.print(f"  Input types: {', '.join(capability.input_types) or '(none)'}")
    console.print(f"  Output type: {capability.output_type or '(none)'}")
    console.print(f"  Pricing:     {capability.pricing_model.value} @ {capability.cost:.4f} USD")
    console.print(
        f"  Latency:     p50={capability.latency.p50_ms}ms  "
        f"p95={capability.latency.p95_ms}ms  p99={capability.latency.p99_ms}ms"
    )
    console.print(f"  Trust level: {capability.trust_level:.4f}")
    console.print(f"\n[bold]Provider[/bold]")
    console.print(f"  Name:         {capability.provider.name}")
    console.print(f"  Organization: {capability.provider.organization or '(none)'}")
    console.print(f"  Email:        {capability.provider.contact_email or '(none)'}")
    console.print(f"  Website:      {capability.provider.website or '(none)'}")
    console.print(f"  Verified:     {'yes' if capability.provider.verified else 'no'}")

    if capability.quality_metrics.metrics:
        console.print(f"\n[bold]Quality Metrics[/bold] (source: {capability.quality_metrics.benchmark_source or 'n/a'})")
        for metric_name, metric_value in capability.quality_metrics.metrics.items():
            console.print(f"  {metric_name}: {metric_value:.4f}")


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------


@cli.command(name="review")
@click.argument("provider_id")
@click.option("--rating", "-r", type=click.IntRange(1, 5), required=True, help="Rating 1–5.")
@click.option("--reviewer-id", "-u", required=True, help="Your reviewer identifier.")
@click.option("--text", "-m", default="", help="Optional review comment.")
def review_command(
    provider_id: str,
    rating: int,
    reviewer_id: str,
    text: str,
) -> None:
    """Submit a review for a provider identified by PROVIDER_ID.

    Examples
    --------
    \\b
      agent-marketplace review acme-corp --rating 5 --reviewer-id user-42
      agent-marketplace review acme-corp -r 3 -u user-42 --text "Decent but slow"
    """
    from agent_marketplace.trust.reviews import Review, ReviewStore

    store = ReviewStore()
    review = Review(
        reviewer_id=reviewer_id,
        provider_id=provider_id,
        rating=rating,
        text=text,
    )
    store.add(review)

    stars = "*" * rating + " " * (5 - rating)
    console.print(
        f"[green]Review submitted[/green] for [bold]{provider_id}[/bold]\n"
        f"  Rating:   {stars} ({rating}/5)\n"
        f"  Reviewer: {reviewer_id}\n"
        f"  Comment:  {text or '(none)'}\n"
        f"  Review ID: {review.review_id}"
    )

    average = store.average_rating(provider_id)
    console.print(
        f"  Average rating for {provider_id}: {average:.1f}/5 "
        f"({store.count_for_provider(provider_id)} review(s))"
    )


# ---------------------------------------------------------------------------
# analytics
# ---------------------------------------------------------------------------


@cli.command(name="analytics")
@click.option("--db", default=":memory:", show_default=True, help="SQLite database path.")
@click.option("--capability", "-c", default=None, help="Capability ID for detailed report.")
@click.option("--json-output", is_flag=True, default=False, help="Output raw JSON.")
def analytics_command(
    db: str,
    capability: str | None,
    json_output: bool,
) -> None:
    """Display marketplace analytics summary or a per-capability report.

    Examples
    --------
    \\b
      agent-marketplace analytics
      agent-marketplace analytics --capability abc123
      agent-marketplace analytics --json-output
    """
    from agent_marketplace.analytics.reporter import MarketplaceReporter
    from agent_marketplace.analytics.usage import UsageTracker

    store = _open_store(db)
    tracker = UsageTracker()
    reporter = MarketplaceReporter(store=store, usage_tracker=tracker)

    if capability:
        report = reporter.capability_report(capability)
    else:
        report = reporter.summary_report()

    if json_output:
        click.echo(json.dumps(report, indent=2, default=str))
        return

    console.print(f"\n[bold]Agent Marketplace Analytics[/bold]")
    console.print(f"  Generated: {report.get('generated_at', 'n/a')}")
    console.print(f"  Version:   {report.get('version', 'n/a')}")

    if "registry" in report:
        registry_data = report["registry"]
        if isinstance(registry_data, dict):
            console.print(f"\n[bold]Registry[/bold]")
            console.print(f"  Total capabilities:  {registry_data.get('total_capabilities', 0)}")
            console.print(f"  Average trust level: {registry_data.get('average_trust_level', 0.0):.4f}")

    if "usage" in report:
        usage_data = report["usage"]
        if isinstance(usage_data, dict):
            console.print(f"\n[bold]Usage[/bold]")
            console.print(f"  Total invocations:   {usage_data.get('total_invocations', 0)}")
            console.print(f"  Global success rate: {usage_data.get('global_success_rate', 0.0):.2%}")
            console.print(f"  Avg latency (ms):    {usage_data.get('global_average_latency_ms', 0.0):.1f}")
            console.print(f"  Total cost (USD):    {usage_data.get('total_cost_usd', 0.0):.6f}")

    if "categories" in report:
        categories_data = report["categories"]
        if isinstance(categories_data, dict) and categories_data:
            console.print(f"\n[bold]Capabilities by Category[/bold]")
            for cat, count in sorted(categories_data.items(), key=lambda kv: kv[1], reverse=True):
                console.print(f"  {cat:<20} {count}")

    if "popular" in report:
        popular_data = report["popular"]
        if isinstance(popular_data, list) and popular_data:
            console.print(f"\n[bold]Most Popular (all-time)[/bold]")
            for entry in popular_data[:5]:
                if isinstance(entry, dict):
                    console.print(
                        f"  {entry.get('capability_id', '?'):<20} "
                        f"{entry.get('total_uses', 0)} uses"
                    )
        else:
            console.print("\n[dim]No usage data recorded yet.[/dim]")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_store(db: str) -> object:
    """Open a registry store appropriate for the given *db* path.

    Uses ``MemoryStore`` when *db* is ``':memory:'``, otherwise uses
    ``SQLiteStore`` (if available) or falls back to ``MemoryStore``.
    """
    from agent_marketplace.registry.memory_store import MemoryStore

    if db == ":memory:":
        return MemoryStore()

    try:
        from agent_marketplace.registry.sqlite_store import SQLiteStore  # type: ignore[attr-defined]

        return SQLiteStore(db)
    except (ImportError, AttributeError):
        console.print(
            f"[yellow]SQLiteStore unavailable; using in-memory store "
            f"(data will not be persisted to {db!r}).[/yellow]"
        )
        return MemoryStore()


if __name__ == "__main__":
    cli()
