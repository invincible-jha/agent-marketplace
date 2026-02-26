# agent-marketplace

Agent capability registry, discovery, and semantic matching

[![CI](https://github.com/aumos-ai/agent-marketplace/actions/workflows/ci.yaml/badge.svg)](https://github.com/aumos-ai/agent-marketplace/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/agent-marketplace.svg)](https://pypi.org/project/agent-marketplace/)
[![Python versions](https://img.shields.io/pypi/pyversions/agent-marketplace.svg)](https://pypi.org/project/agent-marketplace/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Part of the [AumOS](https://github.com/aumos-ai) open-source agent infrastructure portfolio.

---

## Features

- `AgentCapability` Pydantic schema captures category, pricing model, quality metrics (accuracy, latency, throughput), SLA tiers, input/output schemas, and per-capability provider info
- Three registry backends — in-memory, SQLite, and Redis — all implementing the `CapabilityStore` ABC, with namespace support for multi-tenant deployments
- Semantic search and keyword filtering via `DiscoveryClient` with a `Ranker` that scores candidates on quality metrics, trust score, and cost compatibility
- Trust and reputation system with peer reviews, usage history, and a composite `TrustScorer` that aggregates reviewer ratings into a single trust signal
- `MatchingEngine` negotiates capability requests against registered providers, handling partial matches and SLA compatibility checks
- Import adapters for OpenAPI and AsyncAPI specs so existing API documentation can be ingested directly as capability registrations
- REST API server (`server.api`) and health endpoint for embedding the marketplace as a sidecar service in larger agent deployments

## Quick Start

Install from PyPI:

```bash
pip install agent-marketplace
```

Verify the installation:

```bash
agent-marketplace version
```

Basic usage:

```python
import agent_marketplace

# See examples/01_quickstart.py for a working example
```

## Documentation

- [Architecture](docs/architecture.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Examples](examples/README.md)

## Enterprise Upgrade

The open-source edition provides the core foundation. For production
deployments requiring SLA-backed support, advanced integrations, and the full
AgentArena platform, see [docs/UPGRADE_TO_AgentArena.md](docs/UPGRADE_TO_AgentArena.md).

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening a pull request.

## License

Apache 2.0 — see [LICENSE](LICENSE) for full terms.

---

Part of [AumOS](https://github.com/aumos-ai) — open-source agent infrastructure.
