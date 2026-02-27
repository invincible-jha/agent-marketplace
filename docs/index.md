# agent-marketplace

Agent Capability Registry & Discovery — capability schema, semantic matching, MCP discovery.

[![CI](https://github.com/invincible-jha/agent-marketplace/actions/workflows/ci.yaml/badge.svg)](https://github.com/invincible-jha/agent-marketplace/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/agent-marketplace.svg)](https://pypi.org/project/agent-marketplace/)
[![Python versions](https://img.shields.io/pypi/pyversions/agent-marketplace.svg)](https://pypi.org/project/agent-marketplace/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/invincible-jha/agent-marketplace/blob/main/LICENSE)

---

## Installation

```bash
pip install agent-marketplace
```

Verify the installation:

```bash
agent-marketplace version
```

---

## Quick Start

```python
import agent_marketplace

# See examples/01_quickstart.py for a complete working example
```

---

## Key Features

- **`AgentCapability` Pydantic schema** captures category, pricing model, quality metrics (accuracy, latency, throughput), SLA tiers, input/output schemas, and per-capability provider info
- **Three registry backends** — in-memory, SQLite, and Redis — all implementing the `CapabilityStore` ABC, with namespace support for multi-tenant deployments
- **Semantic search and keyword filtering** via `DiscoveryClient` with a `Ranker` that scores candidates on quality metrics, trust score, and cost compatibility
- **Trust and reputation system** with peer reviews, usage history, and a composite `TrustScorer` that aggregates reviewer ratings into a single trust signal
- **`MatchingEngine`** negotiates capability requests against registered providers, handling partial matches and SLA compatibility checks
- **Import adapters** for OpenAPI and AsyncAPI specs so existing API documentation can be ingested directly as capability registrations
- **REST API server** (`server.api`) and health endpoint for embedding the marketplace as a sidecar service in larger agent deployments

---

## Links

- [GitHub Repository](https://github.com/invincible-jha/agent-marketplace)
- [PyPI Package](https://pypi.org/project/agent-marketplace/)
- [Architecture](architecture.md)
- [Contributing](https://github.com/invincible-jha/agent-marketplace/blob/main/CONTRIBUTING.md)
- [Changelog](https://github.com/invincible-jha/agent-marketplace/blob/main/CHANGELOG.md)

---

## License

Apache 2.0 — see [LICENSE](https://github.com/invincible-jha/agent-marketplace/blob/main/LICENSE) for full terms.

---

Part of the [AumOS](https://github.com/aumos-ai) open-source agent infrastructure.
