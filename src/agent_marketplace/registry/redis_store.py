"""Redis-backed registry store for agent-marketplace.

This module is import-guarded: the ``redis`` package is not a declared
dependency, so an ``ImportError`` with a helpful message is raised if the
extra is not installed.

Install the extra with::

    pip install redis

"""
from __future__ import annotations

try:
    import redis as redis_module
    from redis import Redis
except ImportError as _import_error:  # noqa: F841
    raise ImportError(
        "The Redis registry backend requires the 'redis' package. "
        "Install it with: pip install redis"
    ) from _import_error

from agent_marketplace.registry.store import RegistryStore, SearchQuery
from agent_marketplace.schema.capability import AgentCapability


_KEY_PREFIX = "agent_marketplace:capability:"
_INDEX_KEY = "agent_marketplace:index"


class RedisStore(RegistryStore):
    """Capability registry backed by Redis.

    All capability data is stored as JSON under the key
    ``agent_marketplace:capability:<capability_id>``.  A Redis Set at
    ``agent_marketplace:index`` tracks all registered IDs.

    Parameters
    ----------
    host:
        Redis server host name.
    port:
        Redis server port.
    db:
        Redis database index.
    password:
        Optional authentication password.
    decode_responses:
        Must remain True — the store assumes string responses.
    kwargs:
        Additional keyword arguments forwarded to ``Redis()``.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str = "",
        **kwargs: object,
    ) -> None:
        connect_kwargs: dict[str, object] = {
            "host": host,
            "port": port,
            "db": db,
            "decode_responses": True,
            **kwargs,
        }
        if password:
            connect_kwargs["password"] = password
        self._client: Redis[str] = Redis(**connect_kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, capability: AgentCapability) -> None:
        key = self._key(capability.capability_id)
        if self._client.exists(key):
            raise ValueError(
                f"Capability {capability.capability_id!r} is already registered."
            )
        self._save(capability)

    def update(self, capability: AgentCapability) -> None:
        key = self._key(capability.capability_id)
        if not self._client.exists(key):
            raise KeyError(
                f"Capability {capability.capability_id!r} not found. "
                "Use register() to add a new capability."
            )
        self._save(capability)

    def get(self, capability_id: str) -> AgentCapability:
        key = self._key(capability_id)
        data = self._client.get(key)
        if data is None:
            raise KeyError(f"Capability {capability_id!r} not found in registry.")
        return AgentCapability.model_validate_json(data)

    def delete(self, capability_id: str) -> None:
        key = self._key(capability_id)
        if not self._client.exists(key):
            raise KeyError(f"Capability {capability_id!r} not found in registry.")
        self._client.delete(key)
        self._client.srem(_INDEX_KEY, capability_id)

    # ------------------------------------------------------------------
    # Listing / search
    # ------------------------------------------------------------------

    def list_all(self) -> list[AgentCapability]:
        ids: set[str] = self._client.smembers(_INDEX_KEY)  # type: ignore[assignment]
        results: list[AgentCapability] = []
        for capability_id in ids:
            try:
                results.append(self.get(capability_id))
            except KeyError:
                # Index out of sync; ignore stale entry
                pass
        return results

    def search(self, query: SearchQuery) -> list[AgentCapability]:
        """Full scan — Redis store performs all filtering in Python.

        For high-volume deployments consider using RediSearch.
        """
        all_capabilities = self.list_all()
        matched: list[AgentCapability] = []

        for capability in all_capabilities:
            if not self._matches(capability, query):
                continue
            matched.append(capability)

        start = query.offset
        end = start + query.limit if query.limit > 0 else None
        return matched[start:end]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save(self, capability: AgentCapability) -> None:
        key = self._key(capability.capability_id)
        self._client.set(key, capability.model_dump_json())
        self._client.sadd(_INDEX_KEY, capability.capability_id)

    @staticmethod
    def _key(capability_id: str) -> str:
        return f"{_KEY_PREFIX}{capability_id}"

    @staticmethod
    def _matches(capability: AgentCapability, query: SearchQuery) -> bool:
        if query.keyword:
            keyword_lower = query.keyword.lower()
            searchable = (
                capability.name.lower()
                + " "
                + capability.description.lower()
                + " "
                + " ".join(t.lower() for t in capability.tags)
            )
            if keyword_lower not in searchable:
                return False

        if query.category is not None and capability.category != query.category:
            return False

        if query.tags:
            capability_tags = {t.lower() for t in capability.tags}
            for required_tag in query.tags:
                if required_tag.lower() not in capability_tags:
                    return False

        if capability.trust_level < query.min_trust:
            return False

        if capability.cost > query.max_cost:
            return False

        if query.pricing_model and capability.pricing_model.value != query.pricing_model:
            return False

        if query.supported_language:
            lang_lower = query.supported_language.lower()
            if lang_lower not in [lang.lower() for lang in capability.supported_languages]:
                return False

        if query.supported_framework:
            fw_lower = query.supported_framework.lower()
            if fw_lower not in [fw.lower() for fw in capability.supported_frameworks]:
                return False

        return True
