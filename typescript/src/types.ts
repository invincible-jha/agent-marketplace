/**
 * TypeScript interfaces for the agent-marketplace capability registry.
 *
 * Mirrors the Pydantic models defined in:
 *   agent_marketplace.schema.capability
 *   agent_marketplace.schema.provider
 *   agent_marketplace.matching.engine
 *   agent_marketplace.matching.request
 *
 * All interfaces use readonly fields to match Python's frozen Pydantic models.
 */

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/**
 * High-level functional category for an agent capability.
 * Maps to CapabilityCategory enum in Python.
 */
export type CapabilityCategory =
  | "analysis"
  | "generation"
  | "transformation"
  | "extraction"
  | "interaction"
  | "automation"
  | "evaluation"
  | "research"
  | "reasoning"
  | "specialized";

/**
 * How a capability charges for its use.
 * Maps to PricingModel enum in Python.
 */
export type PricingModel =
  | "per_call"
  | "per_token"
  | "per_minute"
  | "free"
  | "custom";

// ---------------------------------------------------------------------------
// Provider info
// ---------------------------------------------------------------------------

/**
 * Identity and contact information for a capability provider.
 * Maps to ProviderInfo in Python.
 */
export interface ProviderInfo {
  /** Human-readable name of the provider (person or organization). */
  readonly name: string;
  /** Optional formal organization name. */
  readonly organization: string;
  /** Email address for support or inquiries. */
  readonly contact_email: string;
  /** Optional provider website URL. */
  readonly website: string;
  /** Optional GitHub username or org handle. */
  readonly github_handle: string;
  /** Optional Hugging Face username or org handle. */
  readonly huggingface_handle: string;
  /** Whether the provider identity has been externally verified. */
  readonly verified: boolean;
}

// ---------------------------------------------------------------------------
// Quality metrics and latency profile
// ---------------------------------------------------------------------------

/**
 * Quantitative quality measurements for a capability.
 * Maps to QualityMetrics in Python.
 */
export interface QualityMetrics {
  /** Mapping of metric name to numeric score (e.g. {"accuracy": 0.94}). */
  readonly metrics: Readonly<Record<string, number>>;
  /** Name or URL of the benchmark that produced these numbers. */
  readonly benchmark_source: string;
  /** ISO 8601 date string when benchmarks were last run. */
  readonly benchmark_date: string;
  /** Whether these metrics have been independently verified. */
  readonly verified: boolean;
}

/**
 * Latency percentile measurements in milliseconds.
 * Maps to LatencyProfile in Python.
 */
export interface LatencyProfile {
  /** Median (50th percentile) latency in milliseconds. */
  readonly p50_ms: number;
  /** 95th percentile latency in milliseconds. */
  readonly p95_ms: number;
  /** 99th percentile latency in milliseconds. */
  readonly p99_ms: number;
}

// ---------------------------------------------------------------------------
// Capability schema
// ---------------------------------------------------------------------------

/**
 * A fully described agent capability registered in the marketplace.
 * Maps to AgentCapability in Python.
 */
export interface CapabilitySchema {
  /** Auto-generated stable identifier (SHA-256 hex prefix of name+version+provider). */
  readonly capability_id: string;
  /** Short, human-readable capability name (e.g. "pdf-extractor"). */
  readonly name: string;
  /** Semantic version string (e.g. "1.2.0"). */
  readonly version: string;
  /** Prose description of what the capability does. */
  readonly description: string;
  /** Primary functional category. */
  readonly category: CapabilityCategory;
  /** Free-form tags for keyword search. */
  readonly tags: readonly string[];
  /** MIME types or schema names accepted by the capability. */
  readonly input_types: readonly string[];
  /** MIME type or schema name produced by the capability. */
  readonly output_type: string;
  /** Benchmark and quality scores. */
  readonly quality_metrics: QualityMetrics;
  /** How usage is billed. */
  readonly pricing_model: PricingModel;
  /** Unit cost corresponding to the pricing model (USD). */
  readonly cost: number;
  /** Latency percentile profile. */
  readonly latency: LatencyProfile;
  /** Natural language codes (ISO 639-1) the capability handles. */
  readonly supported_languages: readonly string[];
  /** Agent framework names this capability has adapters for. */
  readonly supported_frameworks: readonly string[];
  /** Current computed trust score (0.0–1.0); updated by TrustScorer. */
  readonly trust_level: number;
  /** Identity information for the capability publisher. */
  readonly provider: ProviderInfo;
}

// ---------------------------------------------------------------------------
// Agent listing
// ---------------------------------------------------------------------------

/**
 * A marketplace listing entry returned when browsing available capabilities.
 * Combines the capability schema with marketplace metadata.
 */
export interface AgentListing {
  /** The registered capability schema. */
  readonly capability: CapabilitySchema;
  /** ISO-8601 UTC timestamp when this capability was registered. */
  readonly registered_at: string;
  /** Total number of times this capability has been invoked. */
  readonly usage_count: number;
  /** Average user rating in [0.0, 5.0], or null if no ratings. */
  readonly average_rating: number | null;
  /** Namespace this capability belongs to (e.g. "aumos", "community"). */
  readonly namespace: string;
}

// ---------------------------------------------------------------------------
// Discovery query
// ---------------------------------------------------------------------------

/**
 * Parameters for discovering agent capabilities in the marketplace.
 * Maps to FilterConstraints + CapabilityRequest in Python.
 */
export interface DiscoveryQuery {
  /** Required capability keywords, names, or category strings. */
  readonly required_capabilities?: readonly string[];
  /** Preferred median latency in milliseconds (soft preference). */
  readonly preferred_latency_ms?: number;
  /** Maximum acceptable cost per call in USD. */
  readonly max_cost?: number;
  /** Minimum acceptable trust level [0.0, 1.0]. */
  readonly min_trust?: number;
  /** Restrict results to a single capability category. */
  readonly category?: CapabilityCategory;
  /** Tags that must all be present (AND semantics). */
  readonly required_tags?: readonly string[];
  /** Maximum number of results to return. */
  readonly limit?: number;
  /** Namespace to restrict the search to. */
  readonly namespace?: string;
}

// ---------------------------------------------------------------------------
// Match result
// ---------------------------------------------------------------------------

/**
 * A capability paired with its composite match score for a specific request.
 * Maps to MatchResult in Python.
 */
export interface MatchResult {
  /** The matched capability schema. */
  readonly capability: CapabilitySchema;
  /** Composite match quality score in [0.0, 1.0]. Higher is better. */
  readonly match_score: number;
  /** Fraction of required capabilities matched [0.0, 1.0]. */
  readonly capability_overlap: number;
  /** Latency fitness component [0.0, 1.0]. */
  readonly latency_score: number;
  /** Trust component (direct from capability.trust_level). */
  readonly trust_score: number;
  /** Cost fitness component [0.0, 1.0]. */
  readonly cost_score: number;
}

// ---------------------------------------------------------------------------
// Capability validation
// ---------------------------------------------------------------------------

/**
 * Result of validating a capability schema against marketplace business rules.
 * Maps to the output of AgentCapability.validate() in Python.
 */
export interface CapabilityValidation {
  /** Whether the capability passes all validation rules. */
  readonly valid: boolean;
  /** List of human-readable error messages; empty when valid. */
  readonly errors: readonly string[];
  /** The capability_id that was validated. */
  readonly capability_id: string;
}

// ---------------------------------------------------------------------------
// API result wrapper
// ---------------------------------------------------------------------------

/** Standard error payload returned by the agent-marketplace API. */
export interface ApiError {
  readonly error: string;
  readonly detail: string;
}

/** Result type for all client operations. */
export type ApiResult<T> =
  | { readonly ok: true; readonly data: T }
  | { readonly ok: false; readonly error: ApiError; readonly status: number };
