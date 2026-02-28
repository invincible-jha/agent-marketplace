/**
 * HTTP client for the agent-marketplace capability registry API.
 *
 * Uses the Fetch API (available natively in Node 18+, browsers, and Deno).
 * No external dependencies required.
 *
 * @example
 * ```ts
 * import { createAgentMarketplaceClient } from "@aumos/agent-marketplace";
 *
 * const client = createAgentMarketplaceClient({ baseUrl: "http://localhost:8092" });
 *
 * const result = await client.discoverAgents({
 *   required_capabilities: ["analysis"],
 *   min_trust: 0.8,
 * });
 *
 * if (result.ok) {
 *   console.log("Found listings:", result.data.length);
 * }
 * ```
 */

import type {
  AgentListing,
  ApiError,
  ApiResult,
  CapabilitySchema,
  CapabilityValidation,
  DiscoveryQuery,
  MatchResult,
} from "./types.js";

// ---------------------------------------------------------------------------
// Client configuration
// ---------------------------------------------------------------------------

/** Configuration options for the AgentMarketplaceClient. */
export interface AgentMarketplaceClientConfig {
  /** Base URL of the agent-marketplace server (e.g. "http://localhost:8092"). */
  readonly baseUrl: string;
  /** Optional request timeout in milliseconds (default: 30000). */
  readonly timeoutMs?: number;
  /** Optional extra HTTP headers sent with every request. */
  readonly headers?: Readonly<Record<string, string>>;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async function fetchJson<T>(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<ApiResult<T>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    clearTimeout(timeoutId);

    const body = await response.json() as unknown;

    if (!response.ok) {
      const errorBody = body as Partial<ApiError>;
      return {
        ok: false,
        error: {
          error: errorBody.error ?? "Unknown error",
          detail: errorBody.detail ?? "",
        },
        status: response.status,
      };
    }

    return { ok: true, data: body as T };
  } catch (err: unknown) {
    clearTimeout(timeoutId);
    const message = err instanceof Error ? err.message : String(err);
    return {
      ok: false,
      error: { error: "Network error", detail: message },
      status: 0,
    };
  }
}

function buildHeaders(
  extraHeaders: Readonly<Record<string, string>> | undefined,
): Record<string, string> {
  return {
    "Content-Type": "application/json",
    Accept: "application/json",
    ...extraHeaders,
  };
}

// ---------------------------------------------------------------------------
// Client interface
// ---------------------------------------------------------------------------

/** Typed HTTP client for the agent-marketplace server. */
export interface AgentMarketplaceClient {
  /**
   * Register a new capability schema in the marketplace.
   *
   * @param capability - The full capability schema to register.
   * @returns The registered CapabilitySchema with auto-generated capability_id.
   */
  registerCapability(
    capability: Omit<CapabilitySchema, "capability_id">,
  ): Promise<ApiResult<CapabilitySchema>>;

  /**
   * Discover agents in the marketplace matching the given query parameters.
   *
   * @param query - Optional filter and ranking parameters.
   * @returns Array of AgentListing records ranked by relevance.
   */
  discoverAgents(query?: DiscoveryQuery): Promise<ApiResult<readonly AgentListing[]>>;

  /**
   * Match a structured capability request against all registered capabilities.
   *
   * Returns results ranked by composite match score (capability overlap,
   * latency fitness, trust level, cost fitness).
   *
   * @param query - The discovery query specifying required capabilities and constraints.
   * @returns Array of MatchResult records sorted best-first by match_score.
   */
  matchCapabilities(query: DiscoveryQuery): Promise<ApiResult<readonly MatchResult[]>>;

  /**
   * Validate a capability schema against marketplace business rules.
   *
   * @param capability - The capability schema to validate.
   * @returns A CapabilityValidation with any error messages.
   */
  validateCapability(
    capability: Omit<CapabilitySchema, "capability_id">,
  ): Promise<ApiResult<CapabilityValidation>>;

  /**
   * Retrieve all marketplace listings, optionally filtered by namespace.
   *
   * @param options - Optional filter and pagination parameters.
   * @returns Array of AgentListing records.
   */
  getListings(options?: {
    readonly namespace?: string;
    readonly category?: string;
    readonly limit?: number;
  }): Promise<ApiResult<readonly AgentListing[]>>;

  /**
   * Retrieve a single capability by its unique identifier.
   *
   * @param capabilityId - The capability_id to look up.
   * @returns The CapabilitySchema for the requested capability.
   */
  getCapability(capabilityId: string): Promise<ApiResult<CapabilitySchema>>;

  /**
   * Deregister a capability from the marketplace.
   *
   * @param capabilityId - The capability_id to remove.
   * @returns An empty object on successful deregistration.
   */
  deregisterCapability(
    capabilityId: string,
  ): Promise<ApiResult<Readonly<Record<string, never>>>>;
}

// ---------------------------------------------------------------------------
// Client factory
// ---------------------------------------------------------------------------

/**
 * Create a typed HTTP client for the agent-marketplace server.
 *
 * @param config - Client configuration including base URL.
 * @returns An AgentMarketplaceClient instance.
 */
export function createAgentMarketplaceClient(
  config: AgentMarketplaceClientConfig,
): AgentMarketplaceClient {
  const { baseUrl, timeoutMs = 30_000, headers: extraHeaders } = config;
  const baseHeaders = buildHeaders(extraHeaders);

  return {
    async registerCapability(
      capability: Omit<CapabilitySchema, "capability_id">,
    ): Promise<ApiResult<CapabilitySchema>> {
      return fetchJson<CapabilitySchema>(
        `${baseUrl}/capabilities`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(capability),
        },
        timeoutMs,
      );
    },

    async discoverAgents(
      query?: DiscoveryQuery,
    ): Promise<ApiResult<readonly AgentListing[]>> {
      return fetchJson<readonly AgentListing[]>(
        `${baseUrl}/discover`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(query ?? {}),
        },
        timeoutMs,
      );
    },

    async matchCapabilities(
      query: DiscoveryQuery,
    ): Promise<ApiResult<readonly MatchResult[]>> {
      return fetchJson<readonly MatchResult[]>(
        `${baseUrl}/match`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(query),
        },
        timeoutMs,
      );
    },

    async validateCapability(
      capability: Omit<CapabilitySchema, "capability_id">,
    ): Promise<ApiResult<CapabilityValidation>> {
      return fetchJson<CapabilityValidation>(
        `${baseUrl}/capabilities/validate`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(capability),
        },
        timeoutMs,
      );
    },

    async getListings(options?: {
      readonly namespace?: string;
      readonly category?: string;
      readonly limit?: number;
    }): Promise<ApiResult<readonly AgentListing[]>> {
      const params = new URLSearchParams();
      if (options?.namespace !== undefined) {
        params.set("namespace", options.namespace);
      }
      if (options?.category !== undefined) {
        params.set("category", options.category);
      }
      if (options?.limit !== undefined) {
        params.set("limit", String(options.limit));
      }
      const queryString = params.toString();
      const url = queryString
        ? `${baseUrl}/listings?${queryString}`
        : `${baseUrl}/listings`;
      return fetchJson<readonly AgentListing[]>(
        url,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async getCapability(
      capabilityId: string,
    ): Promise<ApiResult<CapabilitySchema>> {
      return fetchJson<CapabilitySchema>(
        `${baseUrl}/capabilities/${encodeURIComponent(capabilityId)}`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async deregisterCapability(
      capabilityId: string,
    ): Promise<ApiResult<Readonly<Record<string, never>>>> {
      return fetchJson<Readonly<Record<string, never>>>(
        `${baseUrl}/capabilities/${encodeURIComponent(capabilityId)}`,
        { method: "DELETE", headers: baseHeaders },
        timeoutMs,
      );
    },
  };
}

