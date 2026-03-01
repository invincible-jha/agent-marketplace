/**
 * HTTP client for the agent-marketplace capability registry API.
 *
 * Delegates all HTTP transport to `@aumos/sdk-core` which provides
 * automatic retry with exponential back-off, timeout management via
 * `AbortSignal.timeout`, interceptor support, and a typed error hierarchy.
 *
 * The public-facing `ApiResult<T>` envelope is preserved for full
 * backward compatibility with existing callers.
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

import {
  createHttpClient,
  HttpError,
  NetworkError,
  TimeoutError,
  AumosError,
  type HttpClient,
} from "@aumos/sdk-core";

import type {
  AgentListing,
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
// Internal adapter
// ---------------------------------------------------------------------------

async function callApi<T>(
  operation: () => Promise<{ readonly data: T; readonly status: number }>,
): Promise<ApiResult<T>> {
  try {
    const response = await operation();
    return { ok: true, data: response.data };
  } catch (error: unknown) {
    if (error instanceof HttpError) {
      return {
        ok: false,
        error: { error: error.message, detail: String(error.body ?? "") },
        status: error.statusCode,
      };
    }
    if (error instanceof TimeoutError) {
      return {
        ok: false,
        error: { error: "Request timed out", detail: error.message },
        status: 0,
      };
    }
    if (error instanceof NetworkError) {
      return {
        ok: false,
        error: { error: "Network error", detail: error.message },
        status: 0,
      };
    }
    if (error instanceof AumosError) {
      return {
        ok: false,
        error: { error: error.code, detail: error.message },
        status: error.statusCode ?? 0,
      };
    }
    const message = error instanceof Error ? error.message : String(error);
    return {
      ok: false,
      error: { error: "Unexpected error", detail: message },
      status: 0,
    };
  }
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
  const http: HttpClient = createHttpClient({
    baseUrl: config.baseUrl,
    timeout: config.timeoutMs ?? 30_000,
    defaultHeaders: config.headers,
  });

  return {
    registerCapability(
      capability: Omit<CapabilitySchema, "capability_id">,
    ): Promise<ApiResult<CapabilitySchema>> {
      return callApi(() => http.post<CapabilitySchema>("/capabilities", capability));
    },

    discoverAgents(query?: DiscoveryQuery): Promise<ApiResult<readonly AgentListing[]>> {
      return callApi(() =>
        http.post<readonly AgentListing[]>("/discover", query ?? {}),
      );
    },

    matchCapabilities(query: DiscoveryQuery): Promise<ApiResult<readonly MatchResult[]>> {
      return callApi(() => http.post<readonly MatchResult[]>("/match", query));
    },

    validateCapability(
      capability: Omit<CapabilitySchema, "capability_id">,
    ): Promise<ApiResult<CapabilityValidation>> {
      return callApi(() =>
        http.post<CapabilityValidation>("/capabilities/validate", capability),
      );
    },

    getListings(options?: {
      readonly namespace?: string;
      readonly category?: string;
      readonly limit?: number;
    }): Promise<ApiResult<readonly AgentListing[]>> {
      const queryParams: Record<string, string> = {};
      if (options?.namespace !== undefined) queryParams["namespace"] = options.namespace;
      if (options?.category !== undefined) queryParams["category"] = options.category;
      if (options?.limit !== undefined) queryParams["limit"] = String(options.limit);
      return callApi(() =>
        http.get<readonly AgentListing[]>("/listings", { queryParams }),
      );
    },

    getCapability(capabilityId: string): Promise<ApiResult<CapabilitySchema>> {
      return callApi(() =>
        http.get<CapabilitySchema>(`/capabilities/${encodeURIComponent(capabilityId)}`),
      );
    },

    deregisterCapability(
      capabilityId: string,
    ): Promise<ApiResult<Readonly<Record<string, never>>>> {
      return callApi(() =>
        http.delete<Readonly<Record<string, never>>>(
          `/capabilities/${encodeURIComponent(capabilityId)}`,
        ),
      );
    },
  };
}
