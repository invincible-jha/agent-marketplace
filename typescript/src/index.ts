/**
 * @aumos/agent-marketplace
 *
 * TypeScript client for the AumOS agent-marketplace library.
 * Provides HTTP client and capability type definitions for agent capability
 * registration, discovery, matching, and validation.
 */

// Client and configuration
export type { AgentMarketplaceClient, AgentMarketplaceClientConfig } from "./client.js";
export { createAgentMarketplaceClient } from "./client.js";

// Core types
export type {
  CapabilityCategory,
  PricingModel,
  ProviderInfo,
  QualityMetrics,
  LatencyProfile,
  CapabilitySchema,
  AgentListing,
  DiscoveryQuery,
  MatchResult,
  CapabilityValidation,
  ApiError,
  ApiResult,
} from "./types.js";
