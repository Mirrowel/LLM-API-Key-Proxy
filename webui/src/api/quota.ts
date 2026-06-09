import { apiFetch } from "./client"

export interface QuotaStatsResponse {
  providers: Record<string, ProviderStats>
  summary: QuotaSummary
  global_summary?: QuotaSummary
  data_source: string
  timestamp: number
}

export interface QuotaSummary {
  total_providers?: number
  total_credentials: number
  active_credentials?: number
  exhausted_credentials?: number
  total_requests: number
  tokens: TokenStats
  approx_total_cost: number | null
  window_name?: string
}

export interface TokenStats {
  input_cached: number
  input_uncached: number
  input_cache_pct: number
  output: number
}

export interface ProviderStats {
  provider: string
  credential_count: number
  active_count: number
  exhausted_count: number
  rotation_mode: string
  total_requests: number
  tokens: TokenStats
  approx_cost: number
  current_period?: {
    total_requests: number
    tokens: TokenStats
    approx_cost: number
    window_name: string
  }
  quota_groups?: Record<string, QuotaGroup>
  credentials: Record<string, CredentialStats>
}

export interface QuotaGroup {
  tiers?: Record<string, { priority?: number; total: number }>
  windows: Record<string, WindowInfo>
  fair_cycle_summary?: FairCycleSummary
}

export interface WindowInfo {
  total_used: number
  total_remaining: number
  total_max: number
  remaining_pct: number
  tier_availability?: Record<string, { total: number; available: number }>
}

export interface FairCycleSummary {
  exhausted_count: number
  total_count: number
}

export interface CredentialStats {
  stable_id: string
  accessor_masked: string
  full_path?: string
  identifier?: string
  email?: string | null
  tier?: string | null
  priority?: number
  status: "active" | "cooldown" | "exhausted" | "mixed" | "needs_reauth" | "error"
  active_requests: number
  totals: {
    request_count: number
    success_count: number
    failure_count: number
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    approx_cost: number
    first_used_at?: string | null
    last_used_at?: string | null
  }
  model_usage?: Record<string, ModelUsageEntry>
  group_usage?: Record<string, {
    windows: Record<string, {
      request_count: number
      remaining: number
      limit: number
      reset_at?: number | null
      approx_cost: number
    }>
    fair_cycle_exhausted?: boolean
    fair_cycle_reason?: string | null
    cooldown_remaining?: number | null
  }>
  monthly_budget?: {
    budget: number
    spent: number
    remaining: number
    percent_used: number
    period_start: number
    reset_at: number
    reset_day: number
  }
  rpd_limits?: Record<string, {
    limit: number
    used: number
    remaining: number
    reset_at: number
  }>
  cooldowns?: Record<string, unknown>
  fair_cycle?: Record<string, unknown>
  current_period?: {
    request_count: number
    prompt_tokens: number
    output_tokens: number
    approx_cost: number
  }
}

export interface ModelUsageWindow {
  request_count: number
  success_count?: number
  failure_count?: number
  prompt_tokens: number
  completion_tokens: number
  thinking_tokens?: number
  output_tokens?: number
  prompt_tokens_cache_read?: number
  total_tokens: number
  limit?: number
  remaining?: number
  approx_cost: number
  first_used_at?: number | null
  last_used_at?: number | null
}

export interface ModelUsageEntry {
  windows?: Record<string, ModelUsageWindow>
  totals?: ModelUsageWindow
  request_count?: number
  prompt_tokens?: number
  completion_tokens?: number
  total_tokens?: number
  approx_cost?: number
  last_used_at?: string | null
}

export interface CooldownInfo {
  reason: string
  source?: string
  remaining?: number
  expires_at?: string
}

export async function getQuotaStats(provider?: string): Promise<QuotaStatsResponse> {
  const qs = provider ? `?provider=${encodeURIComponent(provider)}` : ""
  return apiFetch(`/v1/quota-stats${qs}`)
}

export async function reloadQuotaStats(
  scope: "all" | "provider" | "credential",
  provider?: string,
  credential?: string
): Promise<QuotaStatsResponse> {
  return apiFetch("/v1/quota-stats", {
    method: "POST",
    body: JSON.stringify({
      action: "reload",
      scope,
      ...(provider ? { provider } : {}),
      ...(credential ? { credential } : {}),
    }),
  })
}

export async function forceRefreshQuota(
  scope: "all" | "provider" | "credential",
  provider?: string,
  credential?: string
): Promise<QuotaStatsResponse> {
  return apiFetch("/v1/quota-stats", {
    method: "POST",
    body: JSON.stringify({
      action: "force_refresh",
      scope,
      ...(provider ? { provider } : {}),
      ...(credential ? { credential } : {}),
    }),
  })
}
