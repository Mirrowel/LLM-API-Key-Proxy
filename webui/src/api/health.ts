import { apiFetch } from "./client"

export interface HealthResponse {
  status: string
  uptime_seconds: number
  timestamp: string
  providers: {
    total: number
    active: string[]
    with_errors: string[]
  }
  credentials: {
    total: number
    active: number
    on_cooldown: number
    exhausted: number
    error: number
  }
  models_current_window?: ModelWindowStat[]
  errors?: {
    total_errors: number
    by_provider: Record<string, { count: number; error_types: Record<string, number> }>
    by_model: Record<string, { count: number; error_types: Record<string, number> }>
  }
}

export interface ModelWindowStat {
  model: string
  provider: string
  window_name: string
  requests: number
  success_count: number
  failure_count: number
  tokens: { prompt: number; completion: number; total: number }
  approx_cost: number
  last_used: string
}

export interface ErrorRecord {
  timestamp: string
  provider: string
  model: string
  error_type: string
  status_code: number | null
  error_message: string
  credential?: string
  attempt?: number
}

export async function getHealth(detail: "summary" | "full" = "summary"): Promise<HealthResponse> {
  return apiFetch(`/v1/health?detail=${detail}`)
}

export async function getHealthErrors(params?: {
  provider?: string
  model?: string
  limit?: number
}): Promise<{ errors: ErrorRecord[]; total_matching: number }> {
  const searchParams = new URLSearchParams()
  if (params?.provider) searchParams.set("provider", params.provider)
  if (params?.model) searchParams.set("model", params.model)
  if (params?.limit) searchParams.set("limit", String(params.limit))
  const qs = searchParams.toString()
  return apiFetch(`/v1/health/errors${qs ? `?${qs}` : ""}`)
}
