import { apiFetch } from "./client"

export interface TransactionSummary {
  request_id: string
  timestamp: string
  provider: string
  model: string
  status: string
  duration_ms: number
  tokens_in: number
  tokens_out: number
  tokens_cached: number
  reasoning_tokens: number
  approx_cost: number | null
  prompt_preview: string
  log_level: string
  format: "oai" | "ant"
  credential_masked?: string | null
}

export interface TransactionDetail {
  request_id: string
  timestamp: string
  provider: string
  model: string
  status: string
  duration_ms: number
  tokens: {
    prompt: number
    completion: number
    total: number
    cached: number
    reasoning: number
  }
  approx_cost: number | null
  files: string[]
  has_provider_logs: boolean
}

export interface TransactionListResponse {
  transactions: TransactionSummary[]
  total: number
  page: number
  page_size: number
}

export interface FailureEntry {
  timestamp: string
  model: string
  provider?: string
  error_type: string
  error_message: string
  raw_response?: string
  request_headers?: Record<string, string>
  error_chain?: string[]
  api_key_ending?: string
  attempt_number?: number
}

export interface FailureListResponse {
  failures: FailureEntry[]
  total: number
  page: number
  page_size: number
  error_types?: { type: string; count: number }[]
  providers?: { name: string; count: number }[]
}

export async function getTransactions(params?: {
  page?: number
  page_size?: number
  provider?: string
  model?: string
  status?: string
  date_from?: string
  date_to?: string
  search?: string
}): Promise<TransactionListResponse> {
  const searchParams = new URLSearchParams()
  if (params?.page) searchParams.set("page", String(params.page))
  if (params?.page_size) searchParams.set("page_size", String(params.page_size))
  if (params?.provider) searchParams.set("provider", params.provider)
  if (params?.model) searchParams.set("model", params.model)
  if (params?.status) searchParams.set("status", params.status)
  if (params?.date_from) searchParams.set("date_from", params.date_from)
  if (params?.date_to) searchParams.set("date_to", params.date_to)
  if (params?.search) searchParams.set("search", params.search)
  const qs = searchParams.toString()
  return apiFetch(`/v1/admin/transactions${qs ? `?${qs}` : ""}`)
}

export async function getTransactionDetail(requestId: string): Promise<TransactionDetail> {
  return apiFetch(`/v1/admin/transactions/${encodeURIComponent(requestId)}`)
}

export async function getTransactionFile(requestId: string, filename: string): Promise<unknown> {
  return apiFetch(`/v1/admin/transactions/${encodeURIComponent(requestId)}/files/${encodeURIComponent(filename)}`)
}

export async function getFailures(params?: {
  page?: number
  page_size?: number
  error_type?: string
  provider?: string
}): Promise<FailureListResponse> {
  const searchParams = new URLSearchParams()
  if (params?.page) searchParams.set("page", String(params.page))
  if (params?.page_size) searchParams.set("page_size", String(params.page_size))
  if (params?.error_type) searchParams.set("error_type", params.error_type)
  if (params?.provider) searchParams.set("provider", params.provider)
  const qs = searchParams.toString()
  return apiFetch(`/v1/admin/failures${qs ? `?${qs}` : ""}`)
}
