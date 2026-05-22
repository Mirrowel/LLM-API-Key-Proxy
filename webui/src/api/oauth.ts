import { apiFetch } from "./client"

export interface OAuthProviderInfo {
  provider_id: string
  name: string
  flow_type: string
  description: string
}

export interface OAuthProvidersResponse {
  providers: OAuthProviderInfo[]
}

export interface OAuthStartResponse {
  flow_id: string
  flow_type: "device_code" | "authorization_code_paste"
  verification_uri?: string
  user_code?: string
  expires_in?: number
  auth_url?: string
  paste_hint?: string
}

export interface OAuthStatusResponse {
  flow_id: string
  provider: string
  status: "pending" | "complete" | "error"
  result?: { login: string; provider: string }
  error?: string
}

export async function getOAuthProviders(): Promise<OAuthProvidersResponse> {
  return apiFetch("/v1/admin/oauth/providers")
}

export async function startOAuthFlow(provider: string): Promise<OAuthStartResponse> {
  return apiFetch("/v1/admin/oauth/start", {
    method: "POST",
    body: JSON.stringify({ provider }),
  })
}

export async function getOAuthStatus(flowId: string): Promise<OAuthStatusResponse> {
  return apiFetch(`/v1/admin/oauth/status/${flowId}`)
}

export async function submitOAuthCode(flowId: string, code: string): Promise<OAuthStatusResponse> {
  return apiFetch("/v1/admin/oauth/callback", {
    method: "POST",
    body: JSON.stringify({ flow_id: flowId, code }),
  })
}
