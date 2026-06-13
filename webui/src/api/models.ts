import { apiFetch } from "./client"

export interface ModelCard {
  id: string
  object: string
  created: number
  owned_by: string
  context_length?: number
  max_completion_tokens?: number
  family?: string
  mode?: string
  _sources?: string[]
  _match_type?: string
  _parent_model?: string
  input_cost_per_token?: number
  output_cost_per_token?: number
  pricing?: {
    prompt?: number
    completion?: number
    cached_input?: number
    cache_write?: number
  }
  supported_modalities?: string[]
}

export interface ModelList {
  object: string
  data: ModelCard[]
}

export async function getModels(refresh = false): Promise<ModelList> {
  return apiFetch(`/v1/models${refresh ? "?refresh=true" : ""}`)
}

export async function getProviders(): Promise<string[]> {
  return apiFetch("/v1/providers")
}
