const AUTH_STORAGE_KEY = "proxy_api_key"
const BASE_URL_STORAGE_KEY = "proxy_base_url"

export function getApiKey(): string | null {
  return localStorage.getItem(AUTH_STORAGE_KEY)
}

export function setApiKey(key: string) {
  localStorage.setItem(AUTH_STORAGE_KEY, key)
}

export function clearApiKey() {
  localStorage.removeItem(AUTH_STORAGE_KEY)
}

export function getBaseUrl(): string {
  return localStorage.getItem(BASE_URL_STORAGE_KEY) || ""
}

export function setBaseUrl(url: string) {
  if (url) {
    localStorage.setItem(BASE_URL_STORAGE_KEY, url)
  } else {
    localStorage.removeItem(BASE_URL_STORAGE_KEY)
  }
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const baseUrl = getBaseUrl()
  const url = `${baseUrl}${path}`
  const apiKey = getApiKey()

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string> || {}),
  }

  if (apiKey) {
    headers["Authorization"] = `Bearer ${apiKey}`
  }

  const response = await fetch(url, {
    ...options,
    headers,
  })

  if (response.status === 401) {
    throw new ApiError(401, "Authentication required")
  }

  if (!response.ok) {
    const text = await response.text()
    throw new ApiError(response.status, text || response.statusText)
  }

  const text = await response.text()
  return text ? JSON.parse(text) : undefined as T
}
