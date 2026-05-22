import { useState, useCallback } from "react"
import { getApiKey, setApiKey, clearApiKey } from "@/api/client"

export function useAuth() {
  const [isAuthenticated, setIsAuthenticated] = useState(!!getApiKey())

  const login = useCallback((key: string) => {
    setApiKey(key)
    setIsAuthenticated(true)
  }, [])

  const logout = useCallback(() => {
    clearApiKey()
    setIsAuthenticated(false)
  }, [])

  return { isAuthenticated, login, logout, apiKey: getApiKey() }
}
