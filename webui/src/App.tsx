import { useState, useCallback } from "react"
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { Shell } from "@/components/layout/Shell"
import { LoginForm } from "@/components/LoginForm"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import { Dashboard } from "@/pages/Dashboard"
import { Quota } from "@/pages/Quota"
import { Logs } from "@/pages/Logs"
import { Credentials } from "@/pages/Credentials"
import { Models } from "@/pages/Models"
import { Settings } from "@/pages/Settings"
import { useAuth } from "@/hooks/useAuth"
import { proxyWs } from "@/api/websocket"

export default function App() {
  const { isAuthenticated, login, logout } = useAuth()
  const [loginError, setLoginError] = useState<string | null>(null)

  const handleLogin = useCallback(async (apiKey: string) => {
    setLoginError(null)
    try {
      const response = await fetch("/v1/health", {
        headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
      })
      if (response.status === 401) {
        setLoginError("Invalid API key")
        return
      }
    } catch {
      // connection error — let them proceed, they'll see errors in the UI
    }
    login(apiKey)
    proxyWs.reconnect()
  }, [login])

  const handleLogout = useCallback(() => {
    proxyWs.disconnect()
    logout()
    setLoginError(null)
  }, [logout])

  if (!isAuthenticated) {
    return <LoginForm onLogin={handleLogin} error={loginError} />
  }

  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route element={<Shell onLogout={handleLogout} />}>
            <Route path="/ui" element={<Dashboard />} />
            <Route path="/ui/quota" element={<Quota />} />
            <Route path="/ui/logs" element={<Logs />} />
            <Route path="/ui/credentials" element={<Credentials />} />
            <Route path="/ui/models" element={<Models />} />
            <Route path="/ui/settings" element={<Settings />} />
          </Route>
          <Route path="*" element={<Navigate to="/ui" replace />} />
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  )
}
