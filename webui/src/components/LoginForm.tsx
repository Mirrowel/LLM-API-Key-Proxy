import { useState } from "react"
import { Zap } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { getBaseUrl, setBaseUrl } from "@/api/client"

interface LoginFormProps {
  onLogin: (apiKey: string) => void
  error?: string | null
}

export function LoginForm({ onLogin, error }: LoginFormProps) {
  const [apiKey, setApiKey] = useState("")
  const [baseUrl, setBaseUrlState] = useState(getBaseUrl())
  const [showAdvanced, setShowAdvanced] = useState(!!getBaseUrl())

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    setBaseUrl(baseUrl.replace(/\/$/, ""))
    onLogin(apiKey)
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <div className="flex justify-center mb-2">
            <div className="rounded-full bg-primary/10 p-3">
              <Zap className="h-8 w-8 text-primary" />
            </div>
          </div>
          <CardTitle className="text-2xl">LLM API Proxy</CardTitle>
          <CardDescription>Enter your proxy API key to access the dashboard</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium" htmlFor="api-key">
                API Key
              </label>
              <Input
                id="api-key"
                type="password"
                placeholder="sk-proxy-..."
                value={apiKey}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setApiKey(e.target.value)}
                autoFocus
              />
            </div>

            {error && (
              <p className="text-sm text-destructive">{error}</p>
            )}

            <Button type="submit" className="w-full">
              Sign In
            </Button>

            <button
              type="button"
              className="text-xs text-muted-foreground hover:text-foreground w-full text-center"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              {showAdvanced ? "Hide" : "Show"} advanced options
            </button>

            {showAdvanced && (
              <div className="space-y-2">
                <label className="text-sm font-medium" htmlFor="base-url">
                  Remote Proxy URL
                </label>
                <Input
                  id="base-url"
                  type="url"
                  placeholder="https://proxy.example.com (leave empty for same origin)"
                  value={baseUrl}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setBaseUrlState(e.target.value)}
                />
                <p className="text-xs text-muted-foreground">
                  Leave empty to use the current server. Set to connect to a remote proxy instance.
                </p>
              </div>
            )}
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
