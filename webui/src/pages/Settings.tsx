import { useState, useCallback } from "react"
import { RefreshCw, RotateCcw } from "lucide-react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { usePolling } from "@/hooks/usePolling"
import { getConfig, reloadProxy, type ProxyConfig, type ConcurrencyConfig, type ModelFilterConfig } from "@/api/config"

export function Settings() {
  const { data, loading, refresh } = usePolling<ProxyConfig>({
    fetcher: getConfig,
    interval: 30000,
  })
  const [reloading, setReloading] = useState(false)
  const [reloadMessage, setReloadMessage] = useState<string | null>(null)

  const handleReload = useCallback(async () => {
    setReloading(true)
    setReloadMessage(null)
    try {
      const result = await reloadProxy()
      setReloadMessage(result.message || "Proxy reloaded successfully")
      await refresh()
    } catch (err) {
      setReloadMessage(err instanceof Error ? err.message : "Reload failed")
    } finally {
      setReloading(false)
    }
  }, [refresh])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
          <p className="text-muted-foreground">Proxy configuration and advanced settings</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={refresh} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button size="sm" onClick={handleReload} disabled={reloading}>
            <RotateCcw className={`h-4 w-4 mr-2 ${reloading ? "animate-spin" : ""}`} />
            Reload Proxy
          </Button>
        </div>
      </div>

      {reloadMessage && (
        <div className="rounded-md border p-3 text-sm">
          {reloadMessage}
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Proxy Status</CardTitle>
          <CardDescription>Core proxy configuration</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <SettingRow
              label="API Key Protection"
              value={data?.proxy_api_key_set ? "Enabled" : "Disabled"}
              badge={data?.proxy_api_key_set ? "success" : "destructive"}
            />
            <SettingRow
              label="Providers with credentials"
              value={data ? String(Object.values(data.providers).filter(p => p.api_key_count > 0 || p.oauth_count > 0).length) : "-"}
            />
            <SettingRow
              label="Custom API bases"
              value={data ? String(Object.keys(data.custom_providers).length) : "-"}
            />
            <SettingRow
              label="Default rotation mode"
              value="sequential"
            />
          </div>
        </CardContent>
      </Card>

      {data && Object.keys(data.rotation_modes).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Rotation Modes</CardTitle>
            <CardDescription>Per-provider credential rotation strategy</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {Object.entries(data.rotation_modes).map(([provider, mode]: [string, string]) => (
                <SettingRow key={provider} label={provider} value={mode} />
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Concurrency Limits</CardTitle>
          <CardDescription>
            Per-provider request concurrency settings.{" "}
            <span className="inline-flex gap-3 mt-1">
              <span title="Hard ceiling — requests beyond this are queued or rejected"><strong>Max</strong>: hard limit on simultaneous requests per key</span>
              <span title="Soft target — the proxy prefers to stay at or below this level, spreading load across keys"><strong>Optimal</strong>: preferred concurrency target for load balancing</span>
            </span>
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <SettingRow
              label="Default (sequential)"
              value="max: unlimited, optimal: unlimited"
              mono
            />
            {data && Object.entries(data.concurrency).map(([provider, config]: [string, ConcurrencyConfig]) => (
              <SettingRow
                key={provider}
                label={provider}
                value={`max: ${config.max === -1 ? "unlimited" : config.max}, optimal: ${config.optimal === -1 ? "unlimited" : config.optimal}`}
              />
            ))}
          </div>
        </CardContent>
      </Card>

      {data && Object.keys(data.model_filters).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Model Filters</CardTitle>
            <CardDescription>Per-provider ignore and whitelist rules (only showing enabled providers)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(data.model_filters)
                .filter(([provider]: [string, ModelFilterConfig]) => provider in data.providers)
                .map(([provider, filters]: [string, ModelFilterConfig]) => (
                <div key={provider}>
                  <span className="text-sm font-medium">{provider}</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {filters.ignore.map((pattern: string) => (
                      <Badge key={`i-${pattern}`} variant="destructive" className="text-[10px]">
                        ignore: {pattern}
                      </Badge>
                    ))}
                    {filters.whitelist.map((pattern: string) => (
                      <Badge key={`w-${pattern}`} variant="success" className="text-[10px]">
                        whitelist: {pattern}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {data && Object.keys(data.latest_aliases).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Latest Aliases</CardTitle>
            <CardDescription>Smart model alias mappings</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {Object.entries(data.latest_aliases).map(([alias, target]: [string, string]) => (
                <SettingRow key={alias} label={alias} value={target} mono />
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {data && Object.keys(data.custom_providers).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Custom Provider Bases</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {Object.entries(data.custom_providers).map(([name, url]: [string, string]) => (
                <SettingRow key={name} label={name} value={url} mono />
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {data?.proxy_urls && Object.keys(data.proxy_urls).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Outbound Proxies</CardTitle>
            <CardDescription>PROXY_URL_* settings for routing requests through proxies</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {data.proxy_urls.default && (
                <SettingRow label="Default" value={data.proxy_urls.default} mono />
              )}
              {data.proxy_urls.providers && Object.entries(data.proxy_urls.providers).map(([name, url]: [string, string]) => (
                <SettingRow key={name} label={`Provider: ${name}`} value={url} mono />
              ))}
              {data.proxy_urls.credentials && Object.entries(data.proxy_urls.credentials).map(([slug, url]: [string, string]) => {
                const provider = data.proxy_urls?.credential_providers?.[slug]
                const label = provider && provider !== "unknown"
                  ? `${provider} / ${slug.slice(0, 12)}`
                  : `Credential: ${slug}`
                return <SettingRow key={slug} label={label} value={url} mono />
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function SettingRow({
  label,
  value,
  badge,
  mono,
}: {
  label: string
  value: string
  badge?: "success" | "destructive" | "secondary"
  mono?: boolean
}) {
  return (
    <div className="flex items-center justify-between py-1">
      <span className="text-sm">{label}</span>
      {badge ? (
        <Badge variant={badge}>{value}</Badge>
      ) : (
        <span className={`text-sm text-muted-foreground ${mono ? "font-mono" : ""}`}>{value}</span>
      )}
    </div>
  )
}
