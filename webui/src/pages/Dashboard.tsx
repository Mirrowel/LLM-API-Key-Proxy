import {
  Activity,
  Server,
  KeyRound,
  AlertTriangle,
  Clock,
  Cpu,
  ArrowUpRight,
  RefreshCw,
} from "lucide-react"
import { Link } from "react-router-dom"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { WidgetBoundary } from "@/components/ErrorBoundary"
import { usePolling } from "@/hooks/usePolling"
import { getHealth, getHealthErrors, type HealthResponse, type ErrorRecord } from "@/api/health"
import { getQuotaStats, type QuotaStatsResponse } from "@/api/quota"
import { getFailures, type FailureEntry } from "@/api/logs"
import { formatNumber, formatCost, timeAgo, formatUptime } from "@/lib/utils"

export function Dashboard() {
  const { data: health, loading: healthLoading, refresh: refreshHealth } = usePolling<HealthResponse>({
    fetcher: () => getHealth("full"),
    interval: 15000,
  })

  const { data: errors } = usePolling<{ errors: ErrorRecord[]; total_matching: number }>({
    fetcher: () => getHealthErrors({ limit: 10 }),
    interval: 15000,
  })

  const { data: recentFailures } = usePolling<{ failures: FailureEntry[]; total: number }>({
    fetcher: () => getFailures({ page_size: 10 }),
    interval: 30000,
  })

  const { data: quota } = usePolling<QuotaStatsResponse>({
    fetcher: () => getQuotaStats(),
    interval: 15000,
  })

  const summary = quota?.summary
  const providerCount = health?.providers?.total ?? 0
  const credCount = health?.credentials?.total ?? 0
  const activeCount = health?.credentials?.active ?? 0
  const cooldownCount = health?.credentials?.on_cooldown ?? 0
  const exhaustedCount = health?.credentials?.exhausted ?? 0
  const errorCount = health?.credentials?.error ?? 0

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground">
            {health ? `Uptime: ${formatUptime(health.uptime_seconds)}` : "Loading..."}
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={refreshHealth} disabled={healthLoading}>
          <RefreshCw className={`h-4 w-4 mr-2 ${healthLoading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      <WidgetBoundary>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            title="Providers"
            value={providerCount}
            icon={<Server className="h-4 w-4 text-muted-foreground" />}
            description={`${credCount} total credentials`}
          />
          <StatCard
            title="Credentials"
            value={credCount}
            icon={<KeyRound className="h-4 w-4 text-muted-foreground" />}
            description={
              <span className="flex items-center gap-2 flex-wrap">
                <Badge variant="success" className="text-[10px]">{activeCount} active</Badge>
                {cooldownCount > 0 && <Badge variant="warning" className="text-[10px]">{cooldownCount} cooldown</Badge>}
                {exhaustedCount > 0 && <Badge variant="destructive" className="text-[10px]">{exhaustedCount} exhausted</Badge>}
                {errorCount > 0 && <Badge variant="destructive" className="text-[10px]">{errorCount} error</Badge>}
              </span>
            }
          />
          <StatCard
            title="Requests"
            value={formatNumber(summary?.total_requests ?? 0)}
            icon={<Activity className="h-4 w-4 text-muted-foreground" />}
            description={`${formatNumber((summary?.tokens?.input_uncached ?? 0) + (summary?.tokens?.input_cached ?? 0) + (summary?.tokens?.output ?? 0))} tokens`}
          />
          <StatCard
            title="Cost"
            value={formatCost(summary?.approx_total_cost ?? 0)}
            icon={<Cpu className="h-4 w-4 text-muted-foreground" />}
            description="Approximate total"
          />
        </div>
      </WidgetBoundary>

      {health?.errors && health.errors.total_errors > 0 && (
        <WidgetBoundary>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-base">Error Summary ({health.errors.total_errors})</CardTitle>
              <Link to="/ui/logs" className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1">
                View all <ArrowUpRight className="h-3 w-3" />
              </Link>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 sm:grid-cols-3">
                <div>
                  <h4 className="text-sm font-medium mb-2">By Provider</h4>
                  <div className="space-y-1">
                    {Object.entries(health.errors.by_provider).map(([provider, info]) => (
                      <div key={provider} className="flex items-center justify-between text-sm">
                        <span>{provider}</span>
                        <div className="flex items-center gap-1.5">
                          {Object.entries(info.error_types || {}).slice(0, 2).map(([et, c]) => (
                            <Badge key={et} variant="outline" className="text-[9px] px-1">{et} {c}</Badge>
                          ))}
                          <span className="text-muted-foreground ml-1">{info.count}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-medium mb-2">By Model</h4>
                  <div className="space-y-1">
                    {Object.entries(health.errors.by_model).map(([model, info]) => (
                      <div key={model} className="flex items-center justify-between text-sm">
                        <span className="truncate mr-2">{model}</span>
                        <span className="text-muted-foreground shrink-0">{info.count}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-medium mb-2">By Error Type</h4>
                  <div className="space-y-1">
                    {(() => {
                      const typeCounts: Record<string, number> = {}
                      for (const info of Object.values(health.errors!.by_provider)) {
                        for (const [et, c] of Object.entries(info.error_types || {})) {
                          typeCounts[et] = (typeCounts[et] || 0) + c
                        }
                      }
                      return Object.entries(typeCounts).sort((a, b) => b[1] - a[1]).map(([et, c]) => (
                        <div key={et} className="flex justify-between text-sm">
                          <span>{et}</span>
                          <span className="text-muted-foreground">{c}</span>
                        </div>
                      ))
                    })()}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </WidgetBoundary>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <WidgetBoundary>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-base">Provider Overview</CardTitle>
              <Link to="/ui/quota" className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1">
                View all <ArrowUpRight className="h-3 w-3" />
              </Link>
            </CardHeader>
            <CardContent>
              {quota?.providers && Object.keys(quota.providers).length ? (
                <div className="space-y-3">
                  {Object.entries(quota.providers)
                    .sort(([, a], [, b]) => (b.total_requests ?? 0) - (a.total_requests ?? 0))
                    .map(([name, p]) => (
                    <div key={name} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-sm">{name}</span>
                        <Badge variant="secondary" className="text-[10px]">
                          {p.credential_count} cred{p.credential_count !== 1 ? "s" : ""}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-3 text-xs text-muted-foreground">
                        <span>{formatNumber(p.total_requests)} req</span>
                        <span>{formatCost(p.approx_cost ?? 0)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No providers configured</p>
              )}
            </CardContent>
          </Card>
        </WidgetBoundary>

        <WidgetBoundary>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-base">Recent Errors</CardTitle>
              <Link to="/ui/logs" className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1">
                View all <ArrowUpRight className="h-3 w-3" />
              </Link>
            </CardHeader>
            <CardContent>
              {errors?.errors.length ? (
                <div className="space-y-2">
                  {errors.errors.slice(0, 5).map((err: ErrorRecord, i: number) => (
                    <div key={i} className="flex items-start gap-2 text-sm">
                      <AlertTriangle className="h-4 w-4 text-destructive mt-0.5 shrink-0" />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{err.provider}</span>
                          <Badge variant="outline" className="text-[10px]">{err.error_type}</Badge>
                          {err.credential && (
                            <Badge variant="secondary" className="text-[10px] font-mono">{err.credential}</Badge>
                          )}
                          <span className="text-xs text-muted-foreground ml-auto shrink-0">
                            <Clock className="h-3 w-3 inline mr-0.5" />
                            {timeAgo(err.timestamp)}
                          </span>
                        </div>
                        <p className="text-xs text-muted-foreground truncate">{err.error_message}</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : recentFailures?.failures.length ? (
                <div className="space-y-2">
                  {recentFailures.failures.slice(0, 5).map((f: FailureEntry, i: number) => (
                    <div key={i} className="flex items-start gap-2 text-sm">
                      <AlertTriangle className="h-4 w-4 text-destructive mt-0.5 shrink-0" />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{f.provider || f.model?.split("/")[0] || "unknown"}</span>
                          <Badge variant="outline" className="text-[10px]">{f.error_type}</Badge>
                          {f.api_key_ending && (
                            <Badge variant="secondary" className="text-[10px] font-mono">...{f.api_key_ending}</Badge>
                          )}
                          <span className="text-xs text-muted-foreground ml-auto shrink-0">
                            <Clock className="h-3 w-3 inline mr-0.5" />
                            {timeAgo(f.timestamp)}
                          </span>
                        </div>
                        <p className="text-xs text-muted-foreground truncate">{f.error_message}</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No recent errors</p>
              )}
            </CardContent>
          </Card>
        </WidgetBoundary>
      </div>
    </div>
  )
}

function StatCard({
  title,
  value,
  icon,
  description,
}: {
  title: string
  value: string | number
  icon: React.ReactNode
  description: React.ReactNode
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        <div className="text-xs text-muted-foreground mt-1">{description}</div>
      </CardContent>
    </Card>
  )
}
