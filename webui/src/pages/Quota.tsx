import { useState, useCallback, useMemo } from "react"
import { RefreshCw, ChevronDown, ChevronRight, ArrowLeft, ArrowUpDown, DollarSign, Clock } from "lucide-react"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui/table"
import { usePolling } from "@/hooks/usePolling"
import {
  getQuotaStats,
  reloadQuotaStats,
  forceRefreshQuota,
  type QuotaStatsResponse,
  type ProviderStats,
  type CredentialStats,
  type QuotaGroup,
  type WindowInfo,
  type ModelUsageEntry,
} from "@/api/quota"
import { formatNumber, formatCost, getQuotaColor, formatWindowLabel, formatQuotaValue, formatTimeRemaining, isXaiPercentOnlyQuotaGroup, formatXaiQuotaValueStr, formatPercentUsedFromRemaining } from "@/lib/utils"

function shortenModelName(model: string): string {
  const m = model.toLowerCase().replace(/^(models\/|publishers\/google\/models\/)/, "")
  const stripped = m.replace(/^gemini-|^gemma-/, "")
  if (stripped.startsWith("flash-lite") || stripped.startsWith("3.5-flash-lite") || stripped.startsWith("3.1-flash-lite")) return "flash-lite"
  if (stripped.includes("flash")) return "flash"
  if (stripped.includes("pro")) return "pro"
  if (m.startsWith("gemma-")) {
    const rest = m.replace(/^gemma-/, "").replace(/-it$/, "")
    return `gemma-${rest}`
  }
  if (stripped.startsWith("embedding")) {
    const ver = stripped.match(/embedding-?(\d+)/)?.[1]
    return ver ? `embedding-${ver}` : "embedding"
  }
  return stripped.length > 12 ? stripped.slice(0, 12) : stripped
}

export function Quota() {
  const [viewMode, setViewMode] = useState<"current" | "global">("current")
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)

  const { data, loading, refresh } = usePolling<QuotaStatsResponse>({
    fetcher: () => getQuotaStats(),
    interval: 10000,
  })

  const handleReload = useCallback(async (scope: "all" | "provider", provider?: string) => {
    setRefreshing(true)
    try {
      await reloadQuotaStats(scope, provider)
      await refresh()
    } finally {
      setRefreshing(false)
    }
  }, [refresh])

  const handleForceRefresh = useCallback(async (scope: "all" | "provider" | "credential", provider?: string, credential?: string) => {
    setRefreshing(true)
    try {
      await forceRefreshQuota(scope, provider, credential)
      await refresh()
    } finally {
      setRefreshing(false)
    }
  }, [refresh])

  const [sortCol, setSortCol] = useState<string>("provider")
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc")

  const toggleSort = useCallback((col: string) => {
    if (sortCol === col) {
      setSortDir(d => d === "asc" ? "desc" : "asc")
    } else {
      setSortCol(col)
      setSortDir(col === "provider" ? "asc" : "desc")
    }
  }, [sortCol])

  const providerEntries = useMemo(() => {
    const raw = data?.providers ? Object.entries(data.providers) : []
    const hasQuota = (p: ProviderStats) =>
      p.quota_groups && Object.keys(p.quota_groups).length > 0
    return raw.sort(([aName, a], [bName, b]) => {
      const aQ = hasQuota(a) ? 0 : 1
      const bQ = hasQuota(b) ? 0 : 1
      if (aQ !== bQ) return aQ - bQ
      const getStat = (p: ProviderStats) => {
        const s = viewMode === "current" && p.current_period ? p.current_period : null
        return {
          requests: s?.total_requests ?? p.total_requests ?? 0,
          tokensIn: (s?.tokens?.input_uncached ?? p.tokens?.input_uncached ?? 0) + (s?.tokens?.input_cached ?? p.tokens?.input_cached ?? 0),
          tokensOut: s?.tokens?.output ?? p.tokens?.output ?? 0,
          cost: s?.approx_cost ?? p.approx_cost ?? 0,
        }
      }
      const sa = getStat(a), sb = getStat(b)
      let cmp = 0
      switch (sortCol) {
        case "provider": cmp = aName.localeCompare(bName); break
        case "credentials": cmp = a.credential_count - b.credential_count; break
        case "requests": cmp = sa.requests - sb.requests; break
        case "tokens_in": cmp = sa.tokensIn - sb.tokensIn; break
        case "tokens_out": cmp = sa.tokensOut - sb.tokensOut; break
        case "cost": cmp = sa.cost - sb.cost; break
        default: cmp = 0
      }
      return sortDir === "asc" ? cmp : -cmp
    })
  }, [data, sortCol, sortDir, viewMode])

  if (selectedProvider && data?.providers) {
    const provider = data.providers[selectedProvider]
    if (provider) {
      return (
        <ProviderDetail
          providerName={selectedProvider}
          provider={provider}
          viewMode={viewMode}
          setViewMode={setViewMode}
          onBack={() => setSelectedProvider(null)}
          onReload={() => handleReload("provider", selectedProvider)}
          onForceRefresh={(credential) =>
            handleForceRefresh(credential ? "credential" : "provider", selectedProvider, credential)
          }
          refreshing={refreshing}
        />
      )
    }
  }

  const summary = viewMode === "global" && data?.global_summary ? data.global_summary : data?.summary

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Quota Statistics</h1>
          {data && (
            <p className="text-sm text-muted-foreground">
              Last updated: {new Date(data.timestamp * 1000).toLocaleTimeString()}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Tabs value={viewMode} onValueChange={(v: string) => setViewMode(v as "current" | "global")}>
            <TabsList>
              <TabsTrigger value="current">Current</TabsTrigger>
              <TabsTrigger value="global">Global</TabsTrigger>
            </TabsList>
          </Tabs>
          <Button variant="outline" size="sm" onClick={() => handleReload("all")} disabled={refreshing || loading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? "animate-spin" : ""}`} />
            Reload
          </Button>
          <Button variant="outline" size="sm" onClick={() => handleForceRefresh("all")} disabled={refreshing || loading}>
            Force Refresh
          </Button>
        </div>
      </div>

      {summary && (
        <div className="grid gap-4 sm:grid-cols-4">
          <SummaryCard label="Credentials" value={summary.total_credentials} />
          <SummaryCard label="Requests" value={formatNumber(summary.total_requests)} />
          <SummaryCard
            label="Tokens"
            value={formatNumber(
              (summary.tokens?.input_uncached ?? 0) + (summary.tokens?.input_cached ?? 0) + (summary.tokens?.output ?? 0)
            )}
          />
          <SummaryCard label="Cost" value={formatCost(summary.approx_total_cost)} />
        </div>
      )}

      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <SortableHead col="provider" label="Provider" current={sortCol} dir={sortDir} onClick={toggleSort} />
                <SortableHead col="credentials" label="Credentials" current={sortCol} dir={sortDir} onClick={toggleSort} className="text-center" />
                <TableHead>Quota</TableHead>
                <SortableHead col="requests" label="Requests" current={sortCol} dir={sortDir} onClick={toggleSort} className="text-right" />
                <SortableHead col="tokens_in" label="Tokens In" current={sortCol} dir={sortDir} onClick={toggleSort} className="text-right" />
                <SortableHead col="tokens_out" label="Tokens Out" current={sortCol} dir={sortDir} onClick={toggleSort} className="text-right" />
                <SortableHead col="cost" label="Cost" current={sortCol} dir={sortDir} onClick={toggleSort} className="text-right" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {providerEntries.map(([name, p]) => {
                const stats = viewMode === "current" && p.current_period ? p.current_period : null
                const requests = stats?.total_requests ?? p.total_requests ?? 0
                const tokensIn = (stats?.tokens?.input_uncached ?? p.tokens?.input_uncached ?? 0) + (stats?.tokens?.input_cached ?? p.tokens?.input_cached ?? 0)
                const tokensOut = stats?.tokens?.output ?? p.tokens?.output ?? 0
                const cachePct = stats?.tokens?.input_cache_pct ?? p.tokens?.input_cache_pct ?? 0
                const cost = stats?.approx_cost ?? p.approx_cost ?? 0
                return (
                  <TableRow
                    key={name}
                    className="cursor-pointer"
                    onClick={() => setSelectedProvider(name)}
                  >
                    <TableCell className="font-medium">
                      <div className="flex items-center gap-2">
                        {name}
                        <Badge variant="secondary" className="text-[10px]">{p.rotation_mode}</Badge>
                      </div>
                    </TableCell>
                    <TableCell className="text-center">
                      <div className="flex items-center justify-center gap-1">
                        <span>{p.credential_count}</span>
                        {p.exhausted_count > 0 && (
                          <Badge variant="destructive" className="text-[10px]">{p.exhausted_count} exh</Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <QuotaSummaryBars providerName={name} quotaGroups={p.quota_groups} credentials={p.credentials} />
                    </TableCell>
                    <TableCell className="text-right">{formatNumber(requests)}</TableCell>
                    <TableCell className="text-right">
                      {formatNumber(tokensIn)}
                      {cachePct > 0 && (
                        <span className="text-xs text-muted-foreground ml-1">
                          ({cachePct.toFixed(0)}% cached)
                        </span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">{formatNumber(tokensOut)}</TableCell>
                    <TableCell className="text-right">{formatCost(cost)}</TableCell>
                  </TableRow>
                )
              })}
              {!providerEntries.length && (
                <TableRow>
                  <TableCell colSpan={7} className="text-center text-muted-foreground py-8">
                    {loading ? "Loading..." : "No providers found"}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}

function SummaryCard({ label, value }: { label: string; value: string | number }) {
  return (
    <Card>
      <CardContent className="p-4">
        <p className="text-sm text-muted-foreground">{label}</p>
        <p className="text-xl font-bold">{value}</p>
      </CardContent>
    </Card>
  )
}

function QuotaSummaryBars({
  providerName,
  quotaGroups,
  credentials,
}: {
  providerName?: string
  quotaGroups?: Record<string, QuotaGroup>
  credentials?: Record<string, CredentialStats>
}) {
  const bars: { label: string; key: string; pct: number; valueStr: string; pctSuffix?: string }[] = []

  const xaiResetAt = (() => {
    if (providerName !== "x-ai" || !credentials) return null
    for (const c of Object.values(credentials)) {
      const gu = c.group_usage?.["monthly-limit"]?.windows
      if (!gu) continue
      for (const w of Object.values(gu)) {
        if (w.reset_at) return w.reset_at
      }
    }
    return null
  })()

  if (quotaGroups) {
    for (const [groupName, group] of Object.entries(quotaGroups)) {
      const hasAnyLimit = Object.values(group.windows).some(w => (w.total_max ?? 0) > 0)
      if (!hasAnyLimit) continue
      const windowEntries = Object.entries(group.windows)
      for (const [windowName, win] of windowEntries) {
        if ((win.total_max ?? 0) === 0) continue
        const label =
          providerName === "x-ai" && groupName === "monthly-limit"
            ? "SuperGrok credits"
            : windowEntries.length > 1
              ? `${groupName}/${formatWindowLabel(windowName)}`
              : groupName
        const percentOnly =
          providerName != null && isXaiPercentOnlyQuotaGroup(providerName, groupName)
        const valueStr = percentOnly
          ? formatXaiQuotaValueStr(win.remaining_pct, xaiResetAt)
          : `${formatQuotaValue(win.total_remaining, groupName)}/${formatQuotaValue(win.total_max, groupName)}`
        bars.push({
          label,
          key: `${groupName}-${windowName}`,
          pct: win.remaining_pct ?? 0,
          valueStr,
          pctSuffix: percentOnly
            ? formatPercentUsedFromRemaining(win.remaining_pct).replace(" used", "")
            : undefined,
        })
      }
    }
  }

  if (!bars.length && credentials) {
    const creds = Object.values(credentials)
    let budgetTotal = 0, budgetSpent = 0, hasBudget = false
    for (const c of creds) {
      if (c.monthly_budget) {
        hasBudget = true
        budgetTotal += c.monthly_budget.budget
        budgetSpent += c.monthly_budget.spent
      }
    }
    if (hasBudget && budgetTotal > 0) {
      const remaining = budgetTotal - budgetSpent
      const pct = (remaining / budgetTotal) * 100
      bars.push({ label: "monthly($)", key: "budget", pct, valueStr: `${formatCost(remaining)}/${formatCost(budgetTotal)}` })
    }

    const rpdAgg: Record<string, { used: number; limit: number }> = {}
    for (const c of creds) {
      if (!c.rpd_limits) continue
      for (const [model, info] of Object.entries(c.rpd_limits)) {
        if (!rpdAgg[model]) rpdAgg[model] = { used: 0, limit: 0 }
        rpdAgg[model].used += info.used
        rpdAgg[model].limit += info.limit
      }
    }
    const rpdModels = Object.entries(rpdAgg).sort(
      ([, a], [, b]) => (a.limit - a.used) / Math.max(a.limit, 1) - (b.limit - b.used) / Math.max(b.limit, 1)
    )
    for (const [model, agg] of rpdModels) {
      const remaining = Math.max(0, agg.limit - agg.used)
      const pct = agg.limit > 0 ? (remaining / agg.limit) * 100 : 0
      bars.push({
        label: shortenModelName(model), key: `rpd-${model}`, pct,
        valueStr: `${formatNumber(remaining)}/${formatNumber(agg.limit)}`,
      })
    }
  }

  if (!bars.length) return <span className="text-muted-foreground text-xs">—</span>

  return (
    <div className="space-y-1.5 max-w-[250px]">
      {bars.slice(0, 6).map((w) => (
        <div key={w.key}>
          <div className="flex justify-between text-[10px] text-muted-foreground mb-0.5">
            <span className="truncate">{w.label}</span>
            <span className="whitespace-nowrap ml-1">{w.valueStr}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Progress
              value={w.pct}
              className="h-1.5 flex-1"
              indicatorClassName={getQuotaColor(w.pct)}
            />
            <span className="text-[10px] text-muted-foreground w-10 text-right shrink-0">
              {w.pctSuffix ?? `${w.pct.toFixed(0)}%`}
            </span>
          </div>
        </div>
      ))}
      {bars.length > 6 && (
        <span className="text-[10px] text-muted-foreground">+{bars.length - 6} more</span>
      )}
    </div>
  )
}

function ProviderDetail({
  providerName,
  provider,
  viewMode,
  setViewMode,
  onBack,
  onReload,
  onForceRefresh,
  refreshing,
}: {
  providerName: string
  provider: ProviderStats
  viewMode: "current" | "global"
  setViewMode: (v: "current" | "global") => void
  onBack: () => void
  onReload: () => void
  onForceRefresh: (credential?: string) => void
  refreshing: boolean
}) {
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set())

  function toggleModels(credId: string) {
    setExpandedModels((prev) => {
      const next = new Set(prev)
      if (next.has(credId)) next.delete(credId)
      else next.add(credId)
      return next
    })
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={onBack}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">{providerName}</h1>
            <p className="text-sm text-muted-foreground">
              {provider.credential_count} credentials &middot; {provider.rotation_mode} rotation
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Tabs value={viewMode} onValueChange={(v: string) => setViewMode(v as "current" | "global")}>
            <TabsList>
              <TabsTrigger value="current">Current</TabsTrigger>
              <TabsTrigger value="global">Global</TabsTrigger>
            </TabsList>
          </Tabs>
          <Button variant="outline" size="sm" onClick={onReload} disabled={refreshing}>
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? "animate-spin" : ""}`} />
            Reload
          </Button>
          <Button variant="outline" size="sm" onClick={() => onForceRefresh()} disabled={refreshing}>
            Force Refresh All
          </Button>
        </div>
      </div>

      {provider.quota_groups && Object.entries(provider.quota_groups).some(([, g]) =>
        Object.values(g.windows).some(w => (w.total_max ?? 0) > 0)
      ) && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Quota Groups</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {(Object.entries(provider.quota_groups) as [string, QuotaGroup][])
                .filter(([, group]) => Object.values(group.windows).some(w => (w.total_max ?? 0) > 0))
                .map(([groupName, group]) => (
                <div key={groupName}>
                  <h4 className="text-sm font-medium mb-2">
                    {isXaiPercentOnlyQuotaGroup(providerName, groupName) ? "SuperGrok credits" : groupName}
                  </h4>
                  <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                    {(Object.entries(group.windows) as [string, WindowInfo][])
                      .filter(([, win]) => (win.total_max ?? 0) > 0)
                      .map(([windowName, win]) => (
                      <div key={windowName} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span>
                            {isXaiPercentOnlyQuotaGroup(providerName, groupName)
                              ? "SuperGrok credits"
                              : Object.keys(group.windows).length > 1
                                ? formatWindowLabel(windowName)
                                : groupName}
                          </span>
                          <span>
                            {isXaiPercentOnlyQuotaGroup(providerName, groupName)
                              ? formatXaiQuotaValueStr(
                                  win.remaining_pct,
                                  Object.values(provider.credentials ?? {})[0]?.group_usage?.[groupName]?.windows?.[windowName]?.reset_at ??
                                    Object.values(provider.credentials ?? {}).flatMap(c =>
                                      Object.values(c.group_usage?.[groupName]?.windows ?? {}),
                                    )[0]?.reset_at,
                                )
                              : `${formatQuotaValue(win.total_remaining, groupName)}/${formatQuotaValue(win.total_max, groupName)}`}
                          </span>
                        </div>
                        <Progress
                          value={win.remaining_pct}
                          className="h-2"
                          indicatorClassName={getQuotaColor(win.remaining_pct)}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <div className="space-y-4">
        <h2 className="text-lg font-semibold">Credentials</h2>
        {Object.entries(provider.credentials).map(([credId, cred]: [string, CredentialStats]) => (
          <CredentialCard
            key={credId}
            providerName={providerName}
            cred={cred}
            viewMode={viewMode}
            showModels={expandedModels.has(credId)}
            onToggleModels={() => toggleModels(credId)}
            onForceRefresh={() => onForceRefresh(cred.full_path || credId)}
            refreshing={refreshing}
          />
        ))}
      </div>
    </div>
  )
}

function resolveModelUsage(entry: ModelUsageEntry): { request_count: number; approx_cost: number } {
  if (entry.totals) {
    return { request_count: entry.totals.request_count ?? 0, approx_cost: entry.totals.approx_cost ?? 0 }
  }
  return { request_count: entry.request_count ?? 0, approx_cost: entry.approx_cost ?? 0 }
}

function CredentialCard({
  providerName,
  cred,
  viewMode,
  showModels,
  onToggleModels,
  onForceRefresh,
  refreshing,
}: {
  providerName?: string
  cred: CredentialStats
  viewMode: "current" | "global"
  showModels: boolean
  onToggleModels: () => void
  onForceRefresh: () => void
  refreshing: boolean
}) {
  const statusVariant = cred.status === "active" ? "success"
    : cred.status === "cooldown" ? "warning"
    : cred.status === "needs_reauth" || cred.status === "error" ? "destructive"
    : cred.status === "exhausted" ? "destructive"
    : "secondary"

  const statusTooltips: Record<string, string> = {
    mixed: "Some quota windows are active while others are exhausted or on cooldown",
    needs_reauth: "OAuth token expired — re-authenticate with --add-credential",
    cooldown: "Temporarily rate-limited, will recover automatically",
    exhausted: "All quota windows exhausted for this credential",
  }

  const usePeriod = viewMode === "current" && cred.current_period
  const requestCount = usePeriod ? cred.current_period!.request_count : cred.totals.request_count
  const tokensIn = usePeriod ? cred.current_period!.prompt_tokens : cred.totals.prompt_tokens
  const tokensOut = usePeriod ? cred.current_period!.output_tokens : cred.totals.completion_tokens
  const cost = usePeriod ? cred.current_period!.approx_cost : cred.totals.approx_cost

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <CardTitle className="text-sm font-mono">{cred.accessor_masked}</CardTitle>
            <span title={statusTooltips[cred.status] || ""}>
              <Badge variant={statusVariant} className={statusTooltips[cred.status] ? "cursor-help" : ""}>{cred.status}</Badge>
            </span>
            {cred.email && <span className="text-xs text-muted-foreground">{cred.email}</span>}
            {cred.tier && <Badge variant="outline" className="text-[10px]">{cred.tier}</Badge>}
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={onForceRefresh} disabled={refreshing}>
              <RefreshCw className="h-3 w-3 mr-1" /> Refresh
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 sm:grid-cols-4 text-sm mb-3">
          <div>
            <span className="text-muted-foreground">Requests</span>
            <p className="font-medium">{formatNumber(requestCount)}</p>
          </div>
          <div>
            <span className="text-muted-foreground">Tokens In</span>
            <p className="font-medium">{formatNumber(tokensIn)}</p>
          </div>
          <div>
            <span className="text-muted-foreground">Tokens Out</span>
            <p className="font-medium">{formatNumber(tokensOut)}</p>
          </div>
          <div>
            <span className="text-muted-foreground">Cost</span>
            <p className="font-medium">{formatCost(cost)}</p>
          </div>
        </div>

        {cred.group_usage && Object.entries(cred.group_usage).some(([, g]) =>
          Object.values(g.windows).some(w => w.limit != null)
        ) && (
          <div className="mb-3">
            <h4 className="text-xs font-medium mb-2">Quota Usage</h4>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {Object.entries(cred.group_usage)
                .filter(([, group]) => Object.values(group.windows).some(w => w.limit != null))
                .map(([groupName, group]) =>
                Object.entries(group.windows)
                  .filter(([, win]) => win.limit != null)
                  .map(([windowName, win]) => {
                  const pct = win.limit > 0 ? ((win.remaining / win.limit) * 100) : 0
                  const windowCount = Object.keys(group.windows).length
                  const percentOnly =
                    providerName != null && isXaiPercentOnlyQuotaGroup(providerName, groupName)
                  const resetStr =
                    !percentOnly &&
                    win.reset_at &&
                    (win.request_count > 0 || (group.cooldown_remaining ?? 0) > 0)
                      ? formatTimeRemaining(win.reset_at)
                      : null
                  const label =
                    percentOnly
                      ? "SuperGrok credits"
                      : windowCount > 1
                        ? `${groupName}/${formatWindowLabel(windowName)}`
                        : groupName
                  return (
                    <div key={`${groupName}-${windowName}`} className="space-y-1">
                      <div className="flex justify-between text-[11px]">
                        <span className="truncate">{label}</span>
                        <span>
                          {percentOnly
                            ? formatXaiQuotaValueStr(pct, win.reset_at)
                            : `${formatQuotaValue(win.remaining, groupName)}/${formatQuotaValue(win.limit, groupName)}`}
                        </span>
                      </div>
                      <Progress
                        value={pct}
                        className="h-1.5"
                        indicatorClassName={getQuotaColor(pct)}
                      />
                      {resetStr && (
                        <div className="text-[10px] text-muted-foreground">
                          Resets {resetStr === "now" ? "now" : `in ${resetStr}`}
                        </div>
                      )}
                    </div>
                  )
                })
              )}
            </div>
          </div>
        )}

        {cred.monthly_budget && (
          <div className="mb-3">
            <h4 className="text-xs font-medium mb-2 flex items-center gap-1">
              <DollarSign className="h-3 w-3" /> Monthly Budget
            </h4>
            <div className="space-y-1">
              <div className="flex justify-between text-[11px]">
                <span>
                  {formatCost(cred.monthly_budget.spent)} / {formatCost(cred.monthly_budget.budget)}
                </span>
                <span className="text-muted-foreground">
                  {cred.monthly_budget.remaining > 0
                    ? `${formatCost(cred.monthly_budget.remaining)} remaining`
                    : "Budget exhausted"}
                </span>
              </div>
              <Progress
                value={100 - cred.monthly_budget.percent_used}
                className="h-1.5"
                indicatorClassName={getQuotaColor(100 - cred.monthly_budget.percent_used)}
              />
              <div className="text-[10px] text-muted-foreground">
                Resets {formatTimeRemaining(cred.monthly_budget.reset_at) === "now"
                  ? "now"
                  : `in ${formatTimeRemaining(cred.monthly_budget.reset_at)}`}
                {" "}(day {cred.monthly_budget.reset_day})
              </div>
            </div>
          </div>
        )}

        {cred.rpd_limits && Object.keys(cred.rpd_limits).length > 0 && (
          <div className="mb-3">
            <h4 className="text-xs font-medium mb-2 flex items-center gap-1">
              <Clock className="h-3 w-3" /> RPD Limits ({Object.keys(cred.rpd_limits).length} models)
            </h4>
            <div className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3">
              {Object.entries(cred.rpd_limits)
                .sort(([, a], [, b]) => (a.remaining / Math.max(a.limit, 1)) - (b.remaining / Math.max(b.limit, 1)))
                .map(([model, info]) => {
                const pct = info.limit > 0 ? (info.remaining / info.limit) * 100 : 0
                return (
                  <div key={model} className="space-y-0.5">
                    <div className="flex justify-between text-[10px]">
                      <span className="font-mono truncate">{model}</span>
                      <span className="ml-1 whitespace-nowrap">
                        {info.limit === 0
                          ? "blocked"
                          : `${info.used}/${info.limit}`}
                      </span>
                    </div>
                    <Progress
                      value={pct}
                      className="h-1"
                      indicatorClassName={getQuotaColor(pct)}
                    />
                  </div>
                )
              })}
            </div>
            {(() => {
              const firstReset = Object.values(cred.rpd_limits)[0]?.reset_at
              if (!firstReset) return null
              const resetStr = formatTimeRemaining(firstReset)
              return (
                <div className="text-[10px] text-muted-foreground mt-1">
                  Resets {resetStr === "now" ? "now" : `in ${resetStr}`}
                </div>
              )
            })()}
          </div>
        )}

        {cred.model_usage && Object.keys(cred.model_usage).length > 0 && (
          <div>
            <button
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
              onClick={onToggleModels}
            >
              {showModels ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
              Model usage ({Object.keys(cred.model_usage).length} models)
            </button>
            {showModels && (
              <div className="mt-2 border rounded-md overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="text-xs">Model</TableHead>
                      <TableHead className="text-xs text-right">Requests</TableHead>
                      <TableHead className="text-xs text-right">Cost</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {Object.entries(cred.model_usage).map(([model, usage]) => {
                      const stats = resolveModelUsage(usage)
                      return (
                        <TableRow key={model}>
                          <TableCell className="text-xs font-mono">{model}</TableCell>
                          <TableCell className="text-xs text-right">{stats.request_count}</TableCell>
                          <TableCell className="text-xs text-right">{formatCost(stats.approx_cost)}</TableCell>
                        </TableRow>
                      )
                    })}
                  </TableBody>
                </Table>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function SortableHead({ col, label, current, dir, onClick, className }: {
  col: string; label: string; current: string; dir: "asc" | "desc"
  onClick: (col: string) => void; className?: string
}) {
  const active = current === col
  return (
    <TableHead className={`cursor-pointer select-none hover:text-foreground ${className ?? ""}`} onClick={() => onClick(col)}>
      <span className="inline-flex items-center gap-1">
        {label}
        <ArrowUpDown className={`h-3 w-3 ${active ? "text-foreground" : "text-muted-foreground/50"}`} />
        {active && <span className="text-[10px]">{dir === "asc" ? "\u25b2" : "\u25bc"}</span>}
      </span>
    </TableHead>
  )
}
