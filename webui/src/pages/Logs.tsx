import { useState, useCallback, useEffect, useMemo, useRef } from "react"
import {
  Search,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  FileJson,
  AlertTriangle,
  CheckCircle,
  XCircle,
} from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import {
  getTransactions,
  getTransactionDetail,
  getTransactionFile,
  getFailures,
  type TransactionSummary,
  type TransactionDetail,
  type FailureEntry,
} from "@/api/logs"
import { formatDuration, timeAgo, formatCost, formatNumber } from "@/lib/utils"

function useAutoRefresh(callback: () => void, intervalMs: number | null) {
  const callbackRef = useRef(callback)
  callbackRef.current = callback

  useEffect(() => {
    if (intervalMs === null) return
    const id = setInterval(() => callbackRef.current(), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs])
}

const REFRESH_OPTIONS: { label: string; value: number | null }[] = [
  { label: "Off", value: null },
  { label: "10s", value: 10_000 },
  { label: "30s", value: 30_000 },
  { label: "1m", value: 60_000 },
  { label: "5m", value: 300_000 },
  { label: "10m", value: 600_000 },
]

export function Logs() {
  const [tab, setTab] = useState("transactions")

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Log Explorer</h1>
        <p className="text-muted-foreground">Browse transaction logs and failure records</p>
      </div>

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="transactions">Transactions</TabsTrigger>
          <TabsTrigger value="failures">Failures</TabsTrigger>
        </TabsList>
        <TabsContent value="transactions">
          <TransactionBrowser />
        </TabsContent>
        <TabsContent value="failures">
          <FailureBrowser />
        </TabsContent>
      </Tabs>
    </div>
  )
}

function TransactionBrowser() {
  const [transactions, setTransactions] = useState<TransactionSummary[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState("")
  const [debouncedSearch, setDebouncedSearch] = useState("")
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined)
  const [providerFilter, setProviderFilter] = useState<string | null>(null)
  const [statusFilter, setStatusFilter] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [expandedDetail, setExpandedDetail] = useState<TransactionDetail | null>(null)
  const [fileContent, setFileContent] = useState<{ name: string; content: string } | null>(null)
  const [pageSize, setPageSize] = useState(20)
  const [refreshInterval, setRefreshInterval] = useState<number | null>(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getTransactions({
        page,
        page_size: pageSize,
        search: debouncedSearch || undefined,
        provider: providerFilter ?? undefined,
        status: statusFilter ?? undefined,
      })
      setTransactions(data.transactions)
      setTotal(data.total)
    } catch {
      // handled by empty state
    } finally {
      setLoading(false)
    }
  }, [page, pageSize, debouncedSearch, providerFilter, statusFilter])

  useEffect(() => { fetchData() }, [fetchData])
  useAutoRefresh(fetchData, refreshInterval)

  const toggleExpand = useCallback(async (requestId: string) => {
    if (expanded === requestId) {
      setExpanded(null)
      setExpandedDetail(null)
      setFileContent(null)
      return
    }
    setExpanded(requestId)
    setFileContent(null)
    try {
      const detail = await getTransactionDetail(requestId)
      setExpandedDetail(detail)
    } catch {
      setExpandedDetail(null)
    }
  }, [expanded])

  async function viewFile(requestId: string, filename: string) {
    try {
      const content = await getTransactionFile(requestId, filename)
      setFileContent({ name: filename, content: JSON.stringify(content, null, 2) })
    } catch {
      setFileContent({ name: filename, content: "Error loading file" })
    }
  }

  const totalPages = Math.ceil(total / pageSize)

  const logProviders = useMemo(() => {
    const set = new Set(transactions.map(tx => tx.provider))
    return [...set].sort()
  }, [transactions])

  return (
    <div className="space-y-4">
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search by request ID..."
              className="pl-8"
              value={search}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                setSearch(e.target.value)
                clearTimeout(debounceRef.current)
                debounceRef.current = setTimeout(() => { setDebouncedSearch(e.target.value); setPage(1) }, 400)
              }}
            />
          </div>
          <select
            className="h-8 rounded-md border border-input bg-background px-2 text-xs"
            value={pageSize}
            onChange={e => { setPageSize(Number(e.target.value)); setPage(1) }}
          >
            {[20, 40, 60, 100].map(n => <option key={n} value={n}>{n} / page</option>)}
          </select>
          <select
            className="h-8 rounded-md border border-input bg-background px-2 text-xs"
            value={refreshInterval ?? "off"}
            onChange={e => setRefreshInterval(e.target.value === "off" ? null : Number(e.target.value))}
          >
            {REFRESH_OPTIONS.map(opt => (
              <option key={opt.label} value={opt.value ?? "off"}>{opt.label}</option>
            ))}
          </select>
          <Button variant="outline" size="sm" onClick={fetchData} disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {logProviders.map((p: string) => (
            <Button
              key={p}
              variant={providerFilter === p ? "default" : "outline"}
              size="sm"
              className="h-7 text-xs"
              onClick={() => { setProviderFilter(providerFilter === p ? null : p); setPage(1) }}
            >
              {p}
            </Button>
          ))}
          <span className="mx-1 border-r" />
          <Button
            variant={statusFilter === "success" ? "default" : "outline"}
            size="sm"
            className="h-7 text-xs"
            onClick={() => { setStatusFilter(statusFilter === "success" ? null : "success"); setPage(1) }}
          >
            <CheckCircle className="h-3 w-3 mr-1" />Success
          </Button>
          <Button
            variant={statusFilter === "error" ? "default" : "outline"}
            size="sm"
            className="h-7 text-xs"
            onClick={() => { setStatusFilter(statusFilter === "error" ? null : "error"); setPage(1) }}
          >
            <XCircle className="h-3 w-3 mr-1" />Error
          </Button>
          {(providerFilter || statusFilter) && (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs text-muted-foreground"
              onClick={() => { setProviderFilter(null); setStatusFilter(null); setPage(1) }}
            >
              Clear
            </Button>
          )}
        </div>
      </div>

      <div className="space-y-2">
        {transactions.map((tx) => {
          const shortPrompt = tx.prompt_preview && tx.prompt_preview.length <= 50
          return (
          <Card key={tx.request_id} className="cursor-pointer" onClick={() => toggleExpand(tx.request_id)}>
            <CardContent className="p-3">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 min-w-0">
                  <StatusBadge status={tx.status} />
                  <span className="font-medium text-sm shrink-0">{tx.provider}</span>
                  {tx.credential_masked && (
                    <Badge variant="secondary" className="text-[10px] font-mono shrink-0">{tx.credential_masked}</Badge>
                  )}
                  {shortPrompt && (
                    <span className="text-xs text-muted-foreground truncate hidden sm:inline">&mdash; {tx.prompt_preview}</span>
                  )}
                </div>
                <div className="flex items-center gap-3 shrink-0 text-xs text-muted-foreground">
                  <span className="font-mono text-[11px] truncate max-w-[200px]" title={tx.model}>{tx.model}</span>
                  {tx.tokens_out > 0 && tx.duration_ms > 0 && (
                    <span title="Output tokens per second">{(tx.tokens_out / (tx.duration_ms / 1000)).toFixed(0)} t/s</span>
                  )}
                  <span>{formatDuration(tx.duration_ms)}</span>
                  <span title={`In: ${tx.tokens_in.toLocaleString()}${tx.tokens_cached ? ` + ${tx.tokens_cached.toLocaleString()} cached` : ""} / Out: ${tx.tokens_out.toLocaleString()}`}>
                    {formatNumber(tx.tokens_in)}{tx.tokens_cached ? <span className="text-blue-500">+{formatNumber(tx.tokens_cached)}c</span> : ""}/{formatNumber(tx.tokens_out)}
                  </span>
                  <span className="font-medium text-foreground">{tx.approx_cost != null && tx.approx_cost > 0 ? formatCost(tx.approx_cost) : "—"}</span>
                  <span className="whitespace-nowrap">{timeAgo(tx.timestamp)}</span>
                </div>
              </div>
              {!shortPrompt && tx.prompt_preview && (
                <p className="text-xs text-muted-foreground mt-1 truncate">{tx.prompt_preview}</p>
              )}

              {expanded === tx.request_id && (
                <div className="mt-3 space-y-3 border-t pt-3" onClick={(e) => e.stopPropagation()}>
                  <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6 text-xs">
                    <div>
                      <span className="text-muted-foreground">Request ID</span>
                      <p className="font-mono">{tx.request_id}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Credential</span>
                      <p className="font-mono">{tx.credential_masked || "—"}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Timestamp</span>
                      <p>{new Date(tx.timestamp.endsWith("Z") ? tx.timestamp : tx.timestamp + "Z").toLocaleString()}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Tokens In</span>
                      <p>{tx.tokens_in.toLocaleString()}{tx.tokens_cached ? ` (${tx.tokens_cached.toLocaleString()} cached)` : ""}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Tokens Out</span>
                      <p>{tx.tokens_out.toLocaleString()}{tx.reasoning_tokens ? ` (${tx.reasoning_tokens.toLocaleString()} reasoning)` : ""}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Duration / Speed</span>
                      <p>{formatDuration(tx.duration_ms)}{tx.tokens_out > 0 && tx.duration_ms > 0 ? ` (${(tx.tokens_out / (tx.duration_ms / 1000)).toFixed(1)} t/s)` : ""}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Cost</span>
                      <p>{tx.approx_cost != null && tx.approx_cost > 0 ? formatCost(tx.approx_cost) : "—"}</p>
                    </div>
                  </div>

                  {expandedDetail?.files && expandedDetail.files.length > 0 && (
                    <div>
                      <p className="text-xs font-medium mb-2">Files</p>
                      <div className="flex flex-wrap gap-2">
                        {expandedDetail.files.map((f: string) => (
                          <Button
                            key={f}
                            variant={fileContent?.name === f ? "default" : "outline"}
                            size="sm"
                            onClick={() => viewFile(tx.request_id, f)}
                          >
                            <FileJson className="h-3 w-3 mr-1" />
                            {f}
                          </Button>
                        ))}
                      </div>
                    </div>
                  )}

                  {fileContent && (
                    <div>
                      <p className="text-xs font-medium mb-1">{fileContent.name}</p>
                      <pre className="bg-muted rounded-md p-3 overflow-x-auto text-xs max-h-[400px] overflow-y-auto">
                        <code>{fileContent.content}</code>
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
          )
        })}
        {!transactions.length && (
          <Card>
            <CardContent className="p-8 text-center text-muted-foreground">
              {loading ? "Loading..." : "No transactions found"}
            </CardContent>
          </Card>
        )}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">{total} total transactions</span>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="text-sm">{page} / {totalPages}</span>
            <Button variant="outline" size="sm" disabled={page >= totalPages} onClick={() => setPage(page + 1)}>
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

function FailureBrowser() {
  const [failures, setFailures] = useState<FailureEntry[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState<number | null>(null)
  const [errorTypeFilter, setErrorTypeFilter] = useState<string | null>(null)
  const [providerFilter, setProviderFilter] = useState<string | null>(null)
  const [errorTypes, setErrorTypes] = useState<{ type: string; count: number }[]>([])
  const [failureProviders, setFailureProviders] = useState<{ name: string; count: number }[]>([])
  const [pageSize, setPageSize] = useState(20)
  const [refreshInterval, setRefreshInterval] = useState<number | null>(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getFailures({ page, page_size: pageSize, error_type: errorTypeFilter ?? undefined, provider: providerFilter ?? undefined })
      setFailures(data.failures)
      setTotal(data.total)
      if (data.error_types) setErrorTypes(data.error_types)
      if (data.providers) setFailureProviders(data.providers)
    } catch {
      // handled by empty state
    } finally {
      setLoading(false)
    }
  }, [page, pageSize, errorTypeFilter, providerFilter])

  useEffect(() => { fetchData() }, [fetchData])
  useAutoRefresh(fetchData, refreshInterval)

  const totalPages = Math.ceil(total / pageSize)

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex flex-wrap gap-1.5">
          {failureProviders.map(({ name: pn, count }) => (
            <Button
              key={pn}
              variant={providerFilter === pn ? "default" : "outline"}
              size="sm"
              className="h-7 text-xs"
              onClick={() => { setProviderFilter(providerFilter === pn ? null : pn); setPage(1) }}
            >
              {pn}
              <Badge variant="secondary" className="ml-1 text-[10px] h-4 px-1">{count}</Badge>
            </Button>
          ))}
          {providerFilter && (
            <Button variant="ghost" size="sm" className="h-7 text-xs text-muted-foreground" onClick={() => { setProviderFilter(null); setPage(1) }}>
              Clear
            </Button>
          )}
        </div>
        <div className="flex items-center gap-2">
          <select
            className="h-8 rounded-md border border-input bg-background px-2 text-xs"
            value={pageSize}
            onChange={e => { setPageSize(Number(e.target.value)); setPage(1) }}
          >
            {[20, 40, 60, 100].map(n => <option key={n} value={n}>{n} / page</option>)}
          </select>
          <select
            className="h-8 rounded-md border border-input bg-background px-2 text-xs"
            value={refreshInterval ?? "off"}
            onChange={e => setRefreshInterval(e.target.value === "off" ? null : Number(e.target.value))}
          >
            {REFRESH_OPTIONS.map(opt => (
              <option key={opt.label} value={opt.value ?? "off"}>{opt.label}</option>
            ))}
          </select>
          <Button variant="outline" size="sm" onClick={fetchData} disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </div>
        <div className="flex flex-wrap gap-1.5">
          {errorTypes.map(({ type: et, count }) => (
            <Button
              key={et}
              variant={errorTypeFilter === et ? "default" : "outline"}
              size="sm"
              className="h-7 text-xs"
              onClick={() => { setErrorTypeFilter(errorTypeFilter === et ? null : et); setPage(1) }}
            >
              {et}
              <Badge variant="secondary" className="ml-1 text-[10px] h-4 px-1">{count}</Badge>
            </Button>
          ))}
          {errorTypeFilter && (
            <Button variant="ghost" size="sm" className="h-7 text-xs text-muted-foreground" onClick={() => { setErrorTypeFilter(null); setPage(1) }}>
              Clear
            </Button>
          )}
        </div>
      </div>

      <div className="space-y-2">
        {failures.map((f, i) => (
          <Card key={i} className="cursor-pointer" onClick={() => setExpanded(expanded === i ? null : i)}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-destructive" />
                  {f.provider && <span className="font-medium text-sm">{f.provider}</span>}
                  <span className="font-medium text-sm">{f.error_type}</span>
                  <Badge variant="outline" className="text-[10px]">{f.model}</Badge>
                  {f.api_key_ending && (
                    <Badge variant="secondary" className="text-[10px] font-mono">...{f.api_key_ending}</Badge>
                  )}
                </div>
                <span className="text-xs text-muted-foreground">{timeAgo(f.timestamp)}</span>
              </div>
              <p className="text-xs text-muted-foreground mt-1 truncate">{f.error_message}</p>

              {expanded === i && (
                <div className="mt-3 space-y-2 border-t pt-3">
                  <div className="text-xs space-y-1">
                    <p><span className="font-medium">Timestamp:</span> {new Date(f.timestamp.endsWith("Z") ? f.timestamp : f.timestamp + "Z").toLocaleString()}</p>
                    <p><span className="font-medium">Model:</span> {f.model}</p>
                    {f.attempt_number && <p><span className="font-medium">Attempt:</span> {f.attempt_number}</p>}
                    {f.api_key_ending && <p><span className="font-medium">Key ending:</span> ...{f.api_key_ending}</p>}
                  </div>
                  <div>
                    <p className="text-xs font-medium mb-1">Error Message</p>
                    <pre className="bg-muted rounded p-2 text-xs overflow-x-auto whitespace-pre-wrap">{f.error_message}</pre>
                  </div>
                  {f.error_chain && f.error_chain.length > 0 && (
                    <div>
                      <p className="text-xs font-medium mb-1">Error Chain</p>
                      <div className="space-y-1">
                        {f.error_chain.map((err: string, j: number) => (
                          <pre key={j} className="bg-muted rounded p-2 text-xs overflow-x-auto">{err}</pre>
                        ))}
                      </div>
                    </div>
                  )}
                  {f.raw_response && (
                    <div>
                      <p className="text-xs font-medium mb-1">Raw Response</p>
                      <pre className="bg-muted rounded p-2 text-xs overflow-x-auto max-h-[200px] overflow-y-auto">{f.raw_response}</pre>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        ))}
        {!failures.length && (
          <Card>
            <CardContent className="p-8 text-center text-muted-foreground">
              {loading ? "Loading..." : "No failures recorded"}
            </CardContent>
          </Card>
        )}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">{total} total failures</span>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="text-sm">{page} / {totalPages}</span>
            <Button variant="outline" size="sm" disabled={page >= totalPages} onClick={() => setPage(page + 1)}>
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  if (status === "success" || status === "200") {
    return <Badge variant="success"><CheckCircle className="h-3 w-3 mr-1" />Success</Badge>
  }
  if (status === "error" || parseInt(status) >= 400) {
    return <Badge variant="destructive"><XCircle className="h-3 w-3 mr-1" />Error</Badge>
  }
  return <Badge variant="secondary">{status}</Badge>
}
