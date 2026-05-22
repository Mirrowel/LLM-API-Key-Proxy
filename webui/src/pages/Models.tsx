import { useState, useMemo, useCallback, useRef } from "react"
import { Search, RefreshCw, X } from "lucide-react"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui/table"
import { usePolling } from "@/hooks/usePolling"
import { getModels, type ModelList, type ModelCard } from "@/api/models"

export function Models() {
  const { data, loading, refresh } = usePolling<ModelList>({
    fetcher: getModels,
    interval: 60000,
  })
  const [search, setSearch] = useState("")
  const [activeProviders, setActiveProviders] = useState<Set<string>>(new Set())
  const [contextFilter, setContextFilter] = useState<number | null>(null)
  const [costField, setCostField] = useState<"input" | "output">("input")
  const [costOp, setCostOp] = useState<string>("")
  const [costValue, setCostValue] = useState<string>("")
  const costDebounce = useRef<ReturnType<typeof setTimeout>>(undefined)

  const providers = useMemo(() => {
    if (!data?.data) return []
    const counts: Record<string, number> = {}
    for (const m of data.data) {
      counts[m.owned_by] = (counts[m.owned_by] || 0) + 1
    }
    return Object.entries(counts).sort(([a], [b]) => a.localeCompare(b))
  }, [data])

  const toggleProvider = useCallback((provider: string) => {
    setActiveProviders(prev => {
      const next = new Set(prev)
      if (next.has(provider)) {
        next.delete(provider)
      } else {
        next.add(provider)
      }
      return next
    })
  }, [])

  const contextBuckets = useMemo(() => {
    if (!data?.data) return [] as [string, number, number][]
    const buckets: [string, number, number][] = [
      ["8K+", 8000, 0], ["32K+", 32000, 0], ["128K+", 128000, 0],
      ["200K+", 200000, 0], ["1M+", 1000000, 0],
    ]
    for (const m of data.data) {
      for (const b of buckets) {
        if (m.context_length && m.context_length >= b[1]) b[2]++
      }
    }
    return buckets.filter(b => b[2] > 0)
  }, [data])

  const filteredModels = useMemo(() => {
    if (!data?.data) return []
    const costNum = costValue ? parseFloat(costValue) : NaN
    return data.data.filter((m: ModelCard) => {
      if (activeProviders.size > 0 && !activeProviders.has(m.owned_by)) return false
      if (search && !m.id.toLowerCase().includes(search.toLowerCase())) return false
      if (contextFilter && (!m.context_length || m.context_length < contextFilter)) return false
      if (costOp && !isNaN(costNum)) {
        const raw = costField === "output" ? m.output_cost_per_token : m.input_cost_per_token
        if (raw == null) return false
        const perM = raw * 1_000_000
        switch (costOp) {
          case "lt": if (!(perM < costNum)) return false; break
          case "lte": if (!(perM <= costNum)) return false; break
          case "eq": if (!(Math.abs(perM - costNum) < 0.005)) return false; break
          case "gte": if (!(perM >= costNum)) return false; break
          case "gt": if (!(perM > costNum)) return false; break
        }
      }
      return true
    })
  }, [data, search, activeProviders, contextFilter, costField, costOp, costValue])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Models</h1>
          <p className="text-muted-foreground">
            {data?.data.length ?? 0} models available
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={refresh} disabled={loading}>
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      <div className="space-y-3">
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search models..."
            className="pl-8"
            value={search}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearch(e.target.value)}
          />
        </div>
        <div className="flex flex-wrap gap-1.5">
          {providers.map(([p, count]: [string, number]) => (
            <Button
              key={p}
              variant={activeProviders.has(p) ? "default" : "outline"}
              size="sm"
              className="h-7 text-xs"
              onClick={() => toggleProvider(p)}
            >
              {p}
              <Badge variant="secondary" className="ml-1.5 text-[10px] h-4 px-1">
                {count}
              </Badge>
            </Button>
          ))}
          {activeProviders.size > 0 && (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs text-muted-foreground"
              onClick={() => setActiveProviders(new Set())}
            >
              Clear
            </Button>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs text-muted-foreground mr-1">Context:</span>
          {contextBuckets.map(([label, minCtx, count]) => (
            <Button
              key={label}
              variant={contextFilter === minCtx ? "default" : "outline"}
              size="sm"
              className="h-7 text-xs"
              onClick={() => setContextFilter(contextFilter === minCtx ? null : minCtx)}
            >
              {label}
              <Badge variant="secondary" className="ml-1 text-[10px] h-4 px-1">{count}</Badge>
            </Button>
          ))}
          {contextFilter && (
            <Button variant="ghost" size="sm" className="h-7 text-xs text-muted-foreground" onClick={() => setContextFilter(null)}>
              <X className="h-3 w-3" />
            </Button>
          )}
          <span className="mx-2 border-r h-5" />
          <select
            className="h-7 rounded-md border border-input bg-background px-1.5 text-xs"
            value={costField}
            onChange={e => setCostField(e.target.value as "input" | "output")}
          >
            <option value="input">Input $/M</option>
            <option value="output">Output $/M</option>
          </select>
          <select
            className="h-7 rounded-md border border-input bg-background px-1.5 text-xs"
            value={costOp}
            onChange={e => setCostOp(e.target.value)}
          >
            <option value="">—</option>
            <option value="lt">&lt;</option>
            <option value="lte">&le;</option>
            <option value="eq">=</option>
            <option value="gte">&ge;</option>
            <option value="gt">&gt;</option>
          </select>
          <input
            type="number"
            step="0.01"
            placeholder="$"
            className="h-7 w-20 rounded-md border border-input bg-background px-2 text-xs"
            onChange={e => {
              clearTimeout(costDebounce.current)
              const v = e.target.value
              costDebounce.current = setTimeout(() => setCostValue(v), 300)
            }}
          />
          {costOp && (
            <Button variant="ghost" size="sm" className="h-7 text-xs text-muted-foreground" onClick={() => { setCostOp(""); setCostValue("") }}>
              <X className="h-3 w-3" />
            </Button>
          )}
        </div>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">
            {filteredModels.length} model{filteredModels.length !== 1 ? "s" : ""}
            {search || activeProviders.size > 0 ? " (filtered)" : ""}
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model ID</TableHead>
                <TableHead>Provider</TableHead>
                <TableHead className="text-right">Context</TableHead>
                <TableHead className="text-right">Input $/M</TableHead>
                <TableHead className="text-right">Cache Read $/M</TableHead>
                <TableHead className="text-right">Cache Write $/M</TableHead>
                <TableHead className="text-right">Output $/M</TableHead>
                <TableHead>Source</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredModels.map((m: ModelCard, idx: number) => (
                <TableRow key={`${m.id}-${m.owned_by}-${idx}`}>
                  <TableCell className="font-mono text-sm">{m.id}</TableCell>
                  <TableCell>
                    <Badge variant="secondary">{m.owned_by}</Badge>
                  </TableCell>
                  <TableCell className="text-right text-sm text-muted-foreground">
                    {m.context_length ? `${(m.context_length / 1000).toFixed(0)}K` : "-"}
                  </TableCell>
                  <TableCell className="text-right text-sm text-muted-foreground">
                    {m.input_cost_per_token != null ? `$${(m.input_cost_per_token * 1_000_000).toFixed(2)}` : "-"}
                  </TableCell>
                  <TableCell className="text-right text-sm text-muted-foreground">
                    {m.pricing?.cached_input != null ? `$${(m.pricing.cached_input * 1_000_000).toFixed(2)}` : "-"}
                  </TableCell>
                  <TableCell className="text-right text-sm text-muted-foreground">
                    {m.pricing?.cache_write != null ? `$${(m.pricing.cache_write * 1_000_000).toFixed(2)}` : "-"}
                  </TableCell>
                  <TableCell className="text-right text-sm text-muted-foreground">
                    {m.output_cost_per_token != null ? `$${(m.output_cost_per_token * 1_000_000).toFixed(2)}` : "-"}
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {m._sources?.join(", ") ?? m._match_type ?? ""}
                  </TableCell>
                </TableRow>
              ))}
              {!filteredModels.length && (
                <TableRow>
                  <TableCell colSpan={8} className="text-center text-muted-foreground py-8">
                    {loading ? "Loading..." : "No models found"}
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
