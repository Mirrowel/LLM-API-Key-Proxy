import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatNumber(n: number | null | undefined): string {
  if (n == null) return "0"
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toLocaleString()
}

export function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400)
  const h = Math.floor((seconds % 86400) / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const parts: string[] = []
  if (d > 0) parts.push(`${d}d`)
  if (h > 0) parts.push(`${h}h`)
  parts.push(`${m}m`)
  return parts.join(" ")
}

export function formatCost(cost: number | null | undefined): string {
  if (cost == null) return "$0.00"
  if (cost < 0.01) return `$${cost.toFixed(4)}`
  return `$${cost.toFixed(2)}`
}

export function formatDuration(ms: number | null | undefined): string {
  if (ms == null) return "0ms"
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

export function timeAgo(timestamp: string | number): string {
  let ts = timestamp
  if (typeof ts === "string" && !ts.endsWith("Z") && !ts.includes("+")) {
    ts = ts + "Z"
  }
  const seconds = Math.floor((Date.now() - new Date(ts).getTime()) / 1000)
  if (seconds < 0) return "just now"
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
}

export function getStatusColor(status: string): string {
  switch (status) {
    case "active": return "text-success"
    case "cooldown": return "text-warning"
    case "exhausted": return "text-destructive"
    default: return "text-muted-foreground"
  }
}

export function getQuotaColor(pct: number): string {
  if (pct <= 10) return "bg-destructive"
  if (pct <= 30) return "bg-warning"
  return "bg-success"
}

export function formatWindowLabel(label: string): string {
  const hourMatch = label.match(/^(\d+)h$/)
  if (hourMatch) {
    const hours = parseInt(hourMatch[1])
    if (hours >= 24) {
      const days = Math.floor(hours / 24)
      const rem = hours % 24
      return rem > 0 ? `${days}d ${rem}h` : `${days}d`
    }
  }
  return label
}

export function isDollarGroup(groupName: string): boolean {
  return groupName.includes("($)")
}

export function formatCents(cents: number | null | undefined): string {
  if (cents == null) return "$0"
  if (cents % 100 === 0) return `$${cents / 100}`
  return `$${(cents / 100).toFixed(2)}`
}

export function formatQuotaValue(value: number | null | undefined, groupName: string): string {
  if (value == null) return "0"
  if (isDollarGroup(groupName)) return formatCents(value)
  return formatNumber(value)
}

export function formatTimeRemaining(resetAt: number): string {
  const diff = resetAt - Date.now() / 1000
  if (diff <= 0) return "now"
  const days = Math.floor(diff / 86400)
  const hours = Math.floor((diff % 86400) / 3600)
  const minutes = Math.floor((diff % 3600) / 60)
  if (days > 0) return hours > 0 ? `${days}d ${hours}h` : `${days}d`
  if (hours > 0) return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`
  if (minutes > 0) return `${minutes}m`
  return "< 1m"
}

/** x-ai Grok CLI billing: show % used only (no opaque used/limit integers). */
export function isXaiPercentOnlyQuotaGroup(providerName: string, groupName: string): boolean {
  return providerName === "x-ai" && groupName === "monthly-limit"
}

export function formatPercentUsedFromRemaining(remainingPct: number | null | undefined): string {
  const rem = remainingPct ?? 0
  const used = Math.min(100, Math.max(0, 100 - rem))
  const rounded = Math.round(used * 10) / 10
  return Number.isInteger(rounded) ? `${rounded}% used` : `${rounded.toFixed(1)}% used`
}

/** Calendar reset label (e.g. "Jun 30") from unix reset timestamp. */
export function formatResetCalendarDate(resetAt: number | null | undefined): string | null {
  if (resetAt == null || resetAt <= 0) return null
  const d = new Date(resetAt * 1000)
  if (Number.isNaN(d.getTime())) return null
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" })
}

export function formatXaiQuotaValueStr(
  remainingPct: number | null | undefined,
  resetAt?: number | null,
): string {
  const pct = formatPercentUsedFromRemaining(remainingPct)
  const cal = formatResetCalendarDate(resetAt)
  return cal ? `${pct} · Resets ${cal}` : pct
}
