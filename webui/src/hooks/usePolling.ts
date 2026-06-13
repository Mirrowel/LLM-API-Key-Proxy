import { useState, useEffect, useCallback, useRef } from "react"

interface UsePollingOptions<T> {
  fetcher: (...args: any[]) => Promise<T>
  interval?: number
  enabled?: boolean
}

export function usePolling<T>({ fetcher, interval = 10000, enabled = true }: UsePollingOptions<T>) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [loading, setLoading] = useState(true)
  const fetcherRef = useRef(fetcher)
  const hasFetchedRef = useRef(false)
  fetcherRef.current = fetcher

  const refresh = useCallback(async (...args: any[]) => {
    if (!hasFetchedRef.current) {
      setLoading(true)
    }
    try {
      const result = await fetcherRef.current(...args)
      setData(result)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)))
    } finally {
      hasFetchedRef.current = true
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!enabled) return
    refresh()
    const id = setInterval(refresh, interval)

    const onVisChange = () => {
      if (!document.hidden) refresh()
    }
    document.addEventListener("visibilitychange", onVisChange)

    return () => {
      clearInterval(id)
      document.removeEventListener("visibilitychange", onVisChange)
    }
  }, [enabled, interval, refresh])

  return { data, error, loading, refresh }
}
