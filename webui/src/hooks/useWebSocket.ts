import { useEffect, useCallback, useState, useRef } from "react"
import { proxyWs, type WebSocketMessage } from "@/api/websocket"

export function useWebSocket(onMessage?: (msg: WebSocketMessage) => void) {
  const [connected, setConnected] = useState(proxyWs.connected)
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  useEffect(() => {
    proxyWs.connect()

    const unsubscribe = proxyWs.subscribe((msg: WebSocketMessage) => {
      onMessageRef.current?.(msg)
    })

    const checkInterval = setInterval(() => {
      setConnected(proxyWs.connected)
    }, 2000)

    return () => {
      unsubscribe()
      clearInterval(checkInterval)
    }
  }, [])

  const reconnect = useCallback(() => {
    proxyWs.disconnect()
    proxyWs.connect()
  }, [])

  return { connected, reconnect }
}
