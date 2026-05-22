import { getApiKey, getBaseUrl } from "./client"
import type { QuotaStatsResponse } from "./quota"
import type { ErrorRecord } from "./health"

export type WebSocketMessage =
  | { type: "quota_stats"; data: QuotaStatsResponse }
  | { type: "error_event"; data: ErrorRecord[] }
  | { type: "error"; message: string }
  | { type: "auth_result"; ok: boolean }
  | { type: "ping" }

type MessageHandler = (msg: WebSocketMessage) => void

export class ProxyWebSocket {
  private ws: WebSocket | null = null
  private handlers: Set<MessageHandler> = new Set()
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private reconnectDelay = 1000
  private maxReconnectDelay = 30000
  private _connected = false

  get connected() {
    return this._connected
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return

    const baseUrl = getBaseUrl() || window.location.origin
    const wsUrl = baseUrl.replace(/^http/, "ws")

    try {
      this.ws = new WebSocket(`${wsUrl}/v1/ws`)

      this.ws.onopen = () => {
        const apiKey = getApiKey()
        if (apiKey) {
          this.ws?.send(JSON.stringify({ type: "auth", token: apiKey }))
        } else {
          this._connected = true
          this.reconnectDelay = 1000
        }
      }

      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data) as WebSocketMessage
          if (msg.type === "auth_result") {
            if (msg.ok) {
              this._connected = true
              this.reconnectDelay = 1000
            }
            return
          }
          this.handlers.forEach((h) => h(msg))
        } catch {
          // ignore malformed messages
        }
      }

      this.ws.onclose = () => {
        this._connected = false
        this.scheduleReconnect()
      }

      this.ws.onerror = () => {
        this._connected = false
        this.ws?.close()
      }
    } catch {
      this.scheduleReconnect()
    }
  }

  reconnect() {
    this.disconnect()
    this.connect()
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    this.ws?.close()
    this.ws = null
    this._connected = false
  }

  subscribe(handler: MessageHandler): () => void {
    this.handlers.add(handler)
    return () => this.handlers.delete(handler)
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) return
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay)
      this.connect()
    }, this.reconnectDelay)
  }
}

export const proxyWs = new ProxyWebSocket()
