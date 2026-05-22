import { Outlet } from "react-router-dom"
import { Sidebar } from "./Sidebar"
import { Header } from "./Header"
import { useWebSocket } from "@/hooks/useWebSocket"

interface ShellProps {
  onLogout: () => void
}

export function Shell({ onLogout }: ShellProps) {
  const { connected } = useWebSocket()

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex flex-col flex-1 overflow-hidden">
        <Header wsConnected={connected} onLogout={onLogout} />
        <main className="flex-1 overflow-y-auto p-4 md:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
