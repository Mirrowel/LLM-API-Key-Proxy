import type { LucideIcon } from "lucide-react"
import {
  LayoutDashboard,
  BarChart3,
  ScrollText,
  Settings,
  KeyRound,
  Filter,
} from "lucide-react"

export interface NavItem {
  to: string
  icon: LucideIcon
  label: string
  end?: boolean
}

export const navItems: NavItem[] = [
  { to: "/ui", icon: LayoutDashboard, label: "Dashboard", end: true },
  { to: "/ui/quota", icon: BarChart3, label: "Quota" },
  { to: "/ui/logs", icon: ScrollText, label: "Logs" },
  { to: "/ui/credentials", icon: KeyRound, label: "Credentials" },
  { to: "/ui/models", icon: Filter, label: "Models" },
  { to: "/ui/settings", icon: Settings, label: "Settings" },
]
