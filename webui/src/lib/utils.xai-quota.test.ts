import { describe, it, expect } from "vitest"
import {
  formatPercentUsedFromRemaining,
  formatResetCalendarDate,
  formatXaiQuotaValueStr,
  isXaiPercentOnlyQuotaGroup,
} from "./utils"

describe("x-ai F1 quota display helpers", () => {
  it("isXaiPercentOnlyQuotaGroup", () => {
    expect(isXaiPercentOnlyQuotaGroup("x-ai", "monthly-limit")).toBe(true)
    expect(isXaiPercentOnlyQuotaGroup("codex", "monthly-limit")).toBe(false)
    expect(isXaiPercentOnlyQuotaGroup("x-ai", "on-demand($)")).toBe(false)
  })

  it("formatPercentUsedFromRemaining", () => {
    expect(formatPercentUsedFromRemaining(82.64)).toBe("17.4% used")
    expect(formatPercentUsedFromRemaining(100)).toBe("0% used")
    expect(formatPercentUsedFromRemaining(0)).toBe("100% used")
  })

  it("formatResetCalendarDate", () => {
    // 2026-07-01 UTC
    const ts = Date.UTC(2026, 6, 1) / 1000
    const label = formatResetCalendarDate(ts)
    expect(label).toMatch(/Jul/)
    expect(label).toMatch(/1/)
  })

  it("formatXaiQuotaValueStr", () => {
    const s = formatXaiQuotaValueStr(82, null)
    expect(s).toBe("18% used")
    const ts = Date.UTC(2026, 5, 30) / 1000
    const withReset = formatXaiQuotaValueStr(82, ts)
    expect(withReset).toContain("18% used")
    expect(withReset).toContain("Resets")
  })
})