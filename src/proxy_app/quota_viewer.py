# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

"""
Lightweight Quota Stats Viewer TUI.

Connects to a running proxy to display quota and usage statistics.
Uses only httpx + rich (no heavy rotator_library imports).
"""

import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .quota_viewer_config import QuotaViewerConfig


# =============================================================================
# DISPLAY CONFIGURATION - Adjust these values to customize the layout
# =============================================================================

# Summary screen table column widths
TABLE_PROVIDER_WIDTH = 12
TABLE_CREDS_WIDTH = 3
TABLE_QUOTA_STATUS_WIDTH = 62
TABLE_REQUESTS_WIDTH = 5
TABLE_TOKENS_WIDTH = 20
TABLE_COST_WIDTH = 6

# Quota status formatting in summary screen
QUOTA_NAME_WIDTH = 15  # Width for quota group name (e.g., "claude:")
QUOTA_USAGE_WIDTH = 11  # Width for usage ratio (e.g., "2071/2700")
QUOTA_PCT_WIDTH = 6  # Width for percentage (e.g., "76.7%")
QUOTA_BAR_WIDTH = 10  # Width for progress bar

# Detail view credential panel formatting
DETAIL_GROUP_NAME_WIDTH = (
    18  # Width for group name in detail view (handles "g25-flash (daily)")
)
DETAIL_USAGE_WIDTH = (
    16  # Width for usage ratio in detail view (handles "3000/3000(5000)")
)
DETAIL_PCT_WIDTH = 7  # Width for percentage in detail view

# =============================================================================
# STATUS DISPLAY CONFIGURATION
# =============================================================================

# Credential status icons and colors: (icon, label, color)
# Using Rich emoji markup :name: for consistent width handling
STATUS_DISPLAY = {
    "active": (":white_check_mark:", "Active", "green"),
    "cooldown": (":stopwatch:", "Cooldown", "yellow"),
    "exhausted": (":no_entry:", "Exhausted", "red"),
    "mixed": (":warning:", "Mixed", "yellow"),
}

# Per-group indicator icons (using Rich emoji markup for proper width handling)
INDICATOR_ICONS = {
    "fair_cycle": ":scales:",  # ⚖️ - Rich will handle width
    "custom_cap": ":bar_chart:",  # 📊
    "cooldown": ":stopwatch:",  # ⏱️
}

# =============================================================================


def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def format_tokens(count: int) -> str:
    """Format token count for display (e.g., 125000 -> 125k)."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.0f}k"
    return str(count)


def format_cost(cost: Optional[float]) -> str:
    """Format cost for display."""
    if cost is None or cost == 0:
        return "-"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def format_time_ago(timestamp: Optional[float]) -> str:
    """Format timestamp as relative time (e.g., '5 min ago')."""
    if not timestamp:
        return "Never"
    try:
        delta = time.time() - timestamp
        if delta < 60:
            return f"{int(delta)}s ago"
        elif delta < 3600:
            return f"{int(delta / 60)} min ago"
        elif delta < 86400:
            return f"{int(delta / 3600)}h ago"
        else:
            return f"{int(delta / 86400)}d ago"
    except (ValueError, OSError):
        return "Unknown"


def format_reset_time(iso_time: Optional[str]) -> str:
    """Format ISO time string for display."""
    if not iso_time:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        # Convert to local time
        local_dt = dt.astimezone()
        return local_dt.strftime("%b %d %H:%M")
    except (ValueError, AttributeError):
        return iso_time[:16] if iso_time else "-"


def create_progress_bar(percent: Optional[int], width: int = 10) -> str:
    """Create a text-based progress bar."""
    if percent is None:
        return "░" * width
    filled = int(percent / 100 * width)
    return "▓" * filled + "░" * (width - filled)


def is_local_host(host: str) -> bool:
    """Check if host is a local/private address (should use http, not https)."""
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "::"):
        return True
    # Private IP ranges
    if host.startswith("192.168.") or host.startswith("10."):
        return True
    if host.startswith("172."):
        # 172.16.0.0 - 172.31.255.255
        try:
            second_octet = int(host.split(".")[1])
            if 16 <= second_octet <= 31:
                return True
        except (ValueError, IndexError):
            pass
    return False


def normalize_host_for_connection(host: str) -> str:
    """
    Convert bind addresses to connectable addresses.

    0.0.0.0 and :: are valid for binding a server to all interfaces,
    but clients cannot connect to them. Translate to loopback addresses.
    """
    if host == "0.0.0.0":
        return "127.0.0.1"
    if host == "::":
        return "::1"
    return host


def get_scheme_for_host(host: str, port: int) -> str:
    """Determine http or https scheme based on host and port."""
    if port == 443:
        return "https"
    if is_local_host(host):
        return "http"
    # For external domains, default to https
    if "." in host:
        return "https"
    return "http"


def is_full_url(host: str) -> bool:
    """Check if host is already a full URL (starts with http:// or https://)."""
    return host.startswith("http://") or host.startswith("https://")


def format_cooldown(seconds: int) -> str:
    """Format cooldown seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins}m {secs}s" if secs > 0 else f"{mins}m"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours}h {mins}m" if mins > 0 else f"{hours}h"


def natural_sort_key(item: Any) -> List:
    """
    Generate a sort key for natural/numeric sorting.

    Sorts credentials like proj-1, proj-2, proj-10 correctly
    instead of alphabetically (proj-1, proj-10, proj-2).

    Handles both dict items (new API format) and strings.
    """
    if isinstance(item, dict):
        identifier = item.get("identifier", item.get("stable_id", ""))
    else:
        identifier = str(item)
    # Split into text and numeric parts
    parts = re.split(r"(\d+)", identifier)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def get_credentials_list(prov_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert credentials from dict format to list.

    The new API returns credentials as a dict keyed by stable_id.
    This function converts it to a list for iteration and sorting.
    """
    credentials = prov_stats.get("credentials", {})
    if isinstance(credentials, list):
        return credentials
    if isinstance(credentials, dict):
        return list(credentials.values())
    return []


def get_credential_stats(
    cred: Dict[str, Any], view_mode: str = "current"
) -> Dict[str, Any]:
    """
    Extract display stats from a credential with field name adaptation.

    Maps new API field names to what the viewer expects:
    - totals.request_count -> requests
    - totals.last_used_at -> last_used_ts
    - totals.approx_cost -> approx_cost
    - Derive tokens from totals
    """
    totals = cred.get("totals", {})

    # For global view mode, we'd need global totals (currently same as totals)
    if view_mode == "global":
        stats_source = cred.get("global", totals)
        if stats_source == totals:
            stats_source = totals
    else:
        stats_source = totals

    # Calculate proper token stats
    prompt_tokens = stats_source.get("prompt_tokens", 0)
    cache_read = stats_source.get("prompt_tokens_cache_read", 0)
    output_tokens = stats_source.get("output_tokens", 0)

    # Total input = uncached (prompt_tokens) + cached (cache_read)
    input_total = prompt_tokens + cache_read
    input_cached = cache_read
    input_uncached = prompt_tokens

    cache_pct = round(input_cached / input_total * 100, 1) if input_total > 0 else 0

    return {
        "requests": stats_source.get("request_count", 0),
        "last_used_ts": stats_source.get("last_used_at"),
        "approx_cost": stats_source.get("approx_cost"),
        "tokens": {
            "input_cached": input_cached,
            "input_uncached": input_uncached,
            "input_cache_pct": cache_pct,
            "output": output_tokens,
        },
    }


def provider_sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple:
    """
    Sort key for providers.

    Order: has quota_groups -> has activity -> request count -> credential count
    """
    name, stats = item
    has_quota_groups = bool(stats.get("quota_groups"))
    has_activity = stats.get("total_requests", 0) > 0
    return (
        not has_quota_groups,  # False (has groups) sorts first
        not has_activity,  # False (has activity) sorts first
        -stats.get("total_requests", 0),  # Higher requests first
        -stats.get("credential_count", 0),  # Higher creds first
        name.lower(),  # Alphabetically last
    )


def quota_group_sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple:
    """
    Sort key for quota groups.

    Order: by total quota limit (lowest first), then alphabetically.
    Groups without limits sort last.
    """
    name, group_stats = item
    windows = group_stats.get("windows", {})

    if not windows:
        return (float("inf"), name)  # No windows = sort last

    # Find minimum total_max across windows
    min_limit = float("inf")
    for window_stats in windows.values():
        total_max = window_stats.get("total_max", 0)
        if total_max > 0:
            min_limit = min(min_limit, total_max)

    return (min_limit, name)


class QuotaViewer:
    """Main Quota Viewer TUI class."""

    def __init__(self, config: Optional[QuotaViewerConfig] = None):
        """
        Initialize the viewer.

        Args:
            config: Optional config object. If not provided, one will be created.
        """
        # Use emoji_variant="text" for more consistent width calculations
        self.console = Console(emoji_variant="text")
        self.config = config or QuotaViewerConfig()
        self.config.sync_with_launcher_config()

        self.current_remote: Optional[Dict[str, Any]] = None
        self.cached_stats: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None
        self.running = True
        self.view_mode = "current"  # "current" or "global"

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers including auth if configured."""
        headers = {}
        if self.current_remote and self.current_remote.get("api_key"):
            headers["Authorization"] = f"Bearer {self.current_remote['api_key']}"
        return headers

    def _get_base_url(self) -> str:
        """Get base URL for the current remote."""
        if not self.current_remote:
            return "http://127.0.0.1:8000"
        host = self.current_remote.get("host", "127.0.0.1")
        host = normalize_host_for_connection(host)

        # If host is a full URL, use it directly (strip trailing slash)
        if is_full_url(host):
            return host.rstrip("/")

        # Otherwise construct from host:port
        port = self.current_remote.get("port", 8000)
        scheme = get_scheme_for_host(host, port)
        return f"{scheme}://{host}:{port}"

    def _build_endpoint_url(self, endpoint: str) -> str:
        """
        Build a full endpoint URL with smart path handling.

        Handles cases where base URL already contains a path (e.g., /v1):
        - Base: "https://api.example.com/v1", Endpoint: "/v1/quota-stats"
          -> "https://api.example.com/v1/quota-stats" (no duplication)
        - Base: "http://localhost:8000", Endpoint: "/v1/quota-stats"
          -> "http://localhost:8000/v1/quota-stats"

        Args:
            endpoint: The endpoint path (e.g., "/v1/quota-stats")

        Returns:
            Full URL string
        """
        base_url = self._get_base_url()
        endpoint = endpoint.lstrip("/")

        # Check if base URL already ends with a path segment that matches
        # the start of the endpoint (e.g., base ends with /v1, endpoint starts with v1/)
        from urllib.parse import urlparse

        parsed = urlparse(base_url)
        base_path = parsed.path.rstrip("/")

        # If base has a path and endpoint starts with the same segment, avoid duplication
        if base_path:
            # e.g., base_path = "/v1", endpoint = "v1/quota-stats"
            # We want to produce "/v1/quota-stats", not "/v1/v1/quota-stats"
            base_segments = base_path.split("/")
            endpoint_segments = endpoint.split("/")

            # Check if first endpoint segment matches last base segment
            if base_segments and endpoint_segments:
                if base_segments[-1] == endpoint_segments[0]:
                    # Skip the duplicated segment in endpoint
                    endpoint = "/".join(endpoint_segments[1:])

        return f"{base_url}/{endpoint}"

    def check_connection(
        self, remote: Dict[str, Any], timeout: float = 3.0
    ) -> Tuple[bool, str]:
        """
        Check if a remote proxy is reachable.

        Args:
            remote: Remote configuration dict
            timeout: Connection timeout in seconds

        Returns:
            Tuple of (is_online, status_message)
        """
        host = remote.get("host", "127.0.0.1")
        host = normalize_host_for_connection(host)

        # If host is a full URL, extract scheme and netloc to hit root
        if is_full_url(host):
            from urllib.parse import urlparse

            parsed = urlparse(host)
            # Hit the root domain, not the path (e.g., /v1 would 404)
            url = f"{parsed.scheme}://{parsed.netloc}/"
        else:
            port = remote.get("port", 8000)
            scheme = get_scheme_for_host(host, port)
            url = f"{scheme}://{host}:{port}/"

        headers = {}
        if remote.get("api_key"):
            headers["Authorization"] = f"Bearer {remote['api_key']}"

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(url, headers=headers)
                if response.status_code == 200:
                    return True, "Online"
                elif response.status_code == 401:
                    return False, "Auth failed"
                else:
                    return False, f"HTTP {response.status_code}"
        except httpx.ConnectError:
            return False, "Offline"
        except httpx.TimeoutException:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)[:20]

    def fetch_stats(self, provider: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch quota stats from the current remote.

        Args:
            provider: Optional provider filter

        Returns:
            Stats dict or None on failure
        """
        url = self._build_endpoint_url("/v1/quota-stats")
        if provider:
            url += f"?provider={provider}"

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, headers=self._get_headers())

                if response.status_code == 401:
                    self.last_error = "Authentication failed. Check API key."
                    return None
                elif response.status_code != 200:
                    self.last_error = (
                        f"HTTP {response.status_code}: {response.text[:100]}"
                    )
                    return None

                self.cached_stats = response.json()
                self.last_error = None
                return self.cached_stats

        except httpx.ConnectError:
            self.last_error = "Connection failed. Is the proxy running?"
            return None
        except httpx.TimeoutException:
            self.last_error = "Request timed out."
            return None
        except Exception as e:
            self.last_error = str(e)
            return None

    def _merge_provider_stats(self, provider: str, result: Dict[str, Any]) -> None:
        """
        Merge provider-specific stats into the existing cache.

        Updates just the specified provider's data and recalculates the
        summary fields to reflect the change.

        Args:
            provider: Provider name that was refreshed
            result: API response containing the refreshed provider data
        """
        if not self.cached_stats:
            self.cached_stats = result
            return

        # Merge provider data
        if "providers" in result and provider in result["providers"]:
            if "providers" not in self.cached_stats:
                self.cached_stats["providers"] = {}
            self.cached_stats["providers"][provider] = result["providers"][provider]

        # Update timestamp
        if "timestamp" in result:
            self.cached_stats["timestamp"] = result["timestamp"]

        # Recalculate summary from all providers
        self._recalculate_summary()

    def _recalculate_summary(self) -> None:
        """
        Recalculate summary fields from all provider data in cache.

        Updates both 'summary' and 'global_summary' based on current
        provider stats.
        """
        providers = self.cached_stats.get("providers", {})
        if not providers:
            return

        # Calculate summary from all providers
        total_creds = 0
        active_creds = 0
        exhausted_creds = 0
        total_requests = 0
        total_input_cached = 0
        total_input_uncached = 0
        total_output = 0
        total_cost = 0.0

        for prov_stats in providers.values():
            total_creds += prov_stats.get("credential_count", 0)
            active_creds += prov_stats.get("active_count", 0)
            exhausted_creds += prov_stats.get("exhausted_count", 0)
            total_requests += prov_stats.get("total_requests", 0)

            tokens = prov_stats.get("tokens", {})
            total_input_cached += tokens.get("input_cached", 0)
            total_input_uncached += tokens.get("input_uncached", 0)
            total_output += tokens.get("output", 0)

            cost = prov_stats.get("approx_cost")
            if cost:
                total_cost += cost

        total_input = total_input_cached + total_input_uncached
        input_cache_pct = (
            round(total_input_cached / total_input * 100, 1) if total_input > 0 else 0
        )

        self.cached_stats["summary"] = {
            "total_providers": len(providers),
            "total_credentials": total_creds,
            "active_credentials": active_creds,
            "exhausted_credentials": exhausted_creds,
            "total_requests": total_requests,
            "tokens": {
                "input_cached": total_input_cached,
                "input_uncached": total_input_uncached,
                "input_cache_pct": input_cache_pct,
                "output": total_output,
            },
            "approx_total_cost": total_cost if total_cost > 0 else None,
        }

        # Also recalculate global_summary if it exists
        if "global_summary" in self.cached_stats:
            global_total_requests = 0
            global_input_cached = 0
            global_input_uncached = 0
            global_output = 0
            global_cost = 0.0

            for prov_stats in providers.values():
                global_data = prov_stats.get("global", prov_stats)
                global_total_requests += global_data.get("total_requests", 0)

                tokens = global_data.get("tokens", {})
                global_input_cached += tokens.get("input_cached", 0)
                global_input_uncached += tokens.get("input_uncached", 0)
                global_output += tokens.get("output", 0)

                cost = global_data.get("approx_cost")
                if cost:
                    global_cost += cost

            global_total_input = global_input_cached + global_input_uncached
            global_cache_pct = (
                round(global_input_cached / global_total_input * 100, 1)
                if global_total_input > 0
                else 0
            )

            self.cached_stats["global_summary"] = {
                "total_providers": len(providers),
                "total_credentials": total_creds,
                "total_requests": global_total_requests,
                "tokens": {
                    "input_cached": global_input_cached,
                    "input_uncached": global_input_uncached,
                    "input_cache_pct": global_cache_pct,
                    "output": global_output,
                },
                "approx_total_cost": global_cost if global_cost > 0 else None,
            }

    def post_action(
        self,
        action: str,
        scope: str = "all",
        provider: Optional[str] = None,
        credential: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Post a refresh action to the proxy.

        Args:
            action: "reload" or "force_refresh"
            scope: "all", "provider", or "credential"
            provider: Provider name (required for scope != "all")
            credential: Credential identifier (required for scope == "credential")

        Returns:
            Response dict or None on failure
        """
        url = self._build_endpoint_url("/v1/quota-stats")
        payload = {
            "action": action,
            "scope": scope,
        }
        if provider:
            payload["provider"] = provider
        if credential:
            payload["credential"] = credential

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, headers=self._get_headers(), json=payload)

                if response.status_code == 401:
                    self.last_error = "Authentication failed. Check API key."
                    return None
                elif response.status_code != 200:
                    self.last_error = (
                        f"HTTP {response.status_code}: {response.text[:100]}"
                    )
                    return None

                result = response.json()

                # If scope is provider-specific, merge into existing cache
                if scope == "provider" and provider and self.cached_stats:
                    self._merge_provider_stats(provider, result)
                else:
                    # Full refresh - replace everything
                    self.cached_stats = result

                self.last_error = None
                return result

        except httpx.ConnectError:
            self.last_error = "Connection failed. Is the proxy running?"
            return None
        except httpx.TimeoutException:
            self.last_error = "Request timed out."
            return None
        except Exception as e:
            self.last_error = str(e)
            return None

    # =========================================================================
    # DISPLAY SCREENS
    # =========================================================================

    def show_connection_error(self) -> str:
        """
        Display connection error screen with options to configure remotes.

        Returns:
            User choice: 's' (switch), 'm' (manage), 'r' (retry), 'b' (back/exit)
        """
        clear_screen()

        remote_name = (
            self.current_remote.get("name", "Unknown")
            if self.current_remote
            else "None"
        )
        remote_host = self.current_remote.get("host", "") if self.current_remote else ""
        remote_port = self.current_remote.get("port", "") if self.current_remote else ""

        # Format connection display - handle full URLs
        if is_full_url(remote_host):
            connection_display = remote_host
        elif remote_port:
            connection_display = f"{remote_host}:{remote_port}"
        else:
            connection_display = remote_host

        self.console.print(
            Panel(
                Text.from_markup(
                    "[bold red]Connection Error[/bold red]\n\n"
                    f"Remote: [bold]{remote_name}[/bold] ({connection_display})\n"
                    f"Error: {self.last_error or 'Unknown error'}\n\n"
                    "[bold]This tool requires the proxy to be running.[/bold]\n"
                    "Start the proxy first, or configure a different remote.\n\n"
                    "[dim]Tip: Select option 1 from the main menu to run the proxy.[/dim]"
                ),
                border_style="red",
                expand=False,
            )
        )

        self.console.print()
        self.console.print("━" * 78)
        self.console.print()
        self.console.print("   S. Switch to a different remote")
        self.console.print("   M. Manage remotes (add/edit/delete)")
        self.console.print("   R. Retry connection")
        self.console.print("   B. Back to main menu")
        self.console.print()
        self.console.print("━" * 78)

        choice = Prompt.ask("Select option", default="B").strip().lower()

        if choice in ("s", "m", "r", "b"):
            return choice
        return "b"  # Default to back for invalid input

    def show_summary_screen(self):
        """Display the main summary screen with all providers."""
        clear_screen()

        # Header
        remote_name = (
            self.current_remote.get("name", "Unknown")
            if self.current_remote
            else "None"
        )
        remote_host = self.current_remote.get("host", "") if self.current_remote else ""
        remote_port = self.current_remote.get("port", "") if self.current_remote else ""

        # Format connection display - handle full URLs
        if is_full_url(remote_host):
            connection_display = remote_host
        elif remote_port:
            connection_display = f"{remote_host}:{remote_port}"
        else:
            connection_display = remote_host

        # Calculate data age
        data_age = ""
        if self.cached_stats and self.cached_stats.get("timestamp"):
            age_seconds = int(time.time() - self.cached_stats["timestamp"])
            data_age = f"Data age: {age_seconds}s"

        # View mode indicator
        if self.view_mode == "global":
            view_label = "[magenta]:bar_chart: Global/Lifetime[/magenta]"
        else:
            view_label = "[cyan]:chart_with_upwards_trend: Current Period[/cyan]"

        self.console.print("━" * 78)
        self.console.print(
            f"[bold cyan]:chart_with_upwards_trend: Quota & Usage Statistics[/bold cyan]  |  {view_label}"
        )
        self.console.print("━" * 78)
        self.console.print(
            f"Connected to: [bold]{remote_name}[/bold] ({connection_display}) "
            f"[green]:white_check_mark:[/green] | {data_age}"
        )
        self.console.print()

        if not self.cached_stats:
            self.console.print("[yellow]No data available. Press R to reload.[/yellow]")
        else:
            # Build provider table
            table = Table(
                box=None, show_header=True, header_style="bold", padding=(0, 1)
            )
            table.add_column("Provider", style="cyan", min_width=TABLE_PROVIDER_WIDTH)
            table.add_column("Creds", justify="center", min_width=TABLE_CREDS_WIDTH)
            table.add_column("Quota Status", min_width=TABLE_QUOTA_STATUS_WIDTH)
            table.add_column("Req.", justify="right", min_width=TABLE_REQUESTS_WIDTH)
            table.add_column("Tok. in/out(cached)", min_width=TABLE_TOKENS_WIDTH)
            table.add_column("Cost", justify="right", min_width=TABLE_COST_WIDTH)

            providers = self.cached_stats.get("providers", {})
            # Sort providers: quota_groups -> activity -> requests -> creds
            sorted_providers = sorted(providers.items(), key=provider_sort_key)

            for idx, (provider, prov_stats) in enumerate(sorted_providers, 1):
                cred_count = prov_stats.get("credential_count", 0)

                # Use global stats if in global mode
                if self.view_mode == "global":
                    stats_source = prov_stats.get("global", prov_stats)
                    total_requests = stats_source.get("total_requests", 0)
                    tokens = stats_source.get("tokens", {})
                    cost_value = stats_source.get("approx_cost")
                else:
                    total_requests = prov_stats.get("total_requests", 0)
                    tokens = prov_stats.get("tokens", {})
                    cost_value = prov_stats.get("approx_cost")

                # Format tokens
                input_total = tokens.get("input_cached", 0) + tokens.get(
                    "input_uncached", 0
                )
                output = tokens.get("output", 0)
                cache_pct = tokens.get("input_cache_pct", 0)
                token_str = f"{format_tokens(input_total)}/{format_tokens(output)} ({cache_pct}%)"

                # Format cost
                cost_str = format_cost(cost_value)

                # Build quota status string (for providers with quota groups)
                quota_groups = prov_stats.get("quota_groups", {})
                if quota_groups:
                    quota_lines = []
                    # Sort quota groups by minimum remaining % (lowest first)
                    sorted_groups = sorted(
                        quota_groups.items(), key=quota_group_sort_key
                    )

                    for group_name, group_stats in sorted_groups:
                        tiers = group_stats.get("tiers", {})
                        windows = group_stats.get("windows", {})
                        fc_summary = group_stats.get("fair_cycle_summary", {})

                        if not windows:
                            # No windows = no data, skip
                            continue

                        # Process each window for this group
                        for window_name, window_stats in windows.items():
                            total_remaining = window_stats.get("total_remaining", 0)
                            total_max = window_stats.get("total_max", 0)
                            total_pct = window_stats.get("remaining_pct")
                            tier_availability = window_stats.get(
                                "tier_availability", {}
                            )

                            # Format tier info using per-window availability
                            # "15(18)s" = 15 available out of 18 standard-tier
                            tier_parts = []
                            # Sort tiers by priority (from provider-level tiers)
                            sorted_tier_names = sorted(
                                tier_availability.keys(),
                                key=lambda t: tiers.get(t, {}).get("priority", 10),
                            )
                            for tier_name in sorted_tier_names:
                                if tier_name == "unknown":
                                    continue
                                avail_info = tier_availability[tier_name]
                                total_t = avail_info.get("total", 0)
                                available_t = avail_info.get("available", 0)
                                # Use first letter: standard-tier -> s, free-tier -> f
                                short = tier_name.replace("-tier", "")[0]

                                if available_t < total_t:
                                    tier_parts.append(
                                        f"{available_t}({total_t}){short}"
                                    )
                                else:
                                    tier_parts.append(f"{total_t}{short}")

                            tier_str = "/".join(tier_parts) if tier_parts else ""

                            # Only show tier info if this group has limits
                            if total_max == 0:
                                tier_str = ""

                            # Build FC summary string if any credentials are FC exhausted
                            fc_str = ""
                            fc_exhausted = fc_summary.get("exhausted_count", 0)
                            fc_total = fc_summary.get("total_count", 0)
                            if fc_exhausted > 0:
                                fc_str = f"[yellow]{INDICATOR_ICONS['fair_cycle']} {fc_exhausted}/{fc_total}[/yellow]"

                            # Determine color based on remaining percentage and FC status
                            if total_pct is not None:
                                if total_pct <= 10:
                                    color = "red"
                                elif total_pct < 30 or fc_exhausted > 0:
                                    color = "yellow"
                                else:
                                    color = "green"
                            else:
                                color = "dim"

                            pct_str = f"{total_pct}%" if total_pct is not None else "?"

                            # Format: "group (window): remaining/max pct% bar tier_info fc_info"
                            # Show window name if multiple windows exist
                            if len(windows) > 1:
                                display_name = f"{group_name} ({window_name})"
                            else:
                                display_name = group_name

                            display_name_trunc = display_name[: QUOTA_NAME_WIDTH - 1]
                            usage_str = f"{total_remaining}/{total_max}"
                            bar = create_progress_bar(total_pct, QUOTA_BAR_WIDTH)

                            # Build the line with tier info and FC summary
                            line_parts = [
                                f"[{color}]{display_name_trunc + ':':<{QUOTA_NAME_WIDTH}}{usage_str:>{QUOTA_USAGE_WIDTH}} {pct_str:>{QUOTA_PCT_WIDTH}} {bar}[/{color}]"
                            ]
                            if tier_str:
                                line_parts.append(tier_str)
                            if fc_str:
                                line_parts.append(fc_str)

                            quota_lines.append(" ".join(line_parts))

                    # First line goes in the main row
                    first_quota = quota_lines[0] if quota_lines else "-"
                    table.add_row(
                        provider,
                        str(cred_count),
                        first_quota,
                        str(total_requests),
                        token_str,
                        cost_str,
                    )
                    # Additional quota lines as sub-rows
                    for quota_line in quota_lines[1:]:
                        table.add_row("", "", quota_line, "", "", "")
                else:
                    # No quota groups
                    table.add_row(
                        provider,
                        str(cred_count),
                        "-",
                        str(total_requests),
                        token_str,
                        cost_str,
                    )

                # Add separator between providers (except last)
                if idx < len(sorted_providers):
                    table.add_row(
                        "─" * TABLE_PROVIDER_WIDTH,
                        "─" * TABLE_CREDS_WIDTH,
                        "─" * TABLE_QUOTA_STATUS_WIDTH,
                        "─" * TABLE_REQUESTS_WIDTH,
                        "─" * TABLE_TOKENS_WIDTH,
                        "─" * TABLE_COST_WIDTH,
                    )

            self.console.print(table)

            # Summary line - use global_summary if in global mode
            if self.view_mode == "global":
                summary = self.cached_stats.get(
                    "global_summary", self.cached_stats.get("summary", {})
                )
            else:
                summary = self.cached_stats.get("summary", {})

            total_creds = summary.get("total_credentials", 0)
            total_requests = summary.get("total_requests", 0)
            total_tokens = summary.get("tokens", {})
            total_input = total_tokens.get("input_cached", 0) + total_tokens.get(
                "input_uncached", 0
            )
            total_output = total_tokens.get("output", 0)
            total_cost = format_cost(summary.get("approx_total_cost"))

            self.console.print()
            self.console.print(
                f"[bold]Total:[/bold] {total_creds} credentials | "
                f"{total_requests} requests | "
                f"{format_tokens(total_input)}/{format_tokens(total_output)} tokens | "
                f"{total_cost} cost"
            )

        # Menu
        self.console.print()
        self.console.print("━" * 78)
        self.console.print()

        # Build provider menu options (use same sorted order as display)
        providers = self.cached_stats.get("providers", {}) if self.cached_stats else {}
        sorted_providers = sorted(providers.items(), key=provider_sort_key)
        provider_list = [name for name, _ in sorted_providers]

        for idx, provider in enumerate(provider_list, 1):
            self.console.print(f"   {idx}. View [cyan]{provider}[/cyan] details")

        self.console.print()
        self.console.print("   G. Toggle view mode (current/global)")
        self.console.print("   R. Reload all stats (re-read from proxy)")
        self.console.print("   S. Switch remote")
        self.console.print("   M. Manage remotes")
        self.console.print("   B. Back to main menu")
        self.console.print()
        self.console.print("━" * 78)

        # Get input
        valid_choices = [str(i) for i in range(1, len(provider_list) + 1)]
        valid_choices.extend(["r", "R", "s", "S", "m", "M", "b", "B", "g", "G"])

        choice = Prompt.ask("Select option", default="").strip()

        if choice.lower() == "b":
            self.running = False
        elif choice == "":
            # Empty input - just refresh the screen
            pass
        elif choice.lower() == "g":
            # Toggle view mode
            self.view_mode = "global" if self.view_mode == "current" else "current"
        elif choice.lower() == "r":
            with self.console.status("[bold]Reloading stats...", spinner="dots"):
                self.post_action("reload", scope="all")
        elif choice.lower() == "s":
            self.show_switch_remote_screen()
        elif choice.lower() == "m":
            self.show_manage_remotes_screen()
        elif choice.isdigit() and 1 <= int(choice) <= len(provider_list):
            provider = provider_list[int(choice) - 1]
            self.show_provider_detail_screen(provider)

    def show_provider_detail_screen(self, provider: str):
        """Display detailed stats for a specific provider."""
        while True:
            clear_screen()

            # View mode indicator
            if self.view_mode == "global":
                view_label = "[magenta]Global/Lifetime[/magenta]"
            else:
                view_label = "[cyan]Current Period[/cyan]"

            self.console.print("━" * 78)
            self.console.print(
                f"[bold cyan]:bar_chart: {provider.title()} - Detailed Stats[/bold cyan]  |  {view_label}"
            )
            self.console.print("━" * 78)
            self.console.print()

            if not self.cached_stats:
                self.console.print("[yellow]No data available.[/yellow]")
            else:
                prov_stats = self.cached_stats.get("providers", {}).get(provider, {})
                credentials = get_credentials_list(prov_stats)

                # Sort credentials naturally (1, 2, 10 not 1, 10, 2)
                credentials = sorted(credentials, key=natural_sort_key)

                if not credentials:
                    self.console.print(
                        "[dim]No credentials configured for this provider.[/dim]"
                    )
                else:
                    for idx, cred in enumerate(credentials, 1):
                        self._render_credential_panel(idx, cred, provider)
                        self.console.print()

            # Menu
            self.console.print("━" * 78)
            self.console.print()
            self.console.print("   G.  Toggle view mode (current/global)")

            # Force refresh options (only for providers that support it)
            has_quota_groups = bool(
                self.cached_stats
                and self.cached_stats.get("providers", {})
                .get(provider, {})
                .get("quota_groups")
            )

            # Model toggle option (only show if provider has quota groups) - MOVED UP
            if has_quota_groups:
                show_models_status = (
                    "ON" if self.config.get_show_models(provider) else "OFF"
                )
                self.console.print(
                    f"   T.  Toggle model details ({show_models_status})"
                )

            self.console.print("   R.  Reload stats (from proxy cache)")
            self.console.print("   RA. Reload all stats")

            if has_quota_groups:
                self.console.print()
                self.console.print(
                    f"   F.  [yellow]Force refresh ALL {provider} quotas from API[/yellow]"
                )
                prov_stats_for_menu = (
                    self.cached_stats.get("providers", {}).get(provider, {})
                    if self.cached_stats
                    else {}
                )
                credentials = get_credentials_list(prov_stats_for_menu)
                # Sort credentials naturally
                credentials = sorted(credentials, key=natural_sort_key)
                for idx, cred in enumerate(credentials, 1):
                    identifier = cred.get("identifier", f"credential {idx}")
                    email = cred.get("email", identifier)
                    self.console.print(
                        f"   F{idx}. Force refresh [{idx}] only ({email})"
                    )

            # DEBUG: Add fake window for testing multi-window display
            if has_quota_groups:
                self.console.print()
                self.console.print("   [dim]DEBUG:[/dim]")
                self.console.print(
                    "   W.  [dim]Add fake 'daily' window (test multi-window)[/dim]"
                )

            self.console.print()
            self.console.print("   B.  Back to summary")
            self.console.print()
            self.console.print("━" * 78)

            choice = Prompt.ask("Select option", default="B").strip().upper()

            if choice == "B":
                break
            elif choice == "G":
                # Toggle view mode
                self.view_mode = "global" if self.view_mode == "current" else "current"
            elif choice == "T" and has_quota_groups:
                # Toggle show models
                new_state = self.config.toggle_show_models(provider)
                status_str = "enabled" if new_state else "disabled"
                self.console.print(
                    f"[dim]Model details {status_str} for {provider}[/dim]"
                )
            elif choice == "R":
                with self.console.status(
                    f"[bold]Reloading {provider} stats...", spinner="dots"
                ):
                    self.post_action("reload", scope="provider", provider=provider)
            elif choice == "RA":
                with self.console.status(
                    "[bold]Reloading all stats...", spinner="dots"
                ):
                    self.post_action("reload", scope="all")
            elif choice == "F" and has_quota_groups:
                result = None
                with self.console.status(
                    f"[bold]Fetching live quota for ALL {provider} credentials...",
                    spinner="dots",
                ):
                    result = self.post_action(
                        "force_refresh", scope="provider", provider=provider
                    )
                # Handle result OUTSIDE spinner
                if result and result.get("refresh_result"):
                    rr = result["refresh_result"]
                    self.console.print(
                        f"\n[green]Refreshed {rr.get('credentials_refreshed', 0)} credentials "
                        f"in {rr.get('duration_ms', 0)}ms[/green]"
                    )
                    if rr.get("errors"):
                        for err in rr["errors"]:
                            self.console.print(f"[red]  Error: {err}[/red]")
                    Prompt.ask("Press Enter to continue", default="")
            elif choice == "W" and has_quota_groups:
                # DEBUG: Inject fake "daily" window for testing multi-window display
                self._inject_fake_daily_window(provider)
                self.console.print(
                    "[dim]Injected fake 'daily' window into cached stats[/dim]"
                )
                Prompt.ask("Press Enter to continue", default="")
            elif choice.startswith("F") and choice[1:].isdigit() and has_quota_groups:
                idx = int(choice[1:])
                prov_stats_for_refresh = (
                    self.cached_stats.get("providers", {}).get(provider, {})
                    if self.cached_stats
                    else {}
                )
                credentials = get_credentials_list(prov_stats_for_refresh)
                # Sort credentials naturally to match display order
                credentials = sorted(credentials, key=natural_sort_key)
                if 1 <= idx <= len(credentials):
                    cred = credentials[idx - 1]
                    cred_id = cred.get("identifier", "")
                    email = cred.get("email", cred_id)
                    result = None
                    with self.console.status(
                        f"[bold]Fetching live quota for {email}...", spinner="dots"
                    ):
                        result = self.post_action(
                            "force_refresh",
                            scope="credential",
                            provider=provider,
                            credential=cred_id,
                        )
                    # Handle result OUTSIDE spinner
                    if result and result.get("refresh_result"):
                        rr = result["refresh_result"]
                        self.console.print(
                            f"\n[green]Refreshed in {rr.get('duration_ms', 0)}ms[/green]"
                        )
                        if rr.get("errors"):
                            for err in rr["errors"]:
                                self.console.print(f"[red]  Error: {err}[/red]")
                        Prompt.ask("Press Enter to continue", default="")

    def _render_credential_panel(self, idx: int, cred: Dict[str, Any], provider: str):
        """Render a single credential as a panel."""
        identifier = cred.get("identifier", f"credential {idx}")
        email = cred.get("email")
        tier = cred.get("tier", "")
        status = cred.get("status", "unknown")

        # Check for active cooldowns (new format: cooldowns dict)
        cooldowns = cred.get("cooldowns", {})
        has_cooldown = bool(cooldowns)

        # Check for global cooldown
        global_cooldown = cooldowns.get("_global_", {})
        global_cooldown_remaining = (
            global_cooldown.get("remaining_seconds", 0) if global_cooldown else 0
        )

        # Status indicator using centralized config
        status_config = STATUS_DISPLAY.get(status, STATUS_DISPLAY["active"])
        icon, label, color = status_config

        if status == "cooldown" or (has_cooldown and global_cooldown_remaining > 0):
            if global_cooldown_remaining > 0:
                status_icon = f"[{color}]{icon} {label} ({format_cooldown(int(global_cooldown_remaining))})[/{color}]"
            else:
                status_icon = f"[{color}]{icon} {label}[/{color}]"
        else:
            status_icon = f"[{color}]{icon} {label}[/{color}]"

        # Header line
        display_name = email if email else identifier
        tier_str = f" ({tier})" if tier else ""
        header = f"[{idx}] {display_name}{tier_str} {status_icon}"

        # Get stats using helper function
        cred_stats = get_credential_stats(cred, self.view_mode)
        last_used = format_time_ago(cred_stats.get("last_used_ts"))
        requests = cred_stats.get("requests", 0)
        tokens = cred_stats.get("tokens", {})
        input_total = tokens.get("input_cached", 0) + tokens.get("input_uncached", 0)
        output = tokens.get("output", 0)
        cache_pct = tokens.get("input_cache_pct", 0)
        cost = format_cost(cred_stats.get("approx_cost"))

        stats_line = (
            f"Last used: {last_used} | Requests: {requests} | "
            f"Tokens: {format_tokens(input_total)}/{format_tokens(output)} ({cache_pct}%)"
        )
        if cost != "-":
            stats_line += f" | Cost: {cost}"

        # Build panel content
        content_lines = [
            f"[dim]{stats_line}[/dim]",
        ]

        # Group usage (for providers with quota tracking)
        group_usage = cred.get("group_usage", {})

        # Show cooldowns
        if cooldowns:
            active_cooldowns = []
            for key, cooldown_info in cooldowns.items():
                if key == "_global_":
                    continue  # Already shown in status
                remaining = cooldown_info.get("remaining_seconds", 0)
                if remaining > 0:
                    reason = cooldown_info.get("reason", "")
                    source = cooldown_info.get("source", "")
                    active_cooldowns.append((key, remaining, reason, source))

            if active_cooldowns:
                content_lines.append("")
                content_lines.append("[yellow]Active Cooldowns:[/yellow]")
                for key, remaining, reason, source in active_cooldowns:
                    source_str = f" ({source})" if source else ""
                    content_lines.append(
                        f"  [yellow]:stopwatch: {key}: {format_cooldown(int(remaining))}{source_str}[/yellow]"
                    )

        # Display group usage with per-window breakdown
        # Note: group_usage is pre-sorted by limit (lowest first) from the API
        if group_usage:
            content_lines.append("")
            content_lines.append("[bold]Quota Groups:[/bold]")

            for group_name, group_stats in group_usage.items():
                windows = group_stats.get("windows", {})
                if not windows:
                    continue

                # Get per-group status info
                fc_exhausted = group_stats.get("fair_cycle_exhausted", False)
                fc_reason = group_stats.get("fair_cycle_reason")
                group_cooldown_remaining = group_stats.get("cooldown_remaining")
                group_cooldown_source = group_stats.get("cooldown_source")
                custom_cap = group_stats.get("custom_cap")

                for window_name, window_stats in windows.items():
                    request_count = window_stats.get("request_count", 0)
                    limit = window_stats.get("limit")
                    remaining = window_stats.get("remaining")
                    reset_at = window_stats.get("reset_at")
                    max_recorded = window_stats.get("max_recorded_requests")
                    max_recorded_at = window_stats.get("max_recorded_at")

                    # Calculate remaining percentage
                    if limit is not None and limit > 0:
                        remaining_val = (
                            remaining
                            if remaining is not None
                            else max(0, limit - request_count)
                        )
                        remaining_pct = round(remaining_val / limit * 100, 1)
                        is_exhausted = remaining_val <= 0
                    else:
                        remaining_pct = None
                        remaining_val = None
                        is_exhausted = False

                    # Format reset time (only show if there's actual usage or cooldown)
                    reset_time_str = ""
                    if reset_at and (request_count > 0 or group_cooldown_remaining):
                        try:
                            reset_dt = datetime.fromtimestamp(reset_at)
                            reset_time_str = reset_dt.strftime("%b %d %H:%M")
                        except (ValueError, OSError):
                            reset_time_str = ""

                    # Format max recorded info
                    max_info = ""
                    if max_recorded is not None:
                        max_info = f" Max: {max_recorded}"
                        if max_recorded_at:
                            try:
                                max_dt = datetime.fromtimestamp(max_recorded_at)
                                max_info += f" @ {max_dt.strftime('%b %d')}"
                            except (ValueError, OSError):
                                pass

                    # Build display line
                    bar = create_progress_bar(remaining_pct)

                    # Determine color (account for fair cycle)
                    if is_exhausted:
                        color = "red"
                    elif fc_exhausted:
                        color = "yellow"
                    elif remaining_pct is not None and remaining_pct < 20:
                        color = "yellow"
                    else:
                        color = "green"

                    # Format group name
                    if len(windows) > 1:
                        display_name = f"{group_name} ({window_name})"
                    else:
                        display_name = group_name

                    # Format usage string with custom cap if applicable
                    if custom_cap:
                        cap_remaining = custom_cap.get("remaining", 0)
                        cap_limit = custom_cap.get("limit", 0)
                        api_limit = limit if limit else cap_limit
                        usage_str = f"{cap_remaining}/{cap_limit}({api_limit})"
                        # Recalculate percentage based on custom cap
                        if cap_limit > 0:
                            remaining_pct = round(cap_remaining / cap_limit * 100, 1)
                            bar = create_progress_bar(remaining_pct)
                        pct_str = (
                            f"{remaining_pct}%" if remaining_pct is not None else ""
                        )
                    elif limit is not None:
                        usage_str = f"{remaining_val}/{limit}"
                        pct_str = f"{remaining_pct}%"
                    else:
                        usage_str = f"{request_count} req"
                        pct_str = ""

                    line = f"  [{color}]{display_name:<{DETAIL_GROUP_NAME_WIDTH}} {usage_str:<{DETAIL_USAGE_WIDTH}} {pct_str:>{DETAIL_PCT_WIDTH}} {bar}[/{color}]"

                    # Add reset time if applicable
                    if reset_time_str:
                        line += f"  Resets: {reset_time_str}"

                    # Add indicators
                    indicators = []

                    # Group cooldown indicator (if not already showing reset time)
                    if group_cooldown_remaining and not reset_time_str:
                        indicators.append(
                            f"[yellow]{INDICATOR_ICONS['cooldown']} {format_cooldown(int(group_cooldown_remaining))}[/yellow]"
                        )

                    # Fair cycle indicator
                    if fc_exhausted:
                        indicators.append(
                            f"[yellow]{INDICATOR_ICONS['fair_cycle']} FC[/yellow]"
                        )

                    # Custom cap indicator (only if not on cooldown, just to show cap exists)
                    if custom_cap and not group_cooldown_remaining and not fc_exhausted:
                        indicators.append(f"[dim]{INDICATOR_ICONS['custom_cap']}[/dim]")

                    if indicators:
                        line += "  " + " ".join(indicators)

                    # Add max info at the end
                    if max_info:
                        line += f" [dim]{max_info}[/dim]"

                    content_lines.append(line)

        # Model usage (show if no group usage, or if toggle enabled via config)
        model_usage = cred.get("model_usage", {})
        # Check config for show_models setting, default to showing only if no group_usage
        show_models = self.config.get_show_models(provider) if group_usage else True

        if show_models and model_usage:
            content_lines.append("")
            content_lines.append("[dim]Models used:[/dim]")
            for model_name, model_stats in model_usage.items():
                totals = model_stats.get("totals", {})
                req_count = totals.get("request_count", 0)
                if req_count == 0:
                    continue  # Skip models with no usage

                prompt = totals.get("prompt_tokens", 0)
                cache_read = totals.get("prompt_tokens_cache_read", 0)
                output_tokens = totals.get("output_tokens", 0)
                model_cost = format_cost(totals.get("approx_cost"))

                total_input = prompt + cache_read
                short_name = model_name.split("/")[-1][:30]
                content_lines.append(
                    f"    {short_name}: {req_count} req | {format_tokens(total_input)}/{format_tokens(output_tokens)} tokens"
                    + (f" | {model_cost}" if model_cost != "-" else "")
                )

        self.console.print(
            Panel(
                "\n".join(content_lines),
                title=header,
                title_align="left",
                border_style="dim",
                expand=True,
            )
        )

    def _inject_fake_daily_window(self, provider: str) -> None:
        """
        DEBUG: Inject a fake 'daily' window into cached stats for testing.

        This modifies cached_stats in-place to add a second window to each
        quota group, simulating multi-window display without needing real
        multi-window data from the API.
        """
        if not self.cached_stats:
            return

        prov_stats = self.cached_stats.get("providers", {}).get(provider, {})
        if not prov_stats:
            return

        # Inject into quota_groups (global view)
        quota_groups = prov_stats.get("quota_groups", {})
        for group_name, group_stats in quota_groups.items():
            windows = group_stats.get("windows", {})
            if "daily" not in windows and windows:
                # Copy the first window and modify it
                first_window = next(iter(windows.values()))
                daily_window = {
                    "total_used": int(first_window.get("total_used", 0) * 0.3),
                    "total_remaining": int(first_window.get("total_max", 100) * 0.7),
                    "total_max": first_window.get("total_max", 100),
                    "remaining_pct": 70.0,
                    "tier_availability": first_window.get("tier_availability", {}),
                }
                windows["daily"] = daily_window

        # Inject into credential group_usage (detail view)
        credentials = prov_stats.get("credentials", {})
        if isinstance(credentials, dict):
            cred_list = credentials.values()
        else:
            cred_list = credentials

        for cred in cred_list:
            group_usage = cred.get("group_usage", {})
            for group_name, group_stats in group_usage.items():
                windows = group_stats.get("windows", {})
                if "daily" not in windows and windows:
                    # Copy the first window and create a daily version
                    first_window = next(iter(windows.values()))
                    limit = first_window.get("limit", 100)
                    daily_window = {
                        "request_count": int(
                            first_window.get("request_count", 0) * 0.3
                        ),
                        "success_count": int(
                            first_window.get("success_count", 0) * 0.3
                        ),
                        "failure_count": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "thinking_tokens": 0,
                        "output_tokens": 0,
                        "prompt_tokens_cache_read": 0,
                        "prompt_tokens_cache_write": 0,
                        "total_tokens": 0,
                        "limit": limit,
                        "remaining": int(limit * 0.7),
                        "max_recorded_requests": None,
                        "max_recorded_at": None,
                        "reset_at": time.time() + 3600,  # 1 hour from now
                        "approx_cost": 0.0,
                        "first_used_at": None,
                        "last_used_at": None,
                    }
                    windows["daily"] = daily_window

    def show_switch_remote_screen(self):
        """Display remote selection screen."""
        clear_screen()

        self.console.print("━" * 78)
        self.console.print(
            "[bold cyan]:arrows_counterclockwise: Switch Remote[/bold cyan]"
        )
        self.console.print("━" * 78)
        self.console.print()

        current_name = self.current_remote.get("name") if self.current_remote else None
        self.console.print(f"Current: [bold]{current_name}[/bold]")
        self.console.print()
        self.console.print("Available remotes:")

        remotes = self.config.get_remotes()
        remote_status: List[Tuple[Dict, bool, str]] = []

        # Check status of all remotes
        with self.console.status("[dim]Checking remote status...", spinner="dots"):
            for remote in remotes:
                is_online, status_msg = self.check_connection(remote)
                remote_status.append((remote, is_online, status_msg))

        for idx, (remote, is_online, status_msg) in enumerate(remote_status, 1):
            name = remote.get("name", "Unknown")
            host = remote.get("host", "")
            port = remote.get("port", "")

            # Format connection display - handle full URLs
            if is_full_url(host):
                connection_display = host
            elif port:
                connection_display = f"{host}:{port}"
            else:
                connection_display = host

            is_current = name == current_name
            current_marker = " (current)" if is_current else ""

            if is_online:
                status_icon = "[green]:white_check_mark: Online[/green]"
            else:
                status_icon = f"[red]:warning: {status_msg}[/red]"

            self.console.print(
                f"   {idx}. {name:<20} {connection_display:<30} {status_icon}{current_marker}"
            )

        self.console.print()
        self.console.print("━" * 78)
        self.console.print()

        choice = Prompt.ask(
            f"Select remote (1-{len(remotes)}) or B to go back", default="B"
        ).strip()

        if choice.lower() == "b":
            return

        if choice.isdigit() and 1 <= int(choice) <= len(remotes):
            selected = remotes[int(choice) - 1]
            self.current_remote = selected
            self.config.set_last_used(selected["name"])
            self.cached_stats = None  # Clear cache

            # Try to fetch stats from new remote
            with self.console.status("[bold]Connecting...", spinner="dots"):
                stats = self.fetch_stats()
                if stats is None:
                    # Try with API key from .env for Local
                    if selected["name"] == "Local" and not selected.get("api_key"):
                        env_key = self.config.get_api_key_from_env()
                        if env_key:
                            self.current_remote["api_key"] = env_key
                            stats = self.fetch_stats()

            if stats is None:
                self.show_api_key_prompt()

    def show_api_key_prompt(self):
        """Prompt for API key when authentication fails."""
        self.console.print()
        self.console.print(
            "[yellow]Authentication required or connection failed.[/yellow]"
        )
        self.console.print(f"Error: {self.last_error}")
        self.console.print()

        api_key = Prompt.ask(
            "Enter API key (or press Enter to cancel)", default=""
        ).strip()

        if api_key:
            self.current_remote["api_key"] = api_key
            # Update config with new API key
            self.config.update_remote(self.current_remote["name"], api_key=api_key)

            # Try again
            with self.console.status("[bold]Reconnecting...", spinner="dots"):
                if self.fetch_stats() is None:
                    self.console.print(f"[red]Still failed: {self.last_error}[/red]")
                    Prompt.ask("Press Enter to continue", default="")
        else:
            self.console.print("[dim]Cancelled.[/dim]")
            Prompt.ask("Press Enter to continue", default="")

    def show_manage_remotes_screen(self):
        """Display remote management screen."""
        while True:
            clear_screen()

            self.console.print("━" * 78)
            self.console.print("[bold cyan]:gear: Manage Remotes[/bold cyan]")
            self.console.print("━" * 78)
            self.console.print()

            remotes = self.config.get_remotes()

            table = Table(box=None, show_header=True, header_style="bold")
            table.add_column("#", style="dim", width=3)
            table.add_column("Name", min_width=16)
            table.add_column("Host", min_width=24)
            table.add_column("Port", justify="right", width=6)
            table.add_column("Default", width=8)

            for idx, remote in enumerate(remotes, 1):
                is_default = "★" if remote.get("is_default") else ""
                table.add_row(
                    str(idx),
                    remote.get("name", ""),
                    remote.get("host", ""),
                    str(remote.get("port", 8000)),
                    is_default,
                )

            self.console.print(table)

            self.console.print()
            self.console.print("━" * 78)
            self.console.print()
            self.console.print("   A. Add new remote")
            self.console.print("   E. Edit remote (enter number, e.g., E1)")
            self.console.print("   D. Delete remote (enter number, e.g., D1)")
            self.console.print("   S. Set default remote")
            self.console.print("   B. Back")
            self.console.print()
            self.console.print("━" * 78)

            choice = Prompt.ask("Select option", default="B").strip().upper()

            if choice == "B":
                break
            elif choice == "A":
                self._add_remote_dialog()
            elif choice == "S":
                self._set_default_dialog(remotes)
            elif choice.startswith("E") and choice[1:].isdigit():
                idx = int(choice[1:])
                if 1 <= idx <= len(remotes):
                    self._edit_remote_dialog(remotes[idx - 1])
            elif choice.startswith("D") and choice[1:].isdigit():
                idx = int(choice[1:])
                if 1 <= idx <= len(remotes):
                    self._delete_remote_dialog(remotes[idx - 1])

    def _add_remote_dialog(self):
        """Dialog to add a new remote."""
        self.console.print()
        self.console.print("[bold]Add New Remote[/bold]")
        self.console.print(
            "[dim]For full URLs (e.g., https://api.example.com/v1), leave port empty[/dim]"
        )
        self.console.print()

        name = Prompt.ask("Name", default="").strip()
        if not name:
            self.console.print("[dim]Cancelled.[/dim]")
            return

        host = Prompt.ask("Host (or full URL)", default="").strip()
        if not host:
            self.console.print("[dim]Cancelled.[/dim]")
            return

        # For full URLs, default to empty port
        if is_full_url(host):
            port_default = ""
        else:
            port_default = "8000"

        port_str = Prompt.ask(
            "Port (empty for full URLs)", default=port_default
        ).strip()
        if port_str == "":
            port = ""
        else:
            try:
                port = int(port_str)
            except ValueError:
                port = 8000

        api_key = Prompt.ask("API Key (optional)", default="").strip() or None

        if self.config.add_remote(name, host, port, api_key):
            self.console.print(f"[green]Added remote '{name}'.[/green]")
        else:
            self.console.print(f"[red]Remote '{name}' already exists.[/red]")

        Prompt.ask("Press Enter to continue", default="")

    def _edit_remote_dialog(self, remote: Dict[str, Any]):
        """Dialog to edit an existing remote."""
        self.console.print()
        self.console.print(f"[bold]Edit Remote: {remote['name']}[/bold]")
        self.console.print(
            "[dim]Press Enter to keep current value. For full URLs, leave port empty.[/dim]"
        )
        self.console.print()

        new_name = Prompt.ask("Name", default=remote["name"]).strip()
        new_host = Prompt.ask(
            "Host (or full URL)", default=remote.get("host", "")
        ).strip()

        # Get current port, handle empty string
        current_port = remote.get("port", "")
        port_default = str(current_port) if current_port != "" else ""

        new_port_str = Prompt.ask(
            "Port (empty for full URLs)", default=port_default
        ).strip()
        if new_port_str == "":
            new_port = ""
        else:
            try:
                new_port = int(new_port_str)
            except ValueError:
                new_port = current_port if current_port != "" else 8000

        current_key = remote.get("api_key", "") or ""
        display_key = f"{current_key[:8]}..." if len(current_key) > 8 else current_key
        new_key = Prompt.ask(
            f"API Key (current: {display_key or 'none'})", default=""
        ).strip()

        updates = {}
        if new_name != remote["name"]:
            updates["new_name"] = new_name
        if new_host != remote.get("host"):
            updates["host"] = new_host
        if new_port != remote.get("port"):
            updates["port"] = new_port
        if new_key:
            updates["api_key"] = new_key

        if updates:
            if self.config.update_remote(remote["name"], **updates):
                self.console.print("[green]Remote updated.[/green]")
                # Update current_remote if it was the one being edited
                if (
                    self.current_remote
                    and self.current_remote["name"] == remote["name"]
                ):
                    self.current_remote.update(updates)
                    if "new_name" in updates:
                        self.current_remote["name"] = updates["new_name"]
            else:
                self.console.print("[red]Failed to update remote.[/red]")
        else:
            self.console.print("[dim]No changes made.[/dim]")

        Prompt.ask("Press Enter to continue", default="")

    def _delete_remote_dialog(self, remote: Dict[str, Any]):
        """Dialog to delete a remote."""
        self.console.print()
        self.console.print(f"[yellow]Delete remote '{remote['name']}'?[/yellow]")

        confirm = Prompt.ask("Type 'yes' to confirm", default="no").strip().lower()

        if confirm == "yes":
            if self.config.delete_remote(remote["name"]):
                self.console.print(f"[green]Deleted remote '{remote['name']}'.[/green]")
                # If deleted current remote, switch to another
                if (
                    self.current_remote
                    and self.current_remote["name"] == remote["name"]
                ):
                    self.current_remote = self.config.get_default_remote()
                    self.cached_stats = None
            else:
                self.console.print(
                    "[red]Cannot delete. At least one remote must exist.[/red]"
                )
        else:
            self.console.print("[dim]Cancelled.[/dim]")

        Prompt.ask("Press Enter to continue", default="")

    def _set_default_dialog(self, remotes: List[Dict[str, Any]]):
        """Dialog to set the default remote."""
        self.console.print()
        choice = Prompt.ask(f"Set default (1-{len(remotes)})", default="").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(remotes):
            remote = remotes[int(choice) - 1]
            if self.config.set_default_remote(remote["name"]):
                self.console.print(
                    f"[green]'{remote['name']}' is now the default.[/green]"
                )
            else:
                self.console.print("[red]Failed to set default.[/red]")
            Prompt.ask("Press Enter to continue", default="")

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def run(self):
        """Main viewer loop."""
        # Get initial remote
        self.current_remote = self.config.get_last_used_remote()

        if not self.current_remote:
            self.console.print("[red]No remotes configured.[/red]")
            return

        # Connection loop - allows retry after configuring remotes
        while True:
            # For Local remote, try to get API key from .env if not set
            if self.current_remote["name"] == "Local" and not self.current_remote.get(
                "api_key"
            ):
                env_key = self.config.get_api_key_from_env()
                if env_key:
                    self.current_remote["api_key"] = env_key

            # Try to connect
            with self.console.status("[bold]Connecting to proxy...", spinner="dots"):
                stats = self.fetch_stats()

            if stats is not None:
                break  # Connected successfully

            # Connection failed - show error with options
            choice = self.show_connection_error()

            if choice == "b":
                return  # Exit to main menu
            elif choice == "s":
                self.show_switch_remote_screen()
            elif choice == "m":
                self.show_manage_remotes_screen()
            elif choice == "r":
                continue  # Retry connection

            # After switch/manage, refresh current_remote from config
            # (it may have been changed)
            if self.current_remote:
                updated = self.config.get_remote_by_name(self.current_remote["name"])
                if updated:
                    self.current_remote = updated

        # Main loop
        while self.running:
            self.show_summary_screen()


def run_quota_viewer():
    """Entry point for the quota viewer."""
    viewer = QuotaViewer()
    viewer.run()


if __name__ == "__main__":
    run_quota_viewer()
