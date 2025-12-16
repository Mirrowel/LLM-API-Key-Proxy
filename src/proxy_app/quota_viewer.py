"""
Lightweight Quota Stats Viewer TUI.

Connects to a running proxy to display quota and usage statistics.
Uses only httpx + rich (no heavy rotator_library imports).
"""

import os
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
    if host in ("localhost", "127.0.0.1", "::1"):
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


class QuotaViewer:
    """Main Quota Viewer TUI class."""

    def __init__(self, config: Optional[QuotaViewerConfig] = None):
        """
        Initialize the viewer.

        Args:
            config: Optional config object. If not provided, one will be created.
        """
        self.console = Console()
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
        port = self.current_remote.get("port", 8000)
        scheme = get_scheme_for_host(host, port)
        return f"{scheme}://{host}:{port}"

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
        url = f"{self._get_base_url()}/v1/quota-stats"
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
        url = f"{self._get_base_url()}/v1/quota-stats"
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

    def show_connection_error(self):
        """Display connection error screen."""
        clear_screen()
        self.console.print(
            Panel(
                Text.from_markup(
                    "[bold red]Connection Error[/bold red]\n\n"
                    f"{self.last_error or 'Unknown error'}\n\n"
                    "[bold]This tool requires the proxy to be running.[/bold]\n"
                    "Start the proxy first, then try again.\n\n"
                    "[dim]Tip: Select option 1 from the main menu to run the proxy.[/dim]"
                ),
                border_style="red",
                expand=False,
            )
        )
        Prompt.ask("\nPress Enter to return to main menu", default="")

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

        # Calculate data age
        data_age = ""
        if self.cached_stats and self.cached_stats.get("timestamp"):
            age_seconds = int(time.time() - self.cached_stats["timestamp"])
            data_age = f"Data age: {age_seconds}s"

        # View mode indicator
        if self.view_mode == "global":
            view_label = "[magenta]📊 Global/Lifetime[/magenta]"
        else:
            view_label = "[cyan]📈 Current Period[/cyan]"

        self.console.print("━" * 78)
        self.console.print(
            f"[bold cyan]📈 Quota & Usage Statistics[/bold cyan]  |  {view_label}"
        )
        self.console.print("━" * 78)
        self.console.print(
            f"Connected to: [bold]{remote_name}[/bold] ({remote_host}:{remote_port}) "
            f"[green]✅[/green] | {data_age}"
        )
        self.console.print()

        if not self.cached_stats:
            self.console.print("[yellow]No data available. Press R to reload.[/yellow]")
        else:
            # Build provider table
            table = Table(box=None, show_header=True, header_style="bold")
            table.add_column("Provider", style="cyan", min_width=12)
            table.add_column("Creds", justify="center", min_width=6)
            table.add_column("Quota Status", min_width=28)
            table.add_column("Requests", justify="right", min_width=9)
            table.add_column("Tokens (in/out)", min_width=22)
            table.add_column("Cost", justify="right", min_width=8)

            providers = self.cached_stats.get("providers", {})
            provider_list = list(providers.keys())

            for idx, (provider, prov_stats) in enumerate(providers.items(), 1):
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
                token_str = f"{format_tokens(input_total)}/{format_tokens(output)} ({cache_pct}% cached)"

                # Format cost
                cost_str = format_cost(cost_value)

                # Build quota status string (for providers with quota groups)
                quota_groups = prov_stats.get("quota_groups", {})
                if quota_groups:
                    quota_lines = []
                    for group_name, group_stats in quota_groups.items():
                        avg_pct = group_stats.get("avg_remaining_pct", 0)
                        exhausted = group_stats.get("credentials_exhausted", 0)
                        total = group_stats.get("credentials_total", 0)

                        # Determine color based on remaining
                        if exhausted > 0:
                            color = "red"
                            status = f"({exhausted}/{total} exhausted)"
                        elif avg_pct < 20:
                            color = "yellow"
                            status = ""
                        else:
                            color = "green"
                            status = ""

                        bar = create_progress_bar(avg_pct)
                        display_name = group_name[:10]
                        quota_lines.append(
                            f"[{color}]{display_name}: {avg_pct}% {bar}[/{color}] {status}"
                        )

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
                if idx < len(providers):
                    table.add_row(
                        "─" * 10, "─" * 4, "─" * 26, "─" * 7, "─" * 20, "─" * 6
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

        # Build provider menu options
        providers = self.cached_stats.get("providers", {}) if self.cached_stats else {}
        provider_list = list(providers.keys())

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

        choice = Prompt.ask("Select option", default="B").strip()

        if choice.lower() == "b":
            self.running = False
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
                f"[bold cyan]📊 {provider.title()} - Detailed Stats[/bold cyan]  |  {view_label}"
            )
            self.console.print("━" * 78)
            self.console.print()

            if not self.cached_stats:
                self.console.print("[yellow]No data available.[/yellow]")
            else:
                prov_stats = self.cached_stats.get("providers", {}).get(provider, {})
                credentials = prov_stats.get("credentials", [])

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
            self.console.print("   R.  Reload stats (from proxy cache)")
            self.console.print("   RA. Reload all stats")

            # Force refresh options (only for providers that support it)
            has_quota_groups = bool(
                self.cached_stats
                and self.cached_stats.get("providers", {})
                .get(provider, {})
                .get("quota_groups")
            )

            if has_quota_groups:
                self.console.print()
                self.console.print(
                    f"   F.  [yellow]Force refresh ALL {provider} quotas from API[/yellow]"
                )
                credentials = (
                    self.cached_stats.get("providers", {})
                    .get(provider, {})
                    .get("credentials", [])
                    if self.cached_stats
                    else []
                )
                for idx, cred in enumerate(credentials, 1):
                    identifier = cred.get("identifier", f"credential {idx}")
                    email = cred.get("email", identifier)
                    self.console.print(
                        f"   F{idx}. Force refresh [{idx}] only ({email})"
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
                with self.console.status(
                    f"[bold]Fetching live quota for ALL {provider} credentials...",
                    spinner="dots",
                ):
                    result = self.post_action(
                        "force_refresh", scope="provider", provider=provider
                    )
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
            elif choice.startswith("F") and choice[1:].isdigit() and has_quota_groups:
                idx = int(choice[1:])
                credentials = (
                    self.cached_stats.get("providers", {})
                    .get(provider, {})
                    .get("credentials", [])
                    if self.cached_stats
                    else []
                )
                if 1 <= idx <= len(credentials):
                    cred = credentials[idx - 1]
                    cred_id = cred.get("identifier", "")
                    email = cred.get("email", cred_id)
                    with self.console.status(
                        f"[bold]Fetching live quota for {email}...", spinner="dots"
                    ):
                        result = self.post_action(
                            "force_refresh",
                            scope="credential",
                            provider=provider,
                            credential=cred_id,
                        )
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

        # Check for active cooldowns
        key_cooldown = cred.get("key_cooldown_remaining")
        model_cooldowns = cred.get("model_cooldowns", {})
        has_cooldown = key_cooldown or model_cooldowns

        # Status indicator
        if status == "exhausted":
            status_icon = "[red]⛔ Exhausted[/red]"
        elif status == "cooldown" or has_cooldown:
            if key_cooldown:
                status_icon = f"[yellow]⚠️ Cooldown ({format_cooldown(int(key_cooldown))})[/yellow]"
            else:
                status_icon = "[yellow]⚠️ Cooldown[/yellow]"
        else:
            status_icon = "[green]✅ Active[/green]"

        # Header line
        display_name = email if email else identifier
        tier_str = f" ({tier})" if tier else ""
        header = f"[{idx}] {display_name}{tier_str} {status_icon}"

        # Use global stats if in global mode
        if self.view_mode == "global":
            stats_source = cred.get("global", cred)
        else:
            stats_source = cred

        # Stats line
        last_used = format_time_ago(cred.get("last_used_ts"))  # Always from current
        requests = stats_source.get("requests", 0)
        tokens = stats_source.get("tokens", {})
        input_total = tokens.get("input_cached", 0) + tokens.get("input_uncached", 0)
        output = tokens.get("output", 0)
        cost = format_cost(stats_source.get("approx_cost"))

        stats_line = (
            f"Last used: {last_used} | Requests: {requests} | "
            f"Tokens: {format_tokens(input_total)}/{format_tokens(output)}"
        )
        if cost != "-":
            stats_line += f" | Cost: {cost}"

        # Build panel content
        content_lines = [
            f"[dim]{stats_line}[/dim]",
        ]

        # Show model cooldowns if any
        if model_cooldowns:
            content_lines.append("")
            content_lines.append("[yellow]Active Cooldowns:[/yellow]")
            for model_name, cooldown_info in model_cooldowns.items():
                remaining = cooldown_info.get("remaining_seconds", 0)
                if remaining > 0:
                    # Shorten model name for display
                    short_model = model_name.split("/")[-1][:35]
                    content_lines.append(
                        f"  [yellow]⏱️ {short_model}: {format_cooldown(int(remaining))}[/yellow]"
                    )

        # Model groups (for providers with quota tracking)
        model_groups = cred.get("model_groups", {})
        if model_groups:
            content_lines.append("")
            for group_name, group_stats in model_groups.items():
                remaining_pct = group_stats.get("remaining_pct")
                requests_used = group_stats.get("requests_used", 0)
                requests_max = group_stats.get("requests_max")
                is_exhausted = group_stats.get("is_exhausted", False)
                reset_time = format_reset_time(group_stats.get("reset_time_iso"))
                confidence = group_stats.get("confidence", "low")

                # Format display
                display = group_stats.get("display", f"{requests_used}/?")
                bar = create_progress_bar(remaining_pct)

                # Color based on status
                if is_exhausted:
                    color = "red"
                    status_text = "⛔ EXHAUSTED"
                elif remaining_pct is not None and remaining_pct < 20:
                    color = "yellow"
                    status_text = "⚠️ LOW"
                else:
                    color = "green"
                    status_text = f"Resets: {reset_time}"

                # Confidence indicator
                conf_indicator = ""
                if confidence == "low":
                    conf_indicator = " [dim](~)[/dim]"
                elif confidence == "medium":
                    conf_indicator = " [dim](?)[/dim]"

                pct_str = f"{remaining_pct}%" if remaining_pct is not None else "?%"
                content_lines.append(
                    f"  [{color}]{group_name:<18} {display:<10} {pct_str:>4} {bar}[/{color}]  {status_text}{conf_indicator}"
                )
        else:
            # For providers without quota groups, show model breakdown if available
            models = cred.get("models", {})
            if models:
                content_lines.append("")
                content_lines.append("  [dim]Models used:[/dim]")
                for model_name, model_stats in models.items():
                    req_count = model_stats.get("success_count", 0)
                    model_cost = format_cost(model_stats.get("approx_cost"))
                    # Shorten model name for display
                    short_name = model_name.split("/")[-1][:30]
                    content_lines.append(
                        f"    {short_name}: {req_count} requests, {model_cost}"
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

    def show_switch_remote_screen(self):
        """Display remote selection screen."""
        clear_screen()

        self.console.print("━" * 78)
        self.console.print("[bold cyan]🔄 Switch Remote[/bold cyan]")
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
            port = remote.get("port", 8000)

            is_current = name == current_name
            current_marker = " (current)" if is_current else ""

            if is_online:
                status_icon = "[green]✅ Online[/green]"
            else:
                status_icon = f"[red]⚠️ {status_msg}[/red]"

            self.console.print(
                f"   {idx}. {name:<20} {host}:{port:<6} {status_icon}{current_marker}"
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
            self.console.print("[bold cyan]⚙️ Manage Remotes[/bold cyan]")
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
        self.console.print()

        name = Prompt.ask("Name", default="").strip()
        if not name:
            self.console.print("[dim]Cancelled.[/dim]")
            return

        host = Prompt.ask("Host", default="").strip()
        if not host:
            self.console.print("[dim]Cancelled.[/dim]")
            return

        port_str = Prompt.ask("Port", default="8000").strip()
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
        self.console.print("[dim]Press Enter to keep current value[/dim]")
        self.console.print()

        new_name = Prompt.ask("Name", default=remote["name"]).strip()
        new_host = Prompt.ask("Host", default=remote.get("host", "")).strip()
        new_port_str = Prompt.ask("Port", default=str(remote.get("port", 8000))).strip()
        try:
            new_port = int(new_port_str)
        except ValueError:
            new_port = remote.get("port", 8000)

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

        # For Local remote, try to get API key from .env if not set
        if self.current_remote["name"] == "Local" and not self.current_remote.get(
            "api_key"
        ):
            env_key = self.config.get_api_key_from_env()
            if env_key:
                self.current_remote["api_key"] = env_key

        # Initial fetch
        with self.console.status("[bold]Connecting to proxy...", spinner="dots"):
            stats = self.fetch_stats()

        if stats is None:
            self.show_connection_error()
            return

        # Main loop
        while self.running:
            self.show_summary_screen()


def run_quota_viewer():
    """Entry point for the quota viewer."""
    viewer = QuotaViewer()
    viewer.run()


if __name__ == "__main__":
    run_quota_viewer()
