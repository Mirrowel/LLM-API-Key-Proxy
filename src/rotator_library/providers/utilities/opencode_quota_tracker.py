# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
OpenCode Quota Tracking Mixin
"""

import asyncio
import logging
import time
import re
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

# Use the shared rotator_library logger
lib_logger = logging.getLogger("rotator_library")

OPENCODE_BASE_URL = "https://opencode.ai"
# The specific server function ID for workspace usage
OPENCODE_USAGE_FUNC_ID = "c7389bd0e731f80f49593e5ee53835475f4e28594dd6bd83eb229bab753498cd"
# The specific server function ID for billing/balance
OPENCODE_BILLING_FUNC_ID = "c83b78a614689c38ebee981f9b39a8b377716db85c1fd7dbab604adc02d3313d"

class OpencodeQuotaTracker:
    """
    Mixin class providing quota tracking for the OpenCode provider.
    """

    _balance_cache: Dict[str, Dict[str, Any]]
    _quota_refresh_interval: int

    async def fetch_opencode_data(
        self,
        workspace_id: str,
        auth_cookie: str,
        func_id: str,
        server_instance: str,
        client: Optional[httpx.AsyncClient] = None,
        is_usage: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch data from the OpenCode server-side function endpoint.
        """
        try:
            from datetime import datetime
            now = datetime.now()
            
            if is_usage:
                args = {
                    "t": {
                        "t": 9,
                        "i": 0,
                        "l": 3,
                        "a": [
                            {"t": 1, "s": workspace_id},
                            {"t": 0, "s": now.year},
                            {"t": 0, "s": now.month - 1}
                        ],
                        "o": 0
                    },
                    "f": 31,
                    "m": []
                }
            else:
                args = {
                    "t": {
                        "t": 9,
                        "i": 0,
                        "l": 1,
                        "a": [{"t": 1, "s": workspace_id}],
                        "o": 0
                    },
                    "f": 31,
                    "m": []
                }

            url = f"{OPENCODE_BASE_URL}/_server?id={func_id}"
            
            headers = {
                "x-server-id": func_id,
                "x-server-instance": server_instance,
                "referer": f"{OPENCODE_BASE_URL}/workspace/{workspace_id}/usage",
                "cookie": f"auth={auth_cookie}; oc_locale=en",
                "accept": "*/*",
                "content-type": "application/json"
            }

            if client is not None:
                response = await client.post(url, headers=headers, json=args, timeout=20)
            else:
                async with httpx.AsyncClient() as new_client:
                    response = await new_client.post(url, headers=headers, json=args, timeout=20)
            
            response.raise_for_status()
            text = response.text

            # Find the object assigned to $R[0]
            match = re.search(r'\$R\[0\]=(.*?)\)\(\$R\[', text)
                
            if match:
                json_str = match.group(1)
                
                # Pre-processing for JS-to-JSON:
                # 1. Remove recursive references like $R[1]=
                json_str = re.sub(r'\$R\[\d+\]=', '', json_str)
                # 2. Replace !0 with true, !1 with false
                json_str = json_str.replace("!0", "true").replace("!1", "false")
                # 3. Property names without quotes
                json_str = re.sub(r'(\b[a-zA-Z0-9_]+\b):', r'"\1":', json_str)
                # 4. Remove extra trailing commas
                json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
                
                try:
                    data = json.loads(json_str)
                    lib_logger.debug(f"Successfully parsed OpenCode data for {workspace_id}")
                    return data
                except json.JSONDecodeError as e:
                    lib_logger.debug(f"Failed to parse OpenCode JS-to-JSON: {e} | Raw snippet: {json_str[:200]}")
                    return None
            else:
                lib_logger.debug(f"Regex match failed for OpenCode response: {text[:200]}")
                return None

        except Exception as e:
            lib_logger.warning(f"Failed to fetch OpenCode data for {workspace_id}: {e}")
            return None

    async def refresh_balance(
        self,
        auth_cookie: str,
        credential_identifier: str,
        workspace_id: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Refresh usage and balance information for OpenCode.
        """
        if not workspace_id or not auth_cookie:
            # Fallback parsing
            if "::" in credential_identifier:
                parts = credential_identifier.split("::")
                if len(parts) >= 2:
                    workspace_id = parts[0]
                    last = parts[-1]
                    if last.startswith("auth="):
                        auth_cookie = last[5:]
                    elif last.startswith("Fe26.2**"):
                        auth_cookie = last
        
        if not workspace_id:
            return {"status": "error", "error": "missing workspace_id"}

        if not auth_cookie:
            return {"status": "error", "error": "missing auth cookie"}

        # Fetch usage
        usage_data = await self.fetch_opencode_data(
            workspace_id, auth_cookie, OPENCODE_USAGE_FUNC_ID, "server-fn:3", client, is_usage=True
        )
        
        # Fetch billing/balance
        billing_data = await self.fetch_opencode_data(
            workspace_id, auth_cookie, OPENCODE_BILLING_FUNC_ID, "server-fn:4", client, is_usage=False
        )

        if not usage_data:
            return {"status": "error", "error": "failed to fetch usage data"}

        # Extract reset times and percentages
        rolling = usage_data.get("rollingUsage", {})
        weekly = usage_data.get("weeklyUsage", {})
        monthly = usage_data.get("monthlyUsage", {})
        
        now = time.time()
        resets = []
        for u in [rolling, weekly, monthly]:
            if isinstance(u, dict) and "resetInSec" in u:
                resets.append(now + u["resetInSec"])
        
        next_reset_ts = min(resets) if resets else None
        
        usage_percents = []
        for u in [rolling, weekly, monthly]:
            if isinstance(u, dict):
                usage_percents.append(u.get("usagePercent", 0))
        
        max_usage_percent = max(usage_percents) if usage_percents else 0

        balance_data = {
            "status": "success",
            "workspace_id": workspace_id,
            "usage_percent": max_usage_percent,
            "remaining_fraction": max(0.0, (100.0 - max_usage_percent) / 100.0),
            "quota_reset_ts": next_reset_ts,
            "usage_raw": usage_data,
            "billing_raw": billing_data,
            "fetched_at": now,
        }

        self._balance_cache[credential_identifier] = balance_data
        return balance_data

    def get_remaining_fraction(self, balance_data: Dict[str, Any]) -> float:
        """
        Calculate remaining quota fraction.
        """
        return balance_data.get("remaining_fraction", 1.0)
