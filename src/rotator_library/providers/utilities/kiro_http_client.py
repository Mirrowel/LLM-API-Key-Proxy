import asyncio
import logging
from typing import Optional

import httpx

from .kiro_utils import (
    BASE_RETRY_DELAY,
    FIRST_TOKEN_MAX_RETRIES,
    MAX_RETRIES,
    STREAMING_READ_TIMEOUT,
    get_kiro_headers,
)
from ..kiro_auth_base import KiroAuthManager


lib_logger = logging.getLogger("rotator_library")


class KiroHttpClient:
    def __init__(
        self,
        auth_manager: KiroAuthManager,
        shared_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.auth_manager = auth_manager
        self._shared_client = shared_client
        self._owns_client = shared_client is None
        self.client: Optional[httpx.AsyncClient] = shared_client

    async def _get_client(self, stream: bool = False) -> httpx.AsyncClient:
        if self._shared_client is not None:
            return self._shared_client

        if self.client is None or self.client.is_closed:
            if stream:
                timeout_config = httpx.Timeout(
                    connect=30.0,
                    read=STREAMING_READ_TIMEOUT,
                    write=30.0,
                    pool=30.0,
                )
            else:
                timeout_config = httpx.Timeout(timeout=300.0)
            self.client = httpx.AsyncClient(
                timeout=timeout_config, follow_redirects=True
            )
        return self.client

    async def close(self) -> None:
        if not self._owns_client:
            return
        if self.client and not self.client.is_closed:
            try:
                await self.client.aclose()
            except Exception as exc:
                lib_logger.warning(f"Error closing HTTP client: {exc}")

    async def request_with_retry(
        self,
        method: str,
        url: str,
        json_data: dict,
        stream: bool = False,
    ) -> httpx.Response:
        max_retries = FIRST_TOKEN_MAX_RETRIES if stream else MAX_RETRIES
        client = await self._get_client(stream=stream)
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                token = await self.auth_manager.get_access_token()
                headers = get_kiro_headers(self.auth_manager, token)

                if stream:
                    headers["Connection"] = "close"
                    req = client.build_request(
                        method, url, json=json_data, headers=headers
                    )
                    response = await client.send(req, stream=True)
                else:
                    response = await client.request(
                        method, url, json=json_data, headers=headers
                    )

                if response.status_code == 200:
                    return response

                if response.status_code == 403:
                    await self.auth_manager.force_refresh()
                    continue

                if response.status_code in (429,) or 500 <= response.status_code < 600:
                    delay = BASE_RETRY_DELAY * (2**attempt)
                    await asyncio.sleep(delay)
                    continue

                return response

            except (httpx.TimeoutException, httpx.RequestError) as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    delay = BASE_RETRY_DELAY * (2**attempt)
                    await asyncio.sleep(delay)
                    continue
                break

        if last_error:
            raise last_error
        raise RuntimeError("Kiro request failed after retries")

    async def __aenter__(self) -> "KiroHttpClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
