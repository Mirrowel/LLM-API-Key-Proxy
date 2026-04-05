# SPDX-License-Identifier: LGPL-3.0-only

"""
Cross-provider request execution.

Orchestrates request attempts across multiple providers for a single
canonical model, using the ModelAliasRegistry to resolve targets.

Supports two retry modes:
- round_robin: Try one credential per provider, cycling through providers
- exhaust: Exhaust all credentials on provider N before trying N+1
"""

import json
import logging
from typing import Any, AsyncGenerator, List, Optional, TYPE_CHECKING, Union

from ..model_alias_registry import AliasTarget, ModelAliasRegistry
from ..core.errors import NoAvailableKeysError

if TYPE_CHECKING:
    from ..client.rotating_client import RotatingClient

lib_logger = logging.getLogger("rotator_library")


class CrossProviderExecutor:
    """
    Executes requests across multiple providers for alias-based models.

    This wraps the existing per-provider RotatingClient.acompletion() flow,
    trying multiple provider targets when one fails.
    """

    def __init__(
        self,
        client: "RotatingClient",
        alias_registry: ModelAliasRegistry,
    ) -> None:
        self._client = client
        self._registry = alias_registry

    async def execute(
        self,
        canonical_model: str,
        targets: List[AliasTarget],
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Union[Any, AsyncGenerator[str, None]]:
        """
        Execute a request across multiple providers.

        Args:
            canonical_model: The canonical model name (e.g., "deepseek-v3")
            targets: Ordered list of provider targets to try
            request: FastAPI Request object
            pre_request_callback: Optional callback
            **kwargs: Request parameters (messages, stream, etc.)

        Returns:
            Response object or async generator for streaming
        """
        retry_mode = self._registry.get_retry_mode(canonical_model)
        is_streaming = kwargs.get("stream", False)

        # Filter targets to providers that have credentials
        available_targets = [
            t for t in targets if t.provider in self._client.all_credentials
        ]

        if not available_targets:
            provider_list = ", ".join(t.provider for t in targets)
            raise NoAvailableKeysError(
                f"No credentials available for any provider of alias '{canonical_model}'. "
                f"Configured providers: {provider_list}"
            )

        lib_logger.info(
            f"Cross-provider routing for '{canonical_model}': "
            f"{len(available_targets)} providers available, mode={retry_mode}"
        )

        if is_streaming:
            return self._execute_streaming(
                canonical_model, available_targets, retry_mode,
                request, pre_request_callback, **kwargs,
            )
        else:
            return await self._execute_non_streaming(
                canonical_model, available_targets, retry_mode,
                request, pre_request_callback, **kwargs,
            )

    async def _execute_non_streaming(
        self,
        canonical_model: str,
        targets: List[AliasTarget],
        retry_mode: str,
        request: Optional[Any],
        pre_request_callback: Optional[callable],
        **kwargs,
    ) -> Any:
        """Non-streaming cross-provider execution."""
        last_error: Optional[Exception] = None

        for i, target in enumerate(targets):
            provider_model = target.full_model
            lib_logger.info(
                f"[{canonical_model}] Trying provider {i + 1}/{len(targets)}: "
                f"{target.provider} (model: {target.model_name})"
            )

            try:
                # Build kwargs for this specific provider target
                target_kwargs = kwargs.copy()
                target_kwargs["model"] = provider_model

                response = await self._client.acompletion(
                    request=request,
                    pre_request_callback=pre_request_callback,
                    **target_kwargs,
                )

                # Check if the response is an error response from the executor
                # (RequestErrorAccumulator returns a dict with "error" key)
                if isinstance(response, dict) and "error" in response:
                    error_msg = response["error"].get("message", "Unknown error")
                    lib_logger.warning(
                        f"[{canonical_model}] Provider {target.provider} returned error: "
                        f"{error_msg}. Trying next provider."
                    )
                    last_error = NoAvailableKeysError(error_msg)
                    continue

                lib_logger.info(
                    f"[{canonical_model}] Success via {target.provider}"
                )
                return response

            except NoAvailableKeysError as e:
                lib_logger.warning(
                    f"[{canonical_model}] Provider {target.provider} exhausted: {e}. "
                    f"Trying next provider."
                )
                last_error = e
                continue
            except Exception as e:
                lib_logger.warning(
                    f"[{canonical_model}] Provider {target.provider} failed: {e}. "
                    f"Trying next provider."
                )
                last_error = e
                continue

        # All providers exhausted
        lib_logger.error(
            f"[{canonical_model}] All {len(targets)} providers exhausted."
        )
        if last_error:
            raise last_error
        raise NoAvailableKeysError(
            f"All providers exhausted for alias '{canonical_model}'"
        )

    async def _execute_streaming(
        self,
        canonical_model: str,
        targets: List[AliasTarget],
        retry_mode: str,
        request: Optional[Any],
        pre_request_callback: Optional[callable],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming cross-provider execution.

        Returns an async generator. If a provider fails during streaming,
        it cannot retry mid-stream (data already sent to client). Provider
        failover happens only at connection time (before first chunk).
        """

        async def _stream_with_failover():
            last_error: Optional[Exception] = None

            for i, target in enumerate(targets):
                provider_model = target.full_model
                lib_logger.info(
                    f"[{canonical_model}] Trying streaming provider {i + 1}/"
                    f"{len(targets)}: {target.provider} (model: {target.model_name})"
                )

                try:
                    target_kwargs = kwargs.copy()
                    target_kwargs["model"] = provider_model

                    response_stream = await self._client.acompletion(
                        request=request,
                        pre_request_callback=pre_request_callback,
                        **target_kwargs,
                    )

                    # For streaming, acompletion returns an async generator.
                    # We need to peek at it to check for immediate errors.
                    first_chunk = None
                    async for chunk in response_stream:
                        # Check if the first chunk is an error
                        if first_chunk is None:
                            first_chunk = chunk
                            # Check for error in first chunk
                            if isinstance(chunk, str) and chunk.startswith("data: "):
                                content = chunk[len("data: "):].strip()
                                if content != "[DONE]":
                                    try:
                                        parsed = json.loads(content)
                                        if "error" in parsed:
                                            error_msg = parsed["error"].get(
                                                "message", "Unknown error"
                                            )
                                            # Check if it's a retriable error
                                            error_type = parsed["error"].get("type", "")
                                            if error_type in (
                                                "proxy_error",
                                                "no_available_keys",
                                                "proxy_all_credentials_exhausted",
                                                "proxy_timeout",
                                            ):
                                                lib_logger.warning(
                                                    f"[{canonical_model}] Provider "
                                                    f"{target.provider} stream error: "
                                                    f"{error_msg}. Trying next."
                                                )
                                                last_error = NoAvailableKeysError(
                                                    error_msg
                                                )
                                                break
                                    except json.JSONDecodeError:
                                        pass

                        yield chunk

                    # If we yielded at least one non-error chunk, we're done
                    if first_chunk is not None:
                        lib_logger.info(
                            f"[{canonical_model}] Stream complete via {target.provider}"
                        )
                        return

                except NoAvailableKeysError as e:
                    lib_logger.warning(
                        f"[{canonical_model}] Streaming provider {target.provider} "
                        f"exhausted: {e}. Trying next."
                    )
                    last_error = e
                    continue
                except Exception as e:
                    lib_logger.warning(
                        f"[{canonical_model}] Streaming provider {target.provider} "
                        f"failed: {e}. Trying next."
                    )
                    last_error = e
                    continue

            # All providers exhausted — emit error as SSE
            lib_logger.error(
                f"[{canonical_model}] All {len(targets)} streaming providers exhausted."
            )
            error_msg = str(last_error) if last_error else "All providers exhausted"
            error_data = {
                "error": {
                    "message": f"All providers exhausted for '{canonical_model}': {error_msg}",
                    "type": "proxy_error",
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

        return _stream_with_failover()
