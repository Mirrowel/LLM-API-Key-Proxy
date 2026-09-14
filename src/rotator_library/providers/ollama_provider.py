# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Native protocol provider for a local (or remote) Ollama server (G8 final).

Ollama speaks newline-delimited JSON natively on ``/api/chat``,
``/api/generate``, and ``/api/embed``. ``speaks`` declares that one face;
the endpoints (including ``/api/tags`` for listing) inherit from the
protocol registry. Auth is optional: a bare local server needs no
credential, while a reverse-proxied deployment may present an
``Authorization: Bearer`` token. Zero-credential routing is declared
through ``default_auth_mode = "none"`` so the client mints the internal
no-auth slot; the header override below adds the bearer only when a real
secret is configured.

Model listing is the shared, protocol-aware interface implementation
(``models[].name`` shape, ``/api/tags`` route); a failed listing is an
honest empty.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")


class OllamaProvider(ProviderInterface):
    """Provider plugin exposing Ollama's native HTTP protocol."""

    provider_env_name = "ollama"
    speaks = ("ollama",)
    default_api_base = "http://localhost:11434"
    native_streaming_supported = True
    # Local Ollama needs no credential; the header is added only when a real
    # secret is configured (optional bearer).
    default_auth_mode = "none"
    skip_cost_calculation = True

    def get_native_headers(
        self,
        credential_identifier: str,
        model: str = "",
        operation: str = "chat",
        profile: Optional[str] = None,
    ) -> Dict[str, str]:
        """Optional bearer auth: no header when the no-auth slot is in play.

        The declared auth mode is ``none`` (a local server answers without
        credentials), so the inherited behavior would return no header at
        all — this override is the genuinely custom piece: a reverse-proxied
        Ollama that DOES have a real secret gets the conventional Bearer
        header, while the minted no-auth slot stays anonymous.
        """

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._is_real_credential(credential_identifier):
            headers["Authorization"] = f"Bearer {credential_identifier}"
        return headers

    @staticmethod
    def _is_real_credential(credential_identifier: str) -> bool:
        if not credential_identifier:
            return False
        try:
            from ..client.scopes import NO_AUTH_CREDENTIAL

            return credential_identifier != NO_AUTH_CREDENTIAL
        except Exception:  # pragma: no cover - import guard
            return not str(credential_identifier).startswith("__proxy_")


def get_provider() -> OllamaProvider:
    """Convenience accessor used by tests and tooling."""

    return OllamaProvider()


__all__ = ["OllamaProvider", "get_provider"]
