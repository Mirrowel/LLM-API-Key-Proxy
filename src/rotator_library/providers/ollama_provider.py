# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Native protocol provider for a local (or cloud) Ollama server (G8 final).

Ollama speaks newline-delimited JSON natively on ``/api/chat``,
``/api/generate``, and ``/api/embed``. ``speaks`` declares both faces over
one provider identity (same native routes/shapes, different base):

- ``ollama`` (the default face, first entry): the local daemon on
  ``http://localhost:11434``. Auth is optional — a bare local server needs
  no credential, while a reverse-proxied deployment may present an
  ``Authorization: Bearer`` token. Zero-credential routing is declared
  through ``default_auth_mode = "none"`` so the client mints the internal
  no-auth slot; the header override below adds the bearer only when a real
  secret exists (the user's explicit dual-mode ruling).
- ``cloud``: Ollama's hosted cloud at ``https://ollama.com``, addressed as
  ``ollama:cloud/model``. The routes and response shapes are identical
  (``/api/tags`` lists publicly even unauthenticated), auth is
  ``Authorization: Bearer`` with a real API key, and the direct cloud API
  takes the BARE model names from the library — an addressed id carrying
  the local daemon's trailing ``-cloud`` routing suffix is normalized off
  on this face only (see ``normalize_native_model``).

Model listing is the shared, protocol-aware interface implementation
(``models[].name`` shape, ``/api/tags`` route); a failed listing is an
honest empty.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")

#: The local daemon addresses cloud models with this routing suffix; the
#: direct cloud API serves the bare library name.
_CLOUD_SUFFIX = "-cloud"


class OllamaProvider(ProviderInterface):
    """Provider plugin exposing Ollama's native HTTP protocol."""

    provider_env_name = "ollama"
    # First entry is the default face. The cloud face is the same ollama
    # protocol on Ollama's hosted base; its declared bearer auth is the
    # only auth difference (the local face inherits the protocol's none).
    speaks = (
        "ollama",
        ("cloud", "ollama", {"base": "https://ollama.com", "auth_mode": "bearer"}),
    )
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
        """Per-face credential headers.

        The local face keeps the dual-mode ruling: the minted no-auth slot
        stays anonymous, and a real secret (a reverse-proxied Ollama) rides
        the conventional Bearer header. The cloud face DECLARES
        ``auth_mode: "bearer"`` — but the internal no-auth sentinel is
        never a credential: sending it as a literal Bearer would leak the
        marker string upstream (the same leak class the no-auth closure
        eliminated). A cloud call with no real key therefore goes out
        anonymous and fails the cloud's 401 honestly, rather than
        presenting garbage as a credential.
        """

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        real = self._is_real_credential(credential_identifier)
        if real:
            headers["Authorization"] = f"Bearer {credential_identifier}"
        return headers

    def normalize_native_model(self, model: str, profile: Optional[str] = None) -> str:
        """Upstream model id for a face (cloud ids normalize to the bare name).

        The local daemon routes cloud models through the trailing
        ``-cloud`` suffix, so local-addressed ids keep it verbatim; the
        direct cloud API serves the bare library name and rejects the
        suffix. Stripping is therefore a CLOUD-face rule only — this
        override is the seam that sees both the id and the resolved face.
        """

        base = super().normalize_native_model(model)
        if profile == "cloud" and base.endswith(_CLOUD_SUFFIX):
            return base[: -len(_CLOUD_SUFFIX)]
        return base

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
