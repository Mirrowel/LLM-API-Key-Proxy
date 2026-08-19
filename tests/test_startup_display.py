from __future__ import annotations

from proxy_app.startup_display import mask_secret_for_display


def test_proxy_api_key_display_is_redacted() -> None:
    value = "sk-proxy-secret-value"

    display = mask_secret_for_display(value)

    assert value not in display
    assert display == "Set (sk-p...alue, 21 chars)"


def test_short_proxy_api_key_display_is_fully_redacted() -> None:
    value = "secret"

    display = mask_secret_for_display(value)

    assert value not in display
    assert display == "Set (redacted, 6 chars)"
