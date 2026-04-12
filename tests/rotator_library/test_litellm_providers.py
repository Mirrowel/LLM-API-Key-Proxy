import pytest
from unittest.mock import patch

from rotator_library.litellm_providers import get_provider_route

MOCK_PROVIDERS = {
    "provider_with_slash": {
        "route": "myroute/",
    },
    "provider_without_slash": {
        "route": "myroute",
    },
    "provider_empty_route": {
        "route": "",
    },
    "provider_no_route": {
        "other_key": "value",
    },
    "provider_none_route": {
        "route": None,
    },
    "provider_int_route": {
        "route": 123,
    },
}

@pytest.mark.parametrize("provider_key, expected", [
    ("provider_with_slash", "myroute"),
    ("provider_without_slash", "myroute"),
    ("provider_empty_route", None),
    ("provider_no_route", None),
    ("unknown_provider", None),
    ("provider_none_route", None),
    ("provider_int_route", None),
])
@patch("rotator_library.litellm_providers.SCRAPED_PROVIDERS", MOCK_PROVIDERS)
def test_get_provider_route(provider_key, expected):
    assert get_provider_route(provider_key) == expected
