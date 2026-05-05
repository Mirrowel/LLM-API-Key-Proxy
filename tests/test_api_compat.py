import json

from proxy_app.anthropic_errors import anthropic_error_response


def test_anthropic_error_response_shape_is_not_fastapi_wrapped() -> None:
    response = anthropic_error_response(
        status_code=429,
        error_type="rate_limit_error",
        message="Too many requests",
    )

    payload = json.loads(response.body.decode("utf-8"))
    assert response.status_code == 429
    assert payload == {
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": "Too many requests",
        },
    }
    assert "detail" not in payload
