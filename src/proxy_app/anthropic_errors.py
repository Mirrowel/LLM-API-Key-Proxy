from fastapi.responses import JSONResponse


def anthropic_error_response(
    *,
    status_code: int,
    error_type: str,
    message: str,
) -> JSONResponse:
    payload = {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }
    return JSONResponse(status_code=status_code, content=payload)
