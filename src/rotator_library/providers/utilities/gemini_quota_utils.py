import re
import json
import logging
from typing import Optional, Dict, Any

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


def parse_duration(duration_str: str) -> Optional[int]:
    if not duration_str:
        return None

    pure_seconds_match = re.match(r"^([\d.]+)s$", duration_str)
    if pure_seconds_match:
        return int(float(pure_seconds_match.group(1)))

    total_seconds = 0
    patterns = [
        (r"(\d+)h", 3600),
        (r"(\d+)m", 60),
        (r"([\d.]+)s", 1),
    ]
    for pattern, multiplier in patterns:
        match = re.search(pattern, duration_str)
        if match:
            total_seconds += float(match.group(1)) * multiplier

    return int(total_seconds) if total_seconds > 0 else None


def _extract_body_from_exception(error: Exception) -> str:
    """
    Extract the error body string from various exception types.

    Handles litellm, httpx, and generic exceptions.  When the body is a
    Python object (dict/list) rather than a raw JSON string, we re-serialize
    it with json.dumps so downstream JSON parsing succeeds.
    """
    # httpx response body (raw JSON string)
    if hasattr(error, 'response') and hasattr(error.response, 'text'):
        try:
            return error.response.text
        except Exception:
            pass

    # litellm body attribute – can be dict, list, str, or None
    body_attr = getattr(error, 'body', None)
    if body_attr is not None:
        if isinstance(body_attr, str):
            return body_attr
        if isinstance(body_attr, (dict, list)):
            try:
                return json.dumps(body_attr)
            except (TypeError, ValueError):
                return str(body_attr)
        return str(body_attr)

    # litellm message attribute
    message = getattr(error, 'message', None)
    if message:
        return str(message)

    return str(error)


def parse_google_quota_error(error: Exception, error_body: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Parse Google API 429 RESOURCE_EXHAUSTED errors.

    Google's error format includes:
    - RetryInfo with retryDelay (e.g., "30s")
    - ErrorInfo with reason (e.g., "RATE_LIMIT_EXCEEDED") and metadata

    Returns dict with retry_after, reason, etc., or None if not parseable.
    """
    body = error_body if error_body else _extract_body_from_exception(error)

    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        if "RESOURCE_EXHAUSTED" in body:
            return {
                "retry_after": 60,
                "reason": "per_minute_rate_limit",
            }
        return None

    if isinstance(parsed, list) and len(parsed) > 0:
        first = parsed[0]
        if isinstance(first, dict):
            parsed = first

    error_obj = parsed
    if isinstance(parsed, dict) and "error" in parsed and isinstance(parsed["error"], dict):
        error_obj = parsed["error"]

    if isinstance(error_obj, dict):
        error_status = error_obj.get("status", "")
        error_code = error_obj.get("code", 0)
        if error_status == "RESOURCE_EXHAUSTED" or error_code == 429:
            details = error_obj.get("details", [])
            if not details:
                return {
                    "retry_after": 60,
                    "reason": "per_minute_rate_limit",
                }

            retry_after = None
            reason = None
            quota_reset_timestamp = None

            for detail in details:
                detail_type = detail.get("@type", "")

                if "RetryInfo" in detail_type:
                    retry_delay = detail.get("retryDelay", "")
                    retry_after = parse_duration(retry_delay)

                if "ErrorInfo" in detail_type:
                    reason = detail.get("reason", "")
                    metadata = detail.get("metadata", {})
                    quota_metric = metadata.get("quota_metric", "")
                    reset_delay = metadata.get("quotaResetDelay", "")
                    if reset_delay and not quota_reset_timestamp:
                        parsed_delay = parse_duration(reset_delay)
                        if parsed_delay:
                            import time
                            quota_reset_timestamp = time.time() + parsed_delay
                    if not reason and quota_metric:
                        reason = quota_metric

            if retry_after is None and reason is None:
                return {
                    "retry_after": 60,
                    "reason": "per_minute_rate_limit",
                }

            if not reason:
                reason = "QUOTA_EXHAUSTED"

            result = {
                "retry_after": retry_after,
                "reason": reason,
            }
            if quota_reset_timestamp:
                result["quota_reset_timestamp"] = quota_reset_timestamp

            return result

    return None
