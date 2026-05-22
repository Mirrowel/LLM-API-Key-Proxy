"""Admin API for browsing transaction logs and failure records."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/v1/admin", tags=["admin-logs"])


def _read_transaction_file(target: Path):
    """Read and parse a transaction log file (runs in a thread to avoid blocking)."""
    try:
        with open(target) as f:
            if target.suffix == ".jsonl":
                lines = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            lines.append(json.loads(line))
                        except json.JSONDecodeError:
                            lines.append({"raw": line})
                return lines
            else:
                return json.load(f)
    except json.JSONDecodeError:
        with open(target) as f:
            return {"raw": f.read()}


def _get_logs_dir() -> Path:
    import sys
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path.cwd()
    logs_dir = base / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def _parse_dir_name(name: str) -> Optional[dict]:
    parts = name.split("_")
    if len(parts) < 5:
        return None

    date_str = parts[0]
    time_str = parts[1]

    if len(parts) >= 6 and parts[2] in ("oai", "ant"):
        api_format = parts[2]
        provider = parts[3]
        model = "_".join(parts[4:-1])
    else:
        api_format = "oai"
        provider = parts[2]
        model = "_".join(parts[3:-1])

    request_id = parts[-1]

    try:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        month = int(date_str[:2])
        day = int(date_str[2:])
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6]) if len(time_str) >= 6 else 0
        timestamp = datetime(now.year, month, day, hour, minute, second)
        from datetime import timedelta
        if timestamp > now + timedelta(days=1):
            timestamp = datetime(now.year - 1, month, day, hour, minute, second)
    except (ValueError, IndexError):
        timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

    return {
        "request_id": request_id,
        "timestamp": timestamp.isoformat(),
        "provider": provider,
        "model": model,
        "api_format": api_format,
        "dir_name": name,
    }


def _load_metadata(tx_dir: Path) -> dict:
    meta_path = tx_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return {}


def _extract_prompt_preview(tx_dir: Path, api_format: str, max_len: int = 60) -> str:
    if api_format == "ant":
        req_file = tx_dir / "anthropic_request.json"
    else:
        req_file = tx_dir / "request.json"

    if not req_file.exists():
        return ""

    try:
        with open(req_file) as f:
            data = json.load(f)
        data = data.get("data", data)
        messages = data.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
                    text = "\n".join(parts)
                else:
                    continue
                text = text.strip().replace("\n", " ").replace("\r", " ")
                if text:
                    return text[:max_len] + ("..." if len(text) > max_len else "")
    except Exception:
        pass
    return ""


def _list_transaction_files(tx_dir: Path) -> list[str]:
    files = []
    for f in sorted(tx_dir.iterdir()):
        if f.is_file() and f.name != "metadata.json":
            files.append(f.name)
        elif f.is_dir():
            for sub in sorted(f.iterdir()):
                if sub.is_file():
                    files.append(f"{f.name}/{sub.name}")
    return files


@router.get("/transactions")
async def list_transactions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    provider: Optional[str] = None,
    model: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    logs_dir = _get_logs_dir()
    tx_dir = logs_dir / "transactions"
    if not tx_dir.exists():
        return {"transactions": [], "total": 0, "page": page, "page_size": page_size}

    start = (page - 1) * page_size
    end = start + page_size
    matched = 0
    page_entries: list[dict] = []

    for d in sorted(tx_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        parsed = _parse_dir_name(d.name)
        if not parsed:
            continue

        if provider and parsed["provider"] != provider:
            continue
        if model and model.lower() not in parsed["model"].lower():
            continue
        if search and search.lower() not in parsed["request_id"].lower():
            continue

        meta = _load_metadata(d)
        if meta.get("timestamp_utc"):
            try:
                parsed["timestamp"] = datetime.fromisoformat(
                    meta["timestamp_utc"].replace("Z", "+00:00")
                ).replace(tzinfo=None).isoformat()
            except Exception:
                pass

        status_code = meta.get("status_code")
        if status == "success" and status_code != 200:
            continue
        if status == "error" and (not status_code or status_code == 200):
            continue

        if date_from:
            try:
                from_dt = datetime.fromisoformat(date_from)
                entry_dt = datetime.fromisoformat(parsed["timestamp"])
                if entry_dt < from_dt:
                    continue
            except Exception:
                pass
        if date_to:
            try:
                to_dt = datetime.fromisoformat(date_to)
                entry_dt = datetime.fromisoformat(parsed["timestamp"])
                if entry_dt > to_dt:
                    continue
            except Exception:
                pass

        if start <= matched < end:
            usage = meta.get("usage", {})
            preview = _extract_prompt_preview(d, parsed["api_format"])

            has_provider_logs = meta.get("has_provider_logs", False)
            has_request = (d / "request.json").exists() or (d / "anthropic_request.json").exists()
            has_response = (d / "response.json").exists() or (d / "anthropic_response.json").exists()

            if has_provider_logs:
                log_level = "full"
            elif has_request or has_response:
                log_level = "req_resp"
            else:
                log_level = "metadata"

            tokens_in = usage.get("prompt_tokens", 0) or 0
            tokens_out = usage.get("completion_tokens", 0) or 0
            tokens_cached = 0
            prompt_details = usage.get("prompt_tokens_details")
            if isinstance(prompt_details, dict):
                tokens_cached = prompt_details.get("cached_tokens", 0) or 0
            completion_details = usage.get("completion_tokens_details")
            write_tokens = 0
            if isinstance(completion_details, dict):
                write_tokens = completion_details.get("reasoning_tokens", 0) or 0

            approx_cost = meta.get("approx_cost")

            page_entries.append({
                "request_id": parsed["request_id"],
                "timestamp": parsed["timestamp"],
                "provider": parsed["provider"],
                "model": parsed["model"],
                "status": str(status_code) if status_code else "-",
                "duration_ms": meta.get("duration_ms", 0) or 0,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tokens_cached": tokens_cached,
                "reasoning_tokens": write_tokens,
                "approx_cost": approx_cost,
                "prompt_preview": preview,
                "log_level": log_level,
                "format": parsed["api_format"],
                "credential_masked": meta.get("credential_masked"),
            })

        matched += 1

    return {
        "transactions": page_entries,
        "total": matched,
        "page": page,
        "page_size": page_size,
    }


@router.get("/transactions/{request_id}")
async def get_transaction_detail(request_id: str):
    logs_dir = _get_logs_dir()
    tx_dir = logs_dir / "transactions"
    if not tx_dir.exists():
        raise HTTPException(status_code=404, detail="No transactions directory")

    matching = None
    for d in tx_dir.iterdir():
        if d.is_dir() and d.name.endswith(f"_{request_id}"):
            matching = d
            break

    if not matching:
        raise HTTPException(status_code=404, detail="Transaction not found")

    parsed = _parse_dir_name(matching.name)
    if not parsed:
        raise HTTPException(status_code=404, detail="Invalid transaction directory")

    meta = _load_metadata(matching)
    usage = meta.get("usage", {})
    files = _list_transaction_files(matching)

    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    cached = prompt_details.get("cached_tokens", 0) or 0 if isinstance(prompt_details, dict) else 0
    reasoning = completion_details.get("reasoning_tokens", 0) or 0 if isinstance(completion_details, dict) else 0

    return {
        "request_id": parsed["request_id"],
        "timestamp": parsed["timestamp"],
        "provider": parsed["provider"],
        "model": parsed["model"],
        "status": str(meta.get("status_code", "-")),
        "duration_ms": meta.get("duration_ms", 0) or 0,
        "tokens": {
            "prompt": usage.get("prompt_tokens", 0) or 0,
            "completion": usage.get("completion_tokens", 0) or 0,
            "total": usage.get("total_tokens", 0) or 0,
            "cached": cached,
            "reasoning": reasoning,
        },
        "approx_cost": meta.get("approx_cost"),
        "files": files,
        "has_provider_logs": meta.get("has_provider_logs", False),
    }


@router.get("/transactions/{request_id}/files/{file_path:path}")
async def get_transaction_file(request_id: str, file_path: str):
    logs_dir = _get_logs_dir()
    tx_dir = logs_dir / "transactions"
    if not tx_dir.exists():
        raise HTTPException(status_code=404, detail="No transactions directory")

    matching = None
    for d in tx_dir.iterdir():
        if d.is_dir() and d.name.endswith(f"_{request_id}"):
            matching = d
            break

    if not matching:
        raise HTTPException(status_code=404, detail="Transaction not found")

    target = (matching / file_path).resolve()
    if not str(target).startswith(str(matching.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        content = await asyncio.to_thread(_read_transaction_file, target)
        return JSONResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _tail_lines(path: Path, max_lines: int = 2000) -> list[str]:
    """Read up to max_lines from the end of a file without loading the entire file."""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size == 0:
        return []

    chunk_size = min(size, max_lines * 512)
    lines: list[str] = []
    with open(path, "rb") as f:
        f.seek(max(0, size - chunk_size))
        if f.tell() > 0:
            f.readline()
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                lines.append(stripped.decode("utf-8", errors="replace"))
    return lines[-max_lines:]


@router.get("/failures")
async def list_failures(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    logs_dir = _get_logs_dir()
    failures_path = logs_dir / "failures.log"
    if not failures_path.exists():
        return {"failures": [], "total": 0, "page": page, "page_size": page_size}

    raw_lines = _tail_lines(failures_path)
    entries = []
    provider_counts: dict[str, int] = {}
    error_type_counts: dict[str, int] = {}
    for line in raw_lines:
        try:
            data = json.loads(line)
            model = data.get("model", "N/A")
            provider = model.split("/")[0] if "/" in model else "unknown"
            error_type = data.get("error_type", "Unknown")
            entries.append({
                "timestamp": data.get("timestamp", ""),
                "model": model,
                "provider": provider,
                "error_type": error_type,
                "error_message": data.get("error_message", ""),
                "raw_response": data.get("raw_response", ""),
                "request_headers": data.get("request_headers"),
                "error_chain": [
                    e.get("message", str(e)) if isinstance(e, dict) else str(e)
                    for e in (data.get("error_chain") or [])
                ],
                "api_key_ending": data.get("api_key_ending", ""),
                "attempt_number": data.get("attempt_number", 1),
            })
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        except json.JSONDecodeError:
            continue

    entries.sort(key=lambda x: x["timestamp"], reverse=True)
    total = len(entries)
    start = (page - 1) * page_size
    page_entries = entries[start:start + page_size]

    return {
        "failures": page_entries,
        "total": total,
        "page": page,
        "page_size": page_size,
        "providers": [
            {"name": name, "count": count}
            for name, count in sorted(provider_counts.items(), key=lambda x: -x[1])
        ],
        "error_types": [
            {"type": et, "count": count}
            for et, count in sorted(error_type_counts.items(), key=lambda x: -x[1])
        ],
    }
