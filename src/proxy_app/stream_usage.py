from dataclasses import dataclass
from typing import Any


@dataclass
class StreamUsageTracker:
    response_id: str | None = None
    model: str | None = None
    created: int | None = None
    usage: dict[str, Any] | None = None

    def ingest_chunk(self, chunk_data: dict[str, Any]) -> None:
        if self.response_id is None:
            response_id = chunk_data.get("id")
            if isinstance(response_id, str):
                self.response_id = response_id

        if self.model is None:
            model = chunk_data.get("model")
            if isinstance(model, str):
                self.model = model

        if self.created is None:
            created = chunk_data.get("created")
            if isinstance(created, int):
                self.created = created

        usage = chunk_data.get("usage")
        if isinstance(usage, dict) and usage:
            self.usage = usage

    def build_logging_payload(self) -> dict[str, Any]:
        return {
            "id": self.response_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [],
            "usage": self.usage,
        }
