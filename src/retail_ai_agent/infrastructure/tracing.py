import json
from datetime import UTC, datetime
from pathlib import Path


class JsonlTrace:
    def __init__(self, path: Path):
        self.path = path

    def record(self, event: str, **fields) -> None:
        record = {
            "time": datetime.now(UTC).isoformat(),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
