import os
import re
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path

from retail_ai_agent.application.errors import InvalidRequest, ServiceUnavailable


@dataclass(frozen=True)
class LocalState:
    directory: Path

    def __post_init__(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError:
            raise ServiceUnavailable(
                "Could not create DATA_DIR. Check access permissions."
            ) from None

    @property
    def reports_path(self) -> Path:
        return self.directory / "reports"

    @property
    def trace_path(self) -> Path:
        return self.directory / "trace.jsonl"

    def pseudonym_key(self) -> str:
        """Keep PII pseudonyms stable; publish a complete, owner-only key atomically."""
        path = self.directory / "pseudonym.key"
        try:
            if not path.exists():
                # mkstemp creates mode 0600. A hard link never overwrites a concurrent key.
                descriptor, temporary = tempfile.mkstemp(dir=self.directory, prefix=".pseudonym-")
                try:
                    with os.fdopen(descriptor, "w", encoding="ascii") as output:
                        output.write(secrets.token_hex(32))
                        output.flush()
                        os.fsync(output.fileno())
                    try:
                        os.link(temporary, path)
                    except FileExistsError:
                        pass
                finally:
                    os.unlink(temporary)
            path.chmod(0o600)
            key = path.read_text(encoding="ascii").strip()
        except OSError, UnicodeError:
            raise ServiceUnavailable(
                "Could not read pseudonym.key. Check the file and DATA_DIR permissions."
            ) from None
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise InvalidRequest("pseudonym.key is corrupted; restore the saved key.")
        return key
