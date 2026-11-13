from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from retail_ai_agent.application.errors import InvalidRequest
from retail_ai_agent.domain.models import Actor


class UserProfile(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    role: Literal["manager", "ceo"]
    brands: list[str] = Field(default_factory=list)


class UserDirectory:
    def __init__(self, path: Path):
        try:
            self._profiles = TypeAdapter(dict[str, UserProfile]).validate_json(
                path.read_text(encoding="utf-8")
            )
        except OSError, ValueError:
            raise InvalidRequest(
                "Could not read USERS_FILE: expected JSON with manager/ceo roles "
                "and brands lists containing strings."
            ) from None

    def actor(self, user_id: str) -> Actor:
        profile = self._profiles.get(user_id)
        if profile is None:
            raise InvalidRequest("Unknown user in USERS_FILE.")
        return Actor(user_id, profile.role, tuple(profile.brands))

    def catalog_brands(self) -> tuple[str, ...]:
        return tuple({brand for profile in self._profiles.values() for brand in profile.brands})
