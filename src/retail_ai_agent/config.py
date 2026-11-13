from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_provider: Literal["gemini", "openrouter"] = "gemini"
    google_ai_studio_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    google_cloud_project: str
    gemini_billing_tier: Literal["standard", "free"] = "standard"
    gemini_model: str | None = None
    openrouter_model: str = Field(default="openai/gpt-4.1-mini", min_length=1)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    bigquery_dataset: str = "bigquery-public-data.thelook_ecommerce"
    users_file: Path = Path("config/users.json")
    default_user: str = "manager_a"
    data_dir: Path = Path(".retail-agent")
    startup_timeout_seconds: float = Field(default=15, gt=0)
    request_budget_usd: float = Field(default=1, gt=0)
    question_timeout_seconds: float = Field(default=30, gt=0)
    report_timeout_seconds: float = Field(default=120, gt=0)
    model_timeout_seconds: float = Field(default=30, gt=0)
    model_max_retries: int = Field(default=2, ge=0, le=5)
    max_model_calls: int = Field(default=12, gt=0)
    max_output_tokens: int = Field(default=2048, gt=0)
    max_sql_attempts: int = Field(default=4, gt=0)
    max_result_rows: int = Field(default=50, gt=0, le=500)
    history_max_turns: int = Field(default=6, gt=0)
    history_max_chars: int = Field(default=24_000, gt=0)
    bigquery_max_bytes: int = Field(default=1_000_000_000, gt=0)
    # Legacy OpenRouter accounting only. Gemini uses its published model-specific tariff.
    input_usd_per_million: float = Field(default=10, ge=0)
    output_usd_per_million: float = Field(default=40, ge=0)

    @field_validator("google_ai_studio_api_key", "openrouter_api_key")
    @classmethod
    def require_selected_api_key(
        cls, value: SecretStr | None, info: ValidationInfo
    ) -> SecretStr | None:
        provider = info.data.get("model_provider")
        selected = (provider == "gemini" and info.field_name == "google_ai_studio_api_key") or (
            provider == "openrouter" and info.field_name == "openrouter_api_key"
        )
        if selected and (value is None or not value.get_secret_value().strip()):
            raise ValueError("API key is required for the selected model provider")
        return value

    @field_validator("gemini_model")
    @classmethod
    def require_gemini_model(cls, value: str | None, info: ValidationInfo) -> str | None:
        name = value.strip() if value else None
        if info.data.get("model_provider") == "gemini" and not name:
            raise ValueError("Set GEMINI_MODEL to the model ID from Google AI Studio")
        return name

    @property
    def model_name(self) -> str:
        if self.model_provider == "openrouter":
            return self.openrouter_model
        assert self.gemini_model is not None
        return self.gemini_model
