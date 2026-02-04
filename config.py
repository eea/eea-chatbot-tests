"""Configuration settings for chatbot Playwright tests."""

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables or JSON file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # Chatbot URLs
    chatbot_base_url: str = Field(
        default="http://localhost:3000",
        description="Base URL for the Volto frontend",
    )
    chatbot_path: str = Field(
        default="/chatbot",
        description="Path to the chatbot page",
    )

    # Browser settings
    headless: bool = Field(
        default=True,
        description="Run browser in headless mode",
    )
    browser: Literal["chromium", "firefox", "webkit"] = Field(
        default="chromium",
        description="Browser to use for testing",
    )
    timeout: int = Field(
        default=120000,
        description="Default timeout for operations (ms)",
    )
    expect_timeout: int = Field(
        default=30000,
        description="Default timeout for expect operations (ms)",
    )

    # Report settings
    reports_dir: str = Field(
        default="./chatbot_tests/reports",
        description="Directory for test reports",
    )

    # Fixtures settings
    fixtures_dir: Optional[str] = Field(
        default=None,
        description="Directory containing test fixtures (defaults to ./fixtures relative to package)",
    )

    # PDF settings
    pdf_font: Optional[str] = Field(
        default=None,
        description="Path to TTF font file for PDF generation (must support Unicode/emoji)",
    )

    # LLM Analysis settings
    enable_llm_analysis: bool = Field(
        default=False,
        description="Enable LLM-based response analysis",
    )
    llm_model: str = Field(
        default="Inhouse-LLM/gpt-oss-120b",
        description="LLM model to use for analysis (required if enable_llm_analysis is true)",
    )
    llm_url: str = Field(
        default="https://llmgw.eea.europa.eu",
        description="Base URL for LLM API endpoint (required if enable_llm_analysis is true)",
    )
    llm_api_key: str = Field(
        default="",
        description="API key for LLM provider (required if enable_llm_analysis is true)",
    )

    @model_validator(mode='after')
    def validate_llm_settings(self):
        """Validate that LLM settings are provided when analysis is enabled."""
        if self.enable_llm_analysis:
            if not self.llm_url:
                raise ValueError(
                    "LLM_URL must be set when ENABLE_LLM_ANALYSIS is true"
                )
            if not self.llm_api_key:
                raise ValueError(
                    "LLM_API_KEY must be set when ENABLE_LLM_ANALYSIS is true"
                )

            if not self.llm_model:
                raise ValueError(
                    "LLM_MODEL must be set when ENABLE_LLM_ANALYSIS is true"
                )
        return self

    @property
    def chatbot_url(self) -> str:
        """Full URL to the chatbot page."""
        return f"{self.chatbot_base_url}{self.chatbot_path}"

    @property
    def reports_path(self) -> Path:
        """Path object for reports directory."""
        return Path(self.reports_dir)

    @property
    def fixtures_path(self) -> Path:
        """Get the path to the fixtures directory."""
        if self.fixtures_dir:
            return Path(self.fixtures_dir)
        return Path(__file__).parent / "fixtures"

    @classmethod
    def from_json(cls, json_path: Path) -> "Settings":
        """Load settings from a JSON config file.

        JSON keys use snake_case matching the field names.
        Environment variables and .env still override JSON values.
        """
        with open(json_path) as f:
            config_data = json.load(f)
        return cls(**config_data)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the current settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    """Set the global settings instance."""
    global _settings
    _settings = settings


def load_settings_from_json(json_path: Path) -> Settings:
    """Load settings from JSON file and set as global instance."""
    global _settings
    _settings = Settings.from_json(json_path)
    return _settings
