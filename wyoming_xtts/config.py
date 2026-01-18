from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from TTS.tts.configs.xtts_config import XttsConfig

from wyoming_xtts import SERVICE_NAME

_XTTS_DEFAULTS = XttsConfig()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="XTTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    uri: str = Field(default="tcp://0.0.0.0:10200", description="Server URI")
    assets: Path = Field(default=Path("./assets"), description="Assets directory path")
    deepspeed: bool = Field(default=False, description="Enable DeepSpeed acceleration")
    no_download_model: bool = Field(default=False, description="Disable automatic model download")
    log_level: str = Field(default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")
    zeroconf: str | None = Field(
        default=SERVICE_NAME,
        description="Zeroconf service name (enables discovery if set)",
    )
    language_fallback: str | None = Field(default=None, description="Fallback language when detection fails")
    language_no_detect: bool = Field(
        default=False,
        description="Disable language auto-detection, always use fallback",
    )
    temperature: float = Field(
        default=_XTTS_DEFAULTS.temperature,
        ge=0.0,
        le=1.0,
        description="Synthesis temperature (0.0-1.0)",
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier (0.5-2.0)")
    top_k: int = Field(default=_XTTS_DEFAULTS.top_k, ge=1, le=100, description="Top-k sampling (1-100)")
    top_p: float = Field(
        default=_XTTS_DEFAULTS.top_p,
        ge=0.0,
        le=1.0,
        description="Top-p nucleus sampling (0.0-1.0)",
    )
    repetition_penalty: float = Field(
        default=_XTTS_DEFAULTS.repetition_penalty,
        ge=1.0,
        le=20.0,
        description="Repetition penalty (1.0-20.0)",
    )
    stream_chunk_size: int = Field(default=20, ge=1, le=500, description="Streaming chunk size in tokens")
    min_segment_chars: int = Field(
        default=20,
        ge=1,
        le=500,
        description="Minimum characters before synthesizing a segment",
    )
    seed: int | None = Field(
        default=42,
        description="Fixed seed for reproducible synthesis (None for random)",
    )
