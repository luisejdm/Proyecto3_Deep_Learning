"""Configuration for the API-based agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    model_name: str = "meta/llama-3.3-70b-instruct"
    nvidia_env_key: str = "NVIDIA_API_KEY"
    base_url: str = "https://integrate.api.nvidia.com/v1"


def get_env_path() -> Path:
    return Path(__file__).resolve().parents[1] / ".env"


def load_environment() -> str | None:
    env_path = get_env_path()
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv(AppConfig.nvidia_env_key)
    print(f"NVIDIA API Key loaded: {'✓' if api_key else '✗'}")
    if not api_key:
        print(f"⚠️  {AppConfig.nvidia_env_key} not found in .env")
    return api_key