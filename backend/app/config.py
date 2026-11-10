# Global constants, fraud thresholds, Gemini API config

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash-preview-05-20"

    FRAUD_HARD_REJECT_THRESHOLD: float = 0.8
    FRAUD_MANUAL_REVIEW_THRESHOLD: float = 0.4

    PHASH_HAMMING_DISTANCE_LIMIT: int = 4
    MAX_TIMESTAMP_DELTA_SECONDS: int = 7200  # 2 hours

    DELIVERY_FEE: float = 49.0
    PLATFORM_FEE: float = 5.0


settings = Settings()
