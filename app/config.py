from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_keys: list[str]
    models_dir: str = "/app/models"
    log_level: str = "INFO"

    @field_validator("api_keys", mode="before")
    @classmethod
    def split_keys(cls, v):
        return [k.strip() for k in v.split(",")] if isinstance(v, str) else v

    model_config = {"env_file": ".env"}


settings = Settings()
