from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    vector_timeout_seconds: float = 0.20
    max_retries: int = 2
    model_name: str = "fake-rag-model"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
