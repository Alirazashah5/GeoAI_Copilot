from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "GeoAI Copilot"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str | None = None
    cors_origins: str = "http://localhost:8501"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
