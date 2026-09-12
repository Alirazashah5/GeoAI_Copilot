from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import settings

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="AI-assisted geoscience and petrophysics API.",
)


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = settings.app_name


class ExplainRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=5000)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/explain")
def explain(request: ExplainRequest) -> dict:
    from .llm import explain as llm_explain
    return {"answer": llm_explain(request.prompt)}
