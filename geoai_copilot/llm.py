from .config import settings


def explain(prompt: str) -> str:
    """Optional OpenAI-compatible explanation layer.

    The deterministic geoscience calculations should remain independent of this
    function. If no LLM credentials are configured, return a transparent message.
    """
    if not settings.openai_api_key or not settings.openai_model:
        return (
            "LLM explanation is not configured. Configure OPENAI_API_KEY and "
            "OPENAI_MODEL in .env to enable this optional layer."
        )

    # Kept deliberately dependency-light. The API integration can be added without
    # coupling core petrophysics/ML calculations to a provider SDK.
    return (
        "LLM provider is configured, but the provider call is intentionally not "
        "enabled in the initial scaffold. Use the deterministic outputs as the "
        "source of truth and add a provider adapter under geoai_copilot/providers/."
    )
