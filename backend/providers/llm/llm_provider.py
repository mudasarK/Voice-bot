from backend.core.config import settings
from backend.core.interfaces import LLMProvider

def get_llm_provider() -> LLMProvider:
    provider = settings.LLM_PROVIDER.lower()
    if provider == "openrouter":
        from backend.providers.llm.openrouter import OpenRouterLLMProvider
        return OpenRouterLLMProvider(model_name=settings.LLM_MODEL)
    else:
        raise ValueError(f"Unknown LLM Provider: {provider}")
