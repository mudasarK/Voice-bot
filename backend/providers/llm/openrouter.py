from backend.core.interfaces import LLMProvider
from backend.core.config import settings
from langchain_openai import ChatOpenAI

class OpenRouterLLMProvider(LLMProvider):
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.LLM_MODEL
        # OpenRouter uses the OpenAI compatible API
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
        
    def get_llm(self) -> ChatOpenAI:
        return self.llm
