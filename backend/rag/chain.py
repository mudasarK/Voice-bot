from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from backend.core.interfaces import LLMProvider, VectorStoreProvider


def _format_docs(docs: List[Document]) -> str:
    """Format retrieved documents as a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


class RAGChain:
    def __init__(self, llm_provider: LLMProvider, store_provider: VectorStoreProvider):
        self.llm = llm_provider.get_llm()
        store = store_provider.get_store()
        self.retriever = store.as_retriever(search_kwargs={"k": 3})

        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful Help Desk Assistant. Use the following context to answer the user's question accurately.
            If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
            Keep the answer conversational and concise, as it will be spoken back to the user via TTS.
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
        )
        
        # Format retriever output (list of docs) as a single context string
        self.chain = (
            {
                "context": self.retriever | RunnableLambda(_format_docs),
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
    def invoke(self, question: str) -> str:
        """Invokes the RAG chain with the given question."""
        return self.chain.invoke(question)
