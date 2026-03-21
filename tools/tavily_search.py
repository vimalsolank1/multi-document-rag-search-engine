from typing import List
from langchain_tavily import TavilySearch
from langchain_core.documents import Document

from config.settings import settings


class TavilySearchTool:
    """
    Provides real-time web search using Tavily.

    The results are converted into LangChain Document objects
    so they can be used in the same pipeline as local documents.
    """

    def __init__(self, max_results: int = None):
        """
        Initialize Tavily search client.
        """

        self.max_results = max_results or settings.TOP_K_WEB_RESULTS

        # Tavily search wrapper from LangChain
        self.search = TavilySearch(
            tavily_api_key=settings.TAVILY_API_KEY,
            max_results=self.max_results
        )

    def as_documents(self, query: str) -> List[Document]:
        """
        Perform web search and convert results into Document objects.
        """

        response = self.search.invoke(query)

        docs: List[Document] = []

        for result in response.get("results", []):

            docs.append(
                Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "source_type": "web",
                        "title": result.get("title", "Unknown"),
                        "source": result.get("url")
                    }
                )
            )

        return docs

    def format_results(self, documents: List[Document]) -> str:
        """
        Convert Tavily results into readable context text
        for LLM prompts.
        """

        if not documents:
            return "No web results found."

        formatted = []

        for i, doc in enumerate(documents, 1):

            title = doc.metadata.get("title", "Unknown")
            url = doc.metadata.get("source", "")

            formatted.append(
                f"[Web Result {i}] {title}\n{doc.page_content}\nSource: {url}"
            )

        return "\n\n".join(formatted)