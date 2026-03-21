from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from config.settings import settings


# System prompt - tells LLM to answer ONLY from given context, not from its own knowledge
RAG_PROMPT = """
You are a helpful assistant.

Answer the user's question ONLY using the context provided below.
If the answer is not contained in the context, clearly say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""


class RAGChain:
    """
    Handles the full RAG (Retrieval-Augmented Generation) pipeline.

    Flow: User query → fetch relevant chunks from FAISS
          → build context → send to LLM → return answer

    Uses Groq API for fast LLM inference (FREE tier available).
    """

    def __init__(self, vector_store):
        """
        Initialize the RAG pipeline with LLM and prompt setup.

        Args:
            vector_store: VectorStoreManager instance for document retrieval
        """
        self.vector_store = vector_store

        # Groq gives us fast inference on open-source models like LLaMA
        self.llm = ChatGroq(
            model=settings.GPT_MODEL_NAME,
            temperature=settings.TEMPERATURE,  # 0 = consistent answers, no randomness
            api_key=settings.GROQ_API_KEY
        )

        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        self.parser = StrOutputParser()

        # LangChain pipe: format prompt → call LLM → parse output as plain string
        self.chain = self.prompt | self.llm | self.parser

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """
        Fetch the most relevant document chunks from FAISS for a query.

        Args:
            query: User's question
            k: Number of chunks to retrieve (default from .env)

        Returns:
            List of relevant Document chunks
        """
        if k is None:
            k = settings.TOP_K_RESULTS  # fallback to .env value

        return self.vector_store.search(query, k=k)

    def _build_context(self, documents: List[Document]) -> str:
        """
        Format retrieved document chunks into a single context string for the LLM.

        Each chunk is labeled with its source file so LLM can reference it.

        Args:
            documents: List of retrieved document chunks

        Returns:
            Formatted context string passed to the LLM prompt
        """
        if not documents:
            return "No relevant documents were found."

        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(
                f"[Document {i}] Source: {source}\n{doc.page_content}"
            )

        return "\n\n".join(context_parts)

    def generate(self, query: str, context: str) -> str:
        """
        Send context + question to LLM and get the answer.

        Args:
            query: User's question
            context: Formatted document context from _build_context()

        Returns:
            LLM generated answer as a string
        """
        return self.chain.invoke({
            "context": context,
            "question": query
        })

    def query(self, query: str, k: int = None) -> str:
        """
        Run the complete RAG pipeline in one call.

        retrieve → build context → generate answer

        Args:
            query: User's question
            k: Number of chunks to retrieve (default from .env)

        Returns:
            Final answer string from LLM
        """
        docs = self.retrieve(query, k=k)
        context = self._build_context(docs)
        return self.generate(query, context)

    def query_stream(self, query: str, k: int = None):
        """
        Streaming version of query() - yields answer tokens one by one.

        Used in Streamlit so the answer appears word by word
        instead of waiting for the full response.

        Args:
            query: User's question
            k: Number of chunks to retrieve (default from .env)

        Yields:
            Individual text tokens from LLM response
        """
        docs = self.retrieve(query, k=k)
        context = self._build_context(docs)

        for token in self.chain.stream({
            "context": context,
            "question": query
        }):
            yield token

    def summarize_documents(self, documents: List[Document], top_n: int = 3):
        """
        Generate a short summary for each of the top retrieved documents.

        Shown in the UI as document evidence alongside the main answer.

        Args:
            documents: List of retrieved document chunks
            top_n: How many documents to summarize (default 3)

        Returns:
            List of summary strings with source labels
        """
        summaries = []

        for i, doc in enumerate(documents[:top_n], 1):
            prompt = f"Summarize this text briefly:\n\n{doc.page_content}"
            summary = self.llm.invoke(prompt).content
            source = doc.metadata.get("source", "unknown")
            summaries.append(f"[Document {i}] Source: {source}\n{summary}")

        return summaries