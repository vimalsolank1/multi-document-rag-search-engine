import os
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config.settings import settings
from core.embedding import EmbeddingManager


class VectorStoreManager:
    """
    Manages the FAISS vector database for semantic document search.

    FAISS (Facebook AI Similarity Search) stores document embeddings
    and lets us find the most relevant chunks for any query - very fast!

    Responsibilities:
    - Build vector index from uploaded documents
    - Add new documents to existing index
    - Search for similar documents by query
    - Save and load index from disk
    """

    def __init__(self, embedding_manager: EmbeddingManager = None):
        """
        Initialize the vector store manager.

        Args:
            embedding_manager: Custom embedding manager (default creates a new one)
        """
        # Use provided embedding manager or create a default one
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self._vector_store: Optional[FAISS] = None

        # Where to save/load the FAISS index on disk
        self.index_path = settings.FAISS_INDEX_PATH

    @property
    def vector_store(self) -> Optional[FAISS]:
        """
        Get the current FAISS vector store instance.

        Returns:
            FAISS instance or None if not initialized yet
        """
        return self._vector_store

    @property
    def is_initialized(self) -> bool:
        """
        Check if the vector store has been built and is ready to search.

        Returns:
            True if documents have been indexed, False otherwise
        """
        return self._vector_store is not None

    def _filter_documents(self, documents: List[Document]) -> List[Document]:
        """
        Remove empty documents before indexing.

        Empty documents cause FAISS errors, so we filter them out early.

        Args:
            documents: Raw list of documents to filter

        Returns:
            List of documents with non-empty content only
        """
        return [doc for doc in documents if doc.page_content.strip()]

    def create_from_documents(self, documents: List[Document]) -> FAISS:
        """
        Build a brand new FAISS index from a list of documents.

        Called when the first batch of documents is uploaded.

        Args:
            documents: List of document chunks to index

        Returns:
            Newly created FAISS vector store
        """
        documents = self._filter_documents(documents)

        if not documents:
            raise ValueError("No valid documents provided for indexing.")

        # FAISS embeds all documents and builds the index in one shot
        self._vector_store = FAISS.from_documents(
            documents,
            self.embedding_manager.model
        )

        return self._vector_store

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to the existing index.

        If no index exists yet, creates one automatically.

        Args:
            documents: New document chunks to add
        """
        documents = self._filter_documents(documents)

        if not documents:
            return

        if not self.is_initialized:
            # First upload - build index from scratch
            self.create_from_documents(documents)
        else:
            # Index already exists - just add to it
            self._vector_store.add_documents(documents)

    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Find the most similar documents for a given query.

        Args:
            query: User's question or search text
            k: Number of top results to return (default from .env)

        Returns:
            List of most relevant document chunks
        """
        if not self.is_initialized:
            raise ValueError("Vector store is not initialized. Add documents first.")

        k = k or settings.TOP_K_RESULTS

        return self._vector_store.similarity_search(query, k=k)

    def search_with_scores(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        Same as search() but also returns similarity scores.

        Score is L2 distance - lower means more similar.
        Used in chat.py to filter out irrelevant results by threshold.

        Args:
            query: User's question or search text
            k: Number of top results to return (default from .env)

        Returns:
            List of (document, score) tuples
        """
        if not self.is_initialized:
            raise ValueError("Vector store is not initialized. Add documents first.")

        k = k or settings.TOP_K_RESULTS

        return self._vector_store.similarity_search_with_score(query, k=k)

    def save(self, path: str = None) -> None:
        """
        Save the FAISS index to disk so it can be reloaded later.

        Args:
            path: Custom save path (default from .env settings)
        """
        if not self.is_initialized:
            raise ValueError("Vector store is not initialized. Nothing to save.")

        save_path = path or self.index_path

        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        self._vector_store.save_local(save_path)

    def load(self, path: str = None) -> FAISS:
        """
        Load a previously saved FAISS index from disk.

        Args:
            path: Custom load path (default from .env settings)

        Returns:
            Loaded FAISS vector store
        """
        load_path = path or self.index_path

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No FAISS index found at {load_path}")

        self._vector_store = FAISS.load_local(
            load_path,
            self.embedding_manager.model,
            allow_dangerous_deserialization=True  # required by LangChain for local files
        )

        return self._vector_store

    def get_retriever(self, k: int = None):
        """
        Get a LangChain retriever object from the vector store.

        Alternative to search() - useful when plugging directly
        into LangChain chains or agents.

        Args:
            k: Number of results to retrieve (default from .env)

        Returns:
            LangChain VectorStoreRetriever object
        """
        if not self.is_initialized:
            raise ValueError("Vector store is not initialized.")

        k = k or settings.TOP_K_RESULTS

        return self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def clear(self) -> None:
        """
        Remove the vector store from memory.

        Does NOT delete anything from disk - only clears the in-memory index.
        """
        self._vector_store = None