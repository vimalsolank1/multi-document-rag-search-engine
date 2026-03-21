from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings


class EmbeddingManager:
    """
    Manages text embeddings using HuggingFace sentence-transformer models.

    Uses sentence-transformers which are FREE and run locally!
    No API costs for embeddings - everything runs on your machine.

    The model is loaded only once and shared across all instances
    to avoid reloading it every time a new object is created (singleton pattern).
    """

    # Class-level variable - shared across all instances
    _embedding_model: HuggingFaceEmbeddings | None = None

    def __init__(self, model_name: str = None):
        """
        Initialize the embedding manager.

        If model is already loaded by a previous instance, reuse it.
        Otherwise download and load it for the first time.

        Args:
            model_name: HuggingFace model name (default from .env settings)
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL

        # Load only once - subsequent instances skip this block
        if EmbeddingManager._embedding_model is None:
            EmbeddingManager._embedding_model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},  # change to "cuda" if GPU is available
                encode_kwargs={"normalize_embeddings": True}  # normalize for better similarity scores
            )

    @property
    def model(self) -> HuggingFaceEmbeddings:
        """
        Get the loaded embedding model instance.

        Returns:
            HuggingFaceEmbeddings model ready for use
        """
        return self._embedding_model

    def embed_text(self, text: str) -> List[float]:
        """
        Create an embedding vector for a single text string.

        Args:
            text: The input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        return self.model.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Create embedding vectors for multiple texts at once.

        Batching is faster than embedding one by one.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors, one per input text
        """
        if not texts:
            raise ValueError("Input text list cannot be empty")

        return self.model.embed_documents(texts)

    def get_embedding_dimension(self) -> int:
        """
        Get the size of the embedding vector this model produces.

        Useful when setting up FAISS index which needs to know vector size upfront.

        Returns:
            Integer representing the number of dimensions in each vector
        """
        # Embed a dummy text just to check the output size
        sample_vector = self.embed_text("dimension_probe")
        return len(sample_vector)