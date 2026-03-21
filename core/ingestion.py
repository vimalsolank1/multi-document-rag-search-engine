from typing import List
from pathlib import Path

# LangChain document structure used throughout the RAG pipeline
from langchain_core.documents import Document

# Loaders for different document types
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Text splitter used to break large documents into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Centralized project settings
from config.settings import settings


class DocumentProcessor:
    """
    Handles document ingestion and preprocessing.

    Responsibilities:
    1. Load documents from different file formats (PDF, TXT)
    2. Convert raw text into LangChain Document objects
    3. Split documents into smaller chunks for embedding and retrieval
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize the processor with chunking configuration.

        If no values are provided, the system uses defaults from settings.
        """

        # Use provided values or fallback to environment configuration
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # Recursive splitter intelligently breaks text into overlapping chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,

            # Priority order for splitting text
            separators=["\n\n", "\n", " ", ""]
        )

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document from disk and convert it into LangChain Documents.

        Supported formats:
        - .txt
        - .pdf
        """

        path = Path(file_path)
        extension = path.suffix.lower()

        # Choose loader based on file extension
        if extension == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")

        elif extension == ".pdf":
            loader = PyPDFLoader(file_path)

        else:
            raise ValueError(
                f"Unsupported file type: {extension}. Use .txt or .pdf"
            )

        return loader.load()

    def load_from_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Convert raw text into a LangChain Document.

        This is useful when ingesting data from APIs,
        web scraping, or other dynamic sources.
        """

        metadata = metadata or {}

        return [
            Document(
                page_content=text,
                metadata=metadata
            )
        ]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks suitable for embeddings.

        Each chunk receives a unique chunk_id for traceability.
        """

        chunks = self.text_splitter.split_documents(documents)

        # Attach chunk identifiers for debugging and citation tracking
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx

        return chunks

    def process(self, file_path: str) -> List[Document]:
        """
        Full ingestion pipeline for file-based documents.

        Steps:
        1. Load document
        2. Attach metadata
        3. Split into chunks
        """

        documents = self.load_document(file_path)

        # Add metadata for source tracking and citation
        for doc in documents:
            doc.metadata.update({
                "source": Path(file_path).name,
                "source_type": Path(file_path).suffix.replace(".", "")
            })

        return self.split_documents(documents)

    def process_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Process raw text directly (no file needed).

        Useful for:
        - Web search results
        - API responses
        - Generated text
        """

        documents = self.load_from_text(text, metadata)

        return self.split_documents(documents)