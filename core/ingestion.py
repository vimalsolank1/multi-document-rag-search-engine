from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings


class DocumentProcessor:

    """
    Loads PDF/TXT files, cleans them, and splits into chunks
    ready for embedding and vector search.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the processor with chunking configuration.

        If no values are provided, the system uses defaults from .env settings.
        Args:
        chunk_size: Maximum characters per chunk (default from .env)
            chunk_overlap: Shared characters between chunks (default from .env)
        """

        # Fall back to .env values if not passed directly
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # Splits text smartly — tries paragraph breaks first, then lines, then words
        # This ensures chunks break at natural boundaries, not mid-sentence
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_document(self, file_path: str) -> List[Document]:
        """
        Reads a PDF or TXT file from disk and converts to LangChain Documents.

        For PDFs, returns one Document per page.
        For TXT files, returns one Document for the entire file.

        Args:
        file_path: Full path to the file on disk

        Returns:
            List of LangChain Document objects with raw text
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        # Choose the right loader based on file extension
        if extension == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif extension == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}. Use .txt or .pdf")

        return loader.load()

    def load_from_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Wraps a raw string into a LangChain Document object.

        Used when we already have text in memory (e.g. Tavily web search results)
        and don't need to read from a file.

        Args:
            text: Raw text content to wrap
            metadata: Optional dict with source info like title, URL

        Returns:
            List containing a single Document object
        """
        metadata = metadata or {}
        return [Document(page_content=text, metadata=metadata)]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits large documents into smaller chunks suitable for embedding.

        Each chunk gets a chunk_id in metadata so we can track
        exactly which chunk was used to answer a query.

        Args:
            documents: List of full documents to split

        Returns:
            List of smaller Document chunks ready for embedding
        """
        chunks = self.text_splitter.split_documents(documents)

        # Assign unique ID to each chunk for citation tracking
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx

        return chunks

    def process(self, file_path: str) -> List[Document]:
        """
        Full ingestion pipeline for a file: load → tag metadata → split into chunks.

        This is the main method called when user uploads a document.
        Output chunks are ready to be embedded and stored in FAISS.

        Args:
            file_path: Full path to the uploaded file on disk

        Returns:
            List of tagged and chunked Document objects
        """
        documents = self.load_document(file_path)

        # Tag each doc with filename and type before splitting
        # This ensures every chunk knows which file it came from (used for citations)
        for doc in documents:
            doc.metadata.update({
                "source": Path(file_path).name,      
                "source_type": Path(file_path).suffix.replace(".", "")  # e.g. "pdf"
            })

        return self.split_documents(documents)

    def process_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Same as process() but accepts raw text instead of a file path.

        Used for processing Tavily web search results which come as
        plain text strings, not files.

        Args:
            text: Raw text content to process
            metadata: Optional source info like title, URL

        Returns:
            List of chunked Document objects ready for embedding
        """
        documents = self.load_from_text(text, metadata)
        return self.split_documents(documents)