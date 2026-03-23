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
        # Fall back to .env values if not passed directly
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # Splits text smartly   tries paragraph breaks first, then lines, then words
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_document(self, file_path: str) -> List[Document]:
        """
        Reads a PDF or TXT file from disk.
        Returns a list of LangChain Document objects (one per page for PDF).
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif extension == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}. Use .txt or .pdf")

        return loader.load()

    def load_from_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Wraps a raw string into a Document object.
        Useful for web search results or API responses.
        """
        metadata = metadata or {}
        return [Document(page_content=text, metadata=metadata)]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits large documents into smaller chunks for embedding.
        Each chunk gets a chunk_id for tracking which chunk answered a query.
        """
        chunks = self.text_splitter.split_documents(documents)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx

        return chunks

    def process(self, file_path: str) -> List[Document]:
        """
        Full pipeline for a file: load → tag metadata → split into chunks.
        This is the main method called when user uploads a document.
        """
        documents = self.load_document(file_path)

        # Tag each doc with filename and type so we can show citations later
        for doc in documents:
            doc.metadata.update({
                "source": Path(file_path).name,
                "source_type": Path(file_path).suffix.replace(".", "")
            })

        return self.split_documents(documents)

    def process_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Same as process() but for raw text instead of a file.
        Used for Tavily web search results.
        """
        documents = self.load_from_text(text, metadata)
        return self.split_documents(documents)