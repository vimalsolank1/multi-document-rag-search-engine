from typing import Generator, Optional, List
import streamlit as st

from core.ingestion import DocumentProcessor
from core.vector_store import VectorStoreManager
from core.chain import RAGChain
from tools.tavily_search import TavilySearchTool
from ui.components import save_uploaded_file


class ChatInterface:
    """
    Main controller that connects the Streamlit UI with the RAG backend.

    Handles document processing, retrieval, and answer generation
    for all three modes: document, web, and hybrid.
    """

    def __init__(self):
        """Initialize all backend components needed for the chatbot."""

        self.doc_processor = DocumentProcessor()    # loads and chunks uploaded files
        self.vector_store = VectorStoreManager()    # stores and searches document embeddings
        self.rag_chain: Optional[RAGChain] = None   # created after documents are uploaded
        self.tavily = TavilySearchTool()            # handles live web search

    # ---------------------------------------------------
    # DOCUMENT PROCESSING
    # ---------------------------------------------------

    def process_uploaded_files(self, uploaded_files) -> int:
        """
        Save, chunk, and index all uploaded files into FAISS.

        Called when user clicks "Process Documents" button.

        Args:
            uploaded_files: List of Streamlit UploadedFile objects

        Returns:
            Total number of chunks indexed across all files
        """
        all_chunks = []

        for uploaded_file in uploaded_files:

            # Save file to temp directory and get its path
            file_path = save_uploaded_file(uploaded_file)
            documents = self.doc_processor.process(file_path)

            # Tag each chunk with file name for citation tracking
            for idx, doc in enumerate(documents):
                doc.metadata.update({
                    "source": uploaded_file.name,
                    "chunk_id": idx,
                    "source_type": "doc"
                })

            all_chunks.extend(documents)

            # Track uploaded file names in session to show in sidebar
            if uploaded_file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(uploaded_file.name)

        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            st.session_state.vector_store_initialized = True

        return len(all_chunks)

    # ---------------------------------------------------
    # RAG INITIALIZATION
    # ---------------------------------------------------

    def initialize_rag_chain(self):
        """
        Create the RAG chain once documents are ready.

        We delay creation until documents are indexed because
        RAGChain needs the vector store to be initialized.
        """
        if self.vector_store.is_initialized and self.rag_chain is None:
            self.rag_chain = RAGChain(self.vector_store)

    # ---------------------------------------------------
    # DOCUMENT RETRIEVAL WITH FILTERING
    # ---------------------------------------------------

    def retrieve_documents(self, query: str, threshold: float = 1.5):
        """
        Fetch relevant document chunks and filter out weak matches.

        FAISS returns L2 distance scores — lower score = more similar.
        Threshold 1.5 keeps relevant chunks and removes unrelated ones.

        Args:
            query: User's question
            threshold: Maximum allowed distance score (default 1.5)

        Returns:
            List of relevant Document chunks that passed the filter
        """
        if not self.vector_store.is_initialized:
            return []

        docs_with_scores = self.vector_store.search_with_scores(query)

        # Keep only chunks with score below threshold (good matches)
        filtered_docs = [
            doc for doc, score in docs_with_scores
            if score < threshold
        ]

        return filtered_docs

    # ---------------------------------------------------
    # MAIN RESPONSE GENERATOR
    # ---------------------------------------------------

    def get_response(
        self,
        query: str,
        retrieval_mode: str
    ) -> Generator[str, None, None]:
        """
        Generate answer based on selected retrieval mode.

        Yields tokens one by one for streaming display in Streamlit.

        Args:
            query: User's question
            retrieval_mode: One of "doc", "web", or "hybrid"

        Yields:
            Text tokens from LLM response
        """
        if self.rag_chain is None and self.vector_store.is_initialized:
            self.initialize_rag_chain()

        # Reset metadata for each new query
        st.session_state.last_answer_meta = {
            "answer_type": retrieval_mode,
            "doc_chunks": [],
            "web_docs": [],
            "doc_summaries": []
        }

        # =================================================
        # DOCUMENT MODE - answer only from uploaded files
        # =================================================

        if retrieval_mode == "doc":

            if not self.vector_store.is_initialized:
                yield "❗ Please process documents first."
                return

            docs = self.retrieve_documents(query)

            if not docs:
                yield "No relevant information found in the documents."
                return

            st.session_state.last_answer_meta["doc_chunks"] = docs

            # Generate short summaries for evidence display in UI
            summaries = self.rag_chain.summarize_documents(docs, top_n=3)
            st.session_state.last_answer_meta["doc_summaries"] = summaries

            # Use filtered docs directly to avoid double retrieval
            context = self.rag_chain._build_context(docs)

            for token in self.rag_chain.chain.stream({
                "context": context,
                "question": query
            }):
                yield token

            return

        # =================================================
        # WEB MODE - answer only from live Tavily search
        # =================================================

        if retrieval_mode == "web":

            # RAG chain needed for LLM access even without documents
            if self.rag_chain is None:
                from core.chain import RAGChain
                self.rag_chain = RAGChain(self.vector_store)

            web_docs = self.tavily.as_documents(query)

            if not web_docs:
                yield "No useful web results were found."
                return

            st.session_state.last_answer_meta["web_docs"] = web_docs

            # Format web results into readable context for LLM
            context = "\n\n".join([
                f"Source: {w.metadata.get('title', 'Unknown')}\n"
                f"URL: {w.metadata.get('source')}\n\n"
                f"{w.page_content}"
                for w in web_docs[:5]
            ])

            prompt = f"""
You are an AI assistant answering questions using web search results.

Use the information below to answer clearly.

Web Results:
{context}

Question:
{query}

Answer:
"""
            answer = self.rag_chain.llm.invoke(prompt).content
            yield answer
            return

        # =================================================
        # HYBRID MODE - combine documents + web search
        # =================================================

        if retrieval_mode == "hybrid":

            # RAG chain needed for LLM access even without documents
            if self.rag_chain is None:
                from core.chain import RAGChain
                self.rag_chain = RAGChain(self.vector_store)

            docs = self.retrieve_documents(query)
            web_docs = self.tavily.as_documents(query)

            st.session_state.last_answer_meta["doc_chunks"] = docs
            st.session_state.last_answer_meta["web_docs"] = web_docs

            # Combine document chunks and web results into one context
            context_parts = []

            for d in docs[:3]:
                context_parts.append(
                    f"Document Source: {d.metadata.get('source')}\n\n"
                    f"{d.page_content}"
                )

            for w in web_docs[:3]:
                context_parts.append(
                    f"Web Source: {w.metadata.get('title', 'Unknown')}\n"
                    f"URL: {w.metadata.get('source')}\n\n"
                    f"{w.page_content}"
                )

            context = "\n\n".join(context_parts)

            prompt = f"""
You are an AI assistant answering questions using both
documents and web search results.

Context:
{context}

Question:
{query}

Answer:
"""
            answer = self.rag_chain.llm.invoke(prompt).content
            yield answer

    # ---------------------------------------------------
    # SOURCE DISPLAY
    # ---------------------------------------------------

    def get_sources(self, query: str, retrieval_mode: str) -> List[str]:
        """
        Collect unique source labels used to answer the query.

        Shown below each assistant message in the chat UI.

        Args:
            query: User's question
            retrieval_mode: One of "doc", "web", or "hybrid"

        Returns:
            List of source label strings like "[Doc] file.pdf"
        """
        sources = set()

        if retrieval_mode in ("doc", "hybrid") and self.vector_store.is_initialized:
            docs = self.retrieve_documents(query)
            for d in docs:
                sources.add(f"[Doc] {d.metadata.get('source')}")

        if retrieval_mode in ("web", "hybrid"):
            sources.add("[Web] Tavily Search")

        return list(sources)