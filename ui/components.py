import streamlit as st
from typing import List
import tempfile
import os


# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

def init_session_state():
    """
    Set up default session state variables on first app load.

    Streamlit reruns the script on every interaction,
    so session state is how we persist data between reruns.
    """
    defaults = {
        "messages": [],                      # chat history
        "vector_store_initialized": False,   # tracks if documents are indexed
        "uploaded_files": [],                # list of uploaded file names
        "temp_dir": tempfile.mkdtemp(),      # temp folder for saving uploaded files
        "last_answer_meta": None             # metadata from last response (sources, type)
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------

def display_chat_history():
    """
    Render all previous messages from session state in the chat UI.

    Also shows source references under each assistant message.
    """
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if this message has them
            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.write(f"- {source}")


def add_message(role: str, content: str, sources: List[str] = None):
    """
    Append a message to the chat history in session state.

    Args:
        role: "user" or "assistant"
        content: The message text to display
        sources: Optional list of source references for the answer
    """
    message = {"role": role, "content": content}

    if sources:
        message["sources"] = sources

    st.session_state.messages.append(message)


def clear_chat_history():
    """Wipe all messages from chat history."""
    st.session_state.messages = []


# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

def display_file_uploader():
    """
    Render the file upload widget.

    Returns:
        List of uploaded files (Streamlit UploadedFile objects)
    """
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload documents to ask questions from them"
    )

    return uploaded_files


def process_documents_button() -> bool:
    """
    Render the Process Documents button.

    Returns:
        True if button was clicked, False otherwise
    """
    return st.button("⚡ Process Documents")


# ---------------------------------------------------
# RETRIEVAL MODE
# ---------------------------------------------------

def retrieval_mode_selector() -> str:
    """
    Render radio buttons for selecting retrieval strategy.

    Returns:
        Selected mode string: "doc", "web", or "hybrid"
    """
    return st.radio(
        "Retrieval Mode",
        options=["doc", "web", "hybrid"],
        format_func=lambda x: {
            "doc": "📄 Documents",
            "web": "🌐 Web",
            "hybrid": "🔀 Hybrid"
        }[x],
        index=0
    )


# ---------------------------------------------------
# FILE SAVING
# ---------------------------------------------------

def save_uploaded_file(uploaded_file) -> str:
    """
    Write an uploaded file to the temp directory on disk.

    Streamlit uploaded files are in memory — we save them
    to disk so LangChain loaders can read them by file path.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Full file path where the file was saved
    """
    temp_dir = st.session_state.temp_dir
    file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

def display_sidebar_info():
    """
    Render the sidebar with app info and uploaded file list.

    Also provides the Clear Chat History button.
    """
    with st.sidebar:

        st.header("📚 Multi Document RAG Chatbot")

        st.markdown("""
This chatbot can:

- 📄 Answer questions from your documents  
- 🌐 Search the web using Tavily  
- 🔀 Combine documents and web search  

### How to use

1️⃣ Upload PDF or TXT files  
2️⃣ Click **Process Documents**  
3️⃣ Ask questions
""")

        st.divider()

        st.subheader("Uploaded Files")

        # Show each uploaded file with a checkmark
        if st.session_state.uploaded_files:
            for file in st.session_state.uploaded_files:
                st.write(f"✅ {file}")
        else:
            st.caption("No files uploaded yet")

        st.divider()

        if st.button("🗑 Clear Chat History"):
            clear_chat_history()
            st.rerun()


# ---------------------------------------------------
# ANSWER METADATA
# ---------------------------------------------------

def display_answer_metadata():
    """
    Show answer type indicator and sources used for the last response.

    Displayed below the assistant's answer after each query.
    """
    meta = st.session_state.get("last_answer_meta")

    if not meta:
        return

    # Show which mode generated this answer
    indicator = {
        "doc": "📄 Answer from documents",
        "web": "🌐 Answer from web search",
        "hybrid": "🔀 Hybrid answer"
    }

    st.caption(indicator.get(meta["answer_type"], ""))

    with st.expander("📚 Sources"):

        if not meta.get("doc_chunks") and not meta.get("web_docs"):
            st.write("No sources available")
            return

        # Use set to avoid showing duplicate file names
        doc_sources = {
            doc.metadata.get("source")
            for doc in meta.get("doc_chunks", [])
        }

        for src in doc_sources:
            st.write(f"📄 {src}")

        # Use set to avoid showing duplicate web titles
        web_sources = {
            web.metadata.get("title", "Unknown")
            for web in meta.get("web_docs", [])
        }

        for src in web_sources:
            st.write(f"🌐 {src}")


# ---------------------------------------------------
# STATUS MESSAGES
# ---------------------------------------------------

def display_processing_status(message: str, status: str = "info"):
    """
    Show a status message in the UI.

    Args:
        message: Text to display
        status: One of "success", "warning", "error", or "info"
    """
    if status == "success":
        st.success(message)
    elif status == "warning":
        st.warning(message)
    elif status == "error":
        st.error(message)
    else:
        st.info(message)