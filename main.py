import streamlit as st

from ui.chat import ChatInterface
from ui.components import (
    init_session_state,
    display_chat_history,
    display_sidebar_info,
    display_file_uploader,
    add_message,
    display_answer_metadata,
    retrieval_mode_selector,
    process_documents_button,
)

st.set_page_config(
    page_title="Multi Document RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


def main():
    """
    Entry point for the Streamlit app.

    Streamlit reruns this entire function on every user interaction,
    so session state is used to persist data between reruns.
    """

    # Set up session state defaults on first load
    init_session_state()

    # Render sidebar with app info and file list
    display_sidebar_info()

    # Create ChatInterface once and store in session state
    # so it's not recreated on every rerun
    if "chat_interface" not in st.session_state:
        st.session_state.chat_interface = ChatInterface()

    chat_interface: ChatInterface = st.session_state.chat_interface

    # ---------------------------------------
    # Header
    # ---------------------------------------

    st.title("🤖 Multi Document RAG Chatbot")
    st.caption(
        "Ask questions using your **documents**, **live web search**, or **hybrid retrieval**."
    )
    st.divider()

    # ---------------------------------------
    # Upload + Retrieval Mode
    # ---------------------------------------

    # Two columns: file uploader on left, retrieval mode selector on right
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = display_file_uploader()

    with col2:
        retrieval_mode = retrieval_mode_selector()

    process_clicked = process_documents_button()

    if process_clicked:

        if not uploaded_files:
            st.warning("Please upload at least one document.")

        else:
            with st.spinner("Processing documents..."):
                chunk_count = chat_interface.process_uploaded_files(uploaded_files)

            st.success(f"Indexed {chunk_count} chunks")

    st.divider()

    # ---------------------------------------
    # Chat Area
    # ---------------------------------------

    # Show all previous messages from session state
    display_chat_history()

    user_query = st.chat_input("Ask your question...")

    if user_query:

        # Save and show user message immediately
        add_message("user", user_query)

        with st.chat_message("user"):
            st.markdown(user_query)

        # Stream assistant response token by token
        with st.chat_message("assistant"):

            placeholder = st.empty()
            full_response = ""

            for token in chat_interface.get_response(
                query=user_query,
                retrieval_mode=retrieval_mode
            ):
                full_response += token
                placeholder.markdown(full_response)  # update UI on each token

        # Save completed response with sources to chat history
        add_message(
            "assistant",
            full_response,
            chat_interface.get_sources(user_query, retrieval_mode)
        )

        # Show answer type and sources below the response
        display_answer_metadata()


if __name__ == "__main__":
    main()