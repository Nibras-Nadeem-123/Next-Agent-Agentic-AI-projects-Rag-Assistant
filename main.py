"""RAG Assistant - A Streamlit app for document Q&A using retrieval-augmented generation."""

import streamlit as st
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from dotenv import load_dotenv
import os
import asyncio

from rag import RAGRetriever

# Load environment variables
load_dotenv()
set_tracing_disabled(True)

# Page configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide"
)


def get_retriever() -> RAGRetriever:
    """Get or create the RAG retriever instance."""
    if "retriever" not in st.session_state:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY not found in environment variables!")
            st.stop()
        st.session_state.retriever = RAGRetriever(api_key=api_key)
    return st.session_state.retriever


def get_agent() -> Agent:
    """Get or create the AI agent instance."""
    if "agent" not in st.session_state:
        api_key = os.getenv("GEMINI_API_KEY")
        base_url = os.getenv("GEMINI_API_BASE_URL")

        if not api_key or not base_url:
            st.error("GEMINI_API_KEY or GEMINI_API_BASE_URL not found!")
            st.stop()

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        model = OpenAIChatCompletionsModel(openai_client=client, model="gemini-2.5-flash")

        st.session_state.agent = Agent(
            name="RAG Assistant",
            instructions="""You are a helpful assistant that answers questions based on the provided context from uploaded documents.

When answering:
1. Use information from the provided context to answer questions
2. If the context doesn't contain relevant information, clearly state that
3. Cite the source documents when using specific information
4. Be concise but thorough in your responses
5. If asked about something not in the documents, you can provide general knowledge but clarify it's not from the uploaded documents""",
            model=model,
        )
    return st.session_state.agent


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()


def process_uploaded_file(uploaded_file):
    """Process and ingest an uploaded file."""
    retriever = get_retriever()

    if uploaded_file.name in st.session_state.processed_files:
        return False, "File already processed"

    try:
        # Reset file position
        uploaded_file.seek(0)

        # Ingest the file
        num_chunks = retriever.ingest_file(uploaded_file, uploaded_file.name)

        # Mark as processed
        st.session_state.processed_files.add(uploaded_file.name)

        return True, f"Processed '{uploaded_file.name}' into {num_chunks} chunks"
    except Exception as e:
        return False, f"Error processing file: {str(e)}"


def get_response(user_query: str) -> str:
    """Get a response from the agent with RAG context."""
    retriever = get_retriever()
    agent = get_agent()

    # Build augmented prompt with context
    if retriever.get_document_count() > 0:
        augmented_prompt = retriever.build_augmented_prompt(user_query)
    else:
        augmented_prompt = user_query

    # Run the agent
    result = Runner.run_sync(agent, augmented_prompt)

    # Extract the response text
    if hasattr(result, 'final_output'):
        return result.final_output
    return str(result)


# Main app
def main():
    initialize_session_state()

    # Header
    st.title("📚 RAG Assistant")
    st.markdown("Upload documents and ask questions about their content.")

    # Sidebar for document management
    with st.sidebar:
        st.header("📄 Document Management")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload PDF or TXT files to add to the knowledge base"
        )

        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.processed_files:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        success, message = process_uploaded_file(uploaded_file)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)

        # Display document stats
        st.divider()
        st.subheader("📊 Knowledge Base Stats")

        retriever = get_retriever()
        doc_count = retriever.get_document_count()
        sources = retriever.get_sources()

        st.metric("Total Chunks", doc_count)

        if sources:
            st.write("**Uploaded Documents:**")
            for source in sources:
                st.write(f"• {source}")
        else:
            st.info("No documents uploaded yet")

        # Clear documents button
        st.divider()
        if st.button("🗑️ Clear All Documents", type="secondary"):
            retriever.clear_documents()
            st.session_state.processed_files = set()
            st.session_state.messages = []
            st.rerun()

    # Main chat area
    col1, col2 = st.columns([3, 1])

    with col1:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_response(prompt)
                    st.markdown(response)

            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})

    with col2:
        st.subheader("💡 Tips")
        st.markdown("""
        **How to use:**
        1. Upload PDF or TXT files using the sidebar
        2. Wait for processing to complete
        3. Ask questions about your documents

        **Example questions:**
        - "What is the main topic of this document?"
        - "Summarize the key points"
        - "What does the document say about X?"
        """)

        # Quick actions
        if st.session_state.messages:
            if st.button("🔄 Clear Chat History"):
                st.session_state.messages = []
                st.rerun()


if __name__ == "__main__":
    main()
