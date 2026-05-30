"""RAG Retriever for orchestrating document ingestion and retrieval."""

from typing import BinaryIO, Optional
from .document_loader import load_document
from .text_splitter import split_text
from .embeddings import GeminiEmbeddings
from .vector_store import VectorStore


class RAGRetriever:
    """Orchestrates the RAG pipeline for document ingestion and retrieval."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        embedding_model: str = "text-embedding-004",
        use_local_embeddings: bool = False,
        persist_directory: str = "./chroma_db",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """Initialize the RAG retriever.

        Args:
            api_key: API key for embeddings.
            base_url: Optional base URL for OpenAI-compatible endpoint.
            embedding_model: Embedding model to use.
            use_local_embeddings: If True, use local sentence-transformers.
            persist_directory: Directory to persist the vector store.
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.
        """
        self.embeddings = GeminiEmbeddings(
            api_key=api_key,
            base_url=base_url,
            model=embedding_model,
            use_local=use_local_embeddings
        )
        self.vector_store = VectorStore(persist_directory=persist_directory)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_file(self, file: BinaryIO, filename: str) -> int:
        """Ingest a document file into the vector store.

        Args:
            file: File-like object containing the document.
            filename: Name of the file.

        Returns:
            Number of chunks created and stored.
        """
        # Load document
        text = load_document(file, filename)

        # Split into chunks
        chunks = split_text(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        if not chunks:
            return 0

        # Create embeddings
        embeddings = self.embeddings.embed_texts(chunks)

        # Create metadata for each chunk
        metadatas = [
            {"source": filename, "chunk_index": i}
            for i in range(len(chunks))
        ]

        # Store in vector store
        self.vector_store.add_documents(
            texts=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return len(chunks)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Retrieve relevant document chunks for a query.

        Args:
            query: The user's query.
            k: Number of chunks to retrieve.

        Returns:
            List of relevant chunks with metadata.
        """
        # Embed the query
        query_embedding = self.embeddings.embed_query(query)

        # Search vector store
        results = self.vector_store.search(query_embedding, k=k)

        return results

    def get_context(self, query: str, k: int = 3) -> str:
        """Get formatted context string for a query.

        Args:
            query: The user's query.
            k: Number of chunks to retrieve.

        Returns:
            Formatted context string with source attribution.
        """
        results = self.retrieve(query, k=k)

        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "Unknown")
            context_parts.append(
                f"[Source: {source}]\n{result['document']}"
            )

        return "\n\n---\n\n".join(context_parts)

    def build_augmented_prompt(self, query: str, k: int = 3) -> str:
        """Build a prompt augmented with retrieved context.

        Args:
            query: The user's query.
            k: Number of chunks to retrieve.

        Returns:
            Augmented prompt with context and instructions.
        """
        context = self.get_context(query, k=k)

        if not context:
            return query

        augmented_prompt = f"""Use the following context to answer the user's question. If the answer cannot be found in the context, say so clearly.

CONTEXT:
{context}

USER QUESTION:
{query}

Please provide a helpful answer based on the context above. Cite the source when using information from the documents."""

        return augmented_prompt

    def clear_documents(self) -> None:
        """Clear all documents from the vector store."""
        self.vector_store.clear()

    def get_document_count(self) -> int:
        """Get the number of document chunks in the store."""
        return self.vector_store.get_document_count()

    def get_sources(self) -> list[str]:
        """Get list of all ingested source files."""
        return self.vector_store.get_all_sources()
