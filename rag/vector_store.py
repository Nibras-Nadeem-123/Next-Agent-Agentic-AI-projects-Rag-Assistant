"""ChromaDB vector store for document embeddings."""

import chromadb
from chromadb.config import Settings
from typing import Optional
import uuid


class VectorStore:
    """ChromaDB-based vector store for document storage and retrieval."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents"
    ):
        """Initialize the vector store.

        Args:
            persist_directory: Directory to persist the database.
            collection_name: Name of the collection to use.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None
    ) -> list[str]:
        """Add documents to the vector store.

        Args:
            texts: List of text chunks.
            embeddings: List of embedding vectors.
            metadatas: Optional list of metadata dicts for each chunk.

        Returns:
            List of document IDs.
        """
        if not texts:
            return []

        # Generate unique IDs for each document
        ids = [str(uuid.uuid4()) for _ in texts]

        # Default metadata if not provided
        if metadatas is None:
            metadatas = [{"chunk_index": i} for i in range(len(texts))]

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return ids

    def search(
        self,
        query_embedding: list[float],
        k: int = 3
    ) -> list[dict]:
        """Search for similar documents.

        Args:
            query_embedding: The query embedding vector.
            k: Number of results to return.

        Returns:
            List of results with documents, metadatas, and distances.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None
                })

        return formatted_results

    def clear(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def get_document_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()

    def get_all_sources(self) -> list[str]:
        """Get list of all unique source files in the collection."""
        results = self.collection.get(include=["metadatas"])
        sources = set()
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])
        return sorted(list(sources))
