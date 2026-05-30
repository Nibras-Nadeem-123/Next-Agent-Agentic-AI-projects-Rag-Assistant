"""Gemini embeddings for text vectorization."""

import google.generativeai as genai
from typing import Optional


class GeminiEmbeddings:
    """Wrapper for Gemini's text embedding API."""

    def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
        """Initialize the Gemini embeddings client.

        Args:
            api_key: Gemini API key.
            model: Embedding model to use.
        """
        genai.configure(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts for document storage.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a query for retrieval.

        Args:
            query: The query text to embed.

        Returns:
            Embedding vector for the query.
        """
        result = genai.embed_content(
            model=self.model,
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']
