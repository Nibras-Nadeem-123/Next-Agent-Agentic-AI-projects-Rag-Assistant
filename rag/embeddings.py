"""Embeddings for text vectorization with multiple backend support."""

from typing import Optional
import os


class GeminiEmbeddings:
    """Wrapper for embeddings with multiple backend support."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "text-embedding-004",
        use_local: bool = False
    ):
        """Initialize the embeddings client.

        Args:
            api_key: API key.
            base_url: Optional base URL for OpenAI-compatible endpoint.
            model: Embedding model to use.
            use_local: If True, use local sentence-transformers instead of API.
        """
        self.model = model
        self.use_local = use_local
        self.local_model = None

        if use_local:
            self._init_local_model()
        elif base_url:
            self._init_openai_client(api_key, base_url)
        else:
            self._init_google_client(api_key)

    def _init_local_model(self):
        """Initialize local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.backend = "local"
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def _init_openai_client(self, api_key: str, base_url: str):
        """Initialize OpenAI-compatible client."""
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.backend = "openai"

    def _init_google_client(self, api_key: str):
        """Initialize native Google Generative AI client."""
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        self.model = f"models/{self.model}" if not self.model.startswith("models/") else self.model
        self.backend = "google"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts for document storage."""
        if not texts:
            return []

        if self.backend == "local":
            return self._embed_local(texts)
        elif self.backend == "openai":
            return self._embed_openai(texts)
        else:
            return self._embed_google(texts, task_type="retrieval_document")

    def embed_query(self, query: str) -> list[float]:
        """Embed a query for retrieval."""
        if self.backend == "local":
            return self._embed_local([query])[0]
        elif self.backend == "openai":
            return self._embed_openai([query])[0]
        else:
            return self._embed_google([query], task_type="retrieval_query")[0]

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Embed using local sentence-transformers."""
        embeddings = self.local_model.encode(texts)
        return [emb.tolist() for emb in embeddings]

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Embed using OpenAI-compatible API."""
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def _embed_google(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Embed using native Google Generative AI API."""
        embeddings = []
        for text in texts:
            result = self.genai.embed_content(
                model=self.model,
                content=text,
                task_type=task_type
            )
            embeddings.append(result['embedding'])
        return embeddings
