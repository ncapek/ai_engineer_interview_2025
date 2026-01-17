from openai import OpenAI


class EmbeddingManager:
    """Manages embedding generation for text."""

    def __init__(self, model: str = 'text-embedding-3-small', api_key: str | None = None):
        """Initialize the embedding manager.

        :param model: OpenAI embedding model name
        :param api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text string.

        :param text: Text to embed
        :return: Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )

        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple text strings.

        :param texts: List of text strings to embed
        :return: List of embedding vectors
        """
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        return [item.embedding for item in response.data]
