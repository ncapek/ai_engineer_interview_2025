import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'text-embedding-3-small'

class EmbeddingManager:
    """Manages embedding generation for text."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        """Initialize the embedding manager.

        :param model: OpenAI embedding model name (defaults to DEFAULT_MODEL)
        :param api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        self.model = model or DEFAULT_MODEL
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

        logger.info(
            f'Embedding call: {response.usage.total_tokens} tokens '
            f'(model: {self.model})'
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

        logger.info(
            f'Embedding batch call: {response.usage.total_tokens} tokens '
            f'for {len(texts)} chunks (model: {self.model})'
        )

        return [item.embedding for item in response.data]
