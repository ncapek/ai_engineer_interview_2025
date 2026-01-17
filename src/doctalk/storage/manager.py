import time
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


class VectorStoreManager:
    """Manages MongoDB vector store operations for document chunks."""

    def __init__(
        self,
        mongodb_uri: str,
        db_name: str = 'doctalk',
        collection_name: str = 'chunks',
        vector_index_name: str = 'vector_index',
    ):
        """Initialize the vector store manager.

        :param mongodb_uri: MongoDB connection URI
        :param db_name: Database name
        :param collection_name: Collection name for storing chunks
        :param vector_index_name: Name of the vector search index
        """
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.vector_index_name = vector_index_name
        self._client: MongoClient | None = None
        self._db: Database | None = None
        self._collection: Collection | None = None

    @property
    def collection(self) -> Collection:
        """Get or create collection."""
        if self._collection is None:
            self._client = MongoClient(self.mongodb_uri)
            self._db = self._client[self.db_name]
            self._collection = self._db[self.collection_name]
        return self._collection

    def insert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Insert chunks with embeddings into the vector store.

        :param chunks: List of chunk dictionaries with 'text' and 'embedding' keys
        """
        if not chunks:
            return

        self.collection.insert_many(chunks)

    def vector_search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        num_candidates: int = 50,
    ) -> list[dict[str, Any]]:
        """Perform vector similarity search.

        :param query_embedding: Query vector embedding
        :param limit: Maximum number of results to return
        :param num_candidates: Number of candidates to consider (higher = more accurate but slower)
        :return: List of matching documents with text and score
        """
        pipeline = [
            {
                '$vectorSearch': {
                    'index': self.vector_index_name,
                    'path': 'embedding',
                    'queryVector': query_embedding,
                    'numCandidates': num_candidates,
                    'limit': limit,
                },
            },
            {
                '$project': {
                    '_id': 0,
                    'text': 1,
                    'score': {'$meta': 'vectorSearchScore'},
                },
            },
        ]

        return list(self.collection.aggregate(pipeline))
