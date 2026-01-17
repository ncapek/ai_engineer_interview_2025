"""Data models for document chunks."""

from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represents a document chunk with text, embedding, and metadata."""

    text: str = Field(description='Chunk text content')
    embedding: list[float] = Field(description='Vector embedding')
    document_name: str = Field(description='Source document name/path')
    document_id: str = Field(description='Unique document identifier')
    chunk_index: int = Field(description='Index of chunk within document')
    metadata: dict[str, Any] = Field(default_factory=dict, description='Additional metadata')
