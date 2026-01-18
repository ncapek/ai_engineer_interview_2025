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


class Citation(BaseModel):
    """Represents a citation to a source chunk."""

    chunk_id: str = Field(description='Unique chunk identifier')
    document_name: str = Field(description='Source document name')
    text_excerpt: str = Field(description='Excerpt from the chunk')


class RetrievedChunk(BaseModel):
    """Represents a chunk retrieved from vector search (without embedding)."""

    chunk_id: str = Field(description='Unique chunk identifier (MongoDB _id)')
    text: str = Field(description='Chunk text content')
    document_name: str = Field(description='Source document name/path')
    document_id: str = Field(description='Unique document identifier')
    chunk_index: int = Field(description='Index of chunk within document')
    score: float = Field(description='Similarity score from vector search')
    metadata: dict[str, Any] = Field(default_factory=dict, description='Additional metadata')


class Answer(BaseModel):
    """Represents a generated answer with citations."""

    text: str = Field(description='The answer text')
    citations: list[Citation] = Field(default_factory=list, description='Citations to source chunks')
