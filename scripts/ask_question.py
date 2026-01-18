#!/usr/bin/env python3
"""Ask questions using the RAG agent."""
import logging
import os

from dotenv import load_dotenv

from doctalk.agents import RAGAgent
from doctalk.embedding import EmbeddingManager
from doctalk.storage import VectorStoreManager

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

load_dotenv()

query = input('Enter your question: ')

embedder = EmbeddingManager()
vector_store = VectorStoreManager(
    mongodb_uri=os.environ['MONGODB_URI'],
    db_name='doctalk',
    collection_name='chunks',
)

agent = RAGAgent(embedder, vector_store)

print(f'\nQuestion: {query}\n')
print('Thinking...\n')

answer = agent.ask(query)

print('Answer:')
print(answer.text)

if answer.citations:
    print('\n' + '=' * 60)
    print('Citations:')
    print('=' * 60)
    for i, citation in enumerate(answer.citations, 1):
        print(f'\n[{i}] {citation.document_name}')
        print(f'    Chunk ID: {citation.chunk_id}')
        # Truncate excerpt if too long and format nicely
        excerpt = citation.text_excerpt
        if len(excerpt) > 150:
            excerpt = excerpt[:147] + '...'
        print(f'    Excerpt: {excerpt}')
