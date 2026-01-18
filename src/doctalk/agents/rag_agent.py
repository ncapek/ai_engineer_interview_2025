"""Simple LangGraph agent for RAG."""
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from doctalk.embedding import EmbeddingManager
from doctalk.models import Answer, Citation, RetrievedChunk
from doctalk.storage import VectorStoreManager


class LLMResponse(BaseModel):
    """Structured response from LLM."""

    answer: str
    referenced_chunk_numbers: list[int]


class AgentState(TypedDict):
    """State for the RAG agent."""
    query: str
    retrieved_chunks: list[RetrievedChunk]
    answer: Answer


class RAGAgent:
    """Simple LangGraph agent for question answering with RAG."""

    def __init__(
        self,
        embedder: EmbeddingManager,
        vector_store: VectorStoreManager,
        llm_model: str = 'gpt-4o-mini',
    ):
        """Initialize the RAG agent.

        :param embedder: EmbeddingManager instance
        :param vector_store: VectorStoreManager instance
        :param llm_model: OpenAI LLM model name
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model=llm_model, temperature=0).with_structured_output(LLMResponse)

        self.graph = self._build_graph()

    def _retrieve_node(self, state: AgentState) -> AgentState:
        """Node that retrieves relevant chunks.

        :param state: Current agent state
        :return: Updated state with retrieved chunks
        """
        query = state['query']
        query_embedding = self.embedder.embed(query)
        chunk_dicts = self.vector_store.vector_search(query_embedding, limit=5)

        # Convert dicts to RetrievedChunk objects
        retrieved_chunks = [
            RetrievedChunk(
                chunk_id=str(chunk.get('_id', '')),
                text=chunk.get('text', ''),
                document_name=chunk.get('document_name', 'Unknown'),
                document_id=chunk.get('document_id', ''),
                chunk_index=chunk.get('chunk_index', 0),
                score=chunk.get('score', 0.0),
                metadata=chunk.get('metadata', {}),
            )
            for chunk in chunk_dicts
        ]

        state['retrieved_chunks'] = retrieved_chunks
        return state


    def _generate_answer_node(self, state: AgentState) -> AgentState:
        """Node that generates answer from retrieved chunks.

        :param state: Current agent state
        :return: Updated state with answer and citations
        """
        query = state['query']
        chunks = state['retrieved_chunks']

        if not chunks:
            state['answer'] = Answer(
                text='I could not find any relevant information to answer your question.',
                citations=[],
            )
            return state

        # Number chunks 1, 2, 3... for the LLM prompt
        # Create lookup: chunk_number_in_prompt -> actual_chunk_object
        # Example: If LLM says "I used chunk [2]", this tells us which chunk that is
        chunk_number_to_chunk: dict[int, RetrievedChunk] = {}
        context_parts = []
        for chunk_number, chunk in enumerate(chunks, start=1):
            chunk_number_to_chunk[chunk_number] = chunk
            context_parts.append(f'[{chunk_number}] {chunk.text}')

        context = '\n\n'.join(context_parts)

        prompt = f"""You are a helpful assistant that answers questions based only on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer the question using only the information from the context above
- If the context doesn't contain enough information, say so clearly
- In your response, specify which chunk numbers (1, 2, 3, etc.) you used to construct your answer
- Only include chunk numbers that you actually used to answer the question
- Be concise and accurate

Answer the question and specify which chunks you used:"""

        response: LLMResponse = self.llm.invoke(prompt)

        # Build citations: map LLM's chunk numbers (1, 2, 3...) back to actual chunks
        citations = []
        for chunk_number in response.referenced_chunk_numbers:
            if chunk_number not in chunk_number_to_chunk:
                continue

            chunk = chunk_number_to_chunk[chunk_number]

            text_excerpt = chunk.text[:100] + '...' if len(chunk.text) > 100 else chunk.text
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    document_name=chunk.document_name,
                    text_excerpt=text_excerpt,
                )
            )

        state['answer'] = Answer(
            text=response.answer,
            citations=citations,
        )
        return state

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph agent graph.

        :return: Compiled graph
        """
        workflow = StateGraph(AgentState)

        workflow.add_node('retrieve', self._retrieve_node)
        workflow.add_node('generate', self._generate_answer_node)

        workflow.set_entry_point('retrieve')
        workflow.add_edge('retrieve', 'generate')
        workflow.add_edge('generate', END)

        return workflow.compile()

    def ask(self, query: str) -> Answer:
        """Ask a question and get an answer with citations.

        :param query: User question
        :return: Answer with text and citations
        """
        initial_state: AgentState = {
            'query': query,
            'retrieved_chunks': [],
            'answer': Answer(text='', citations=[]),
        }

        result = self.graph.invoke(initial_state)
        return result['answer']
