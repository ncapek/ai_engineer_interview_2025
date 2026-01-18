# DocTalk – Question Your Documents

DocTalk is a lightweight RAG (Retrieval-Augmented Generation) system that enables users to upload internal documents (PDF, DOCX, MD) and ask questions about their content. The system provides grounded answers with citations, ensuring users can verify the source of information.

## Features

### Core Requirements ✅

- **Document Upload**: Support for multiple document formats (PDF, DOCX, TXT, MD)
- **Q&A Interface**: Interactive CLI for asking questions about uploaded documents
- **Grounded Answers**: Answers are generated only from uploaded document content
- **Citations**: Each answer includes citations with:
  - Document name
  - Chunk ID (unique identifier)
  - Text excerpt from the source

### Stretch Goals Implemented ✅

- **RAG with Vector Store**: 
  - OpenAI embeddings (`text-embedding-3-small`)
  - MongoDB Atlas Vector Search for similarity search
  - Recursive text chunking with configurable overlap
  
- **LangGraph Agents**: 
  - Two-node agent workflow (retrieve → generate)
  - Structured output for citation tracking

- **Basic Observability**:
  - Token usage logging for embedding calls

### Not Implemented ❌

- **Azure Deployment**: FastAPI wrapper, AAD authentication, Key Vault integration, and Azure App Service/Container Apps deployment
- **Advanced Observability**: Metrics dashboard, evaluation hooks, cost tracking
- **Governance**: Collibra integration, data governance checklist

## Assumptions & Limitations

### Assumptions

- **Infrastructure**: 
  - MongoDB Atlas cluster is set up and accessible
  - Vector search index is configured on the `chunks` collection
  - OpenAI API access is available with valid API key

- **Environment**: 
  - Required environment variables are set (`MONGODB_URI`, `OPENAI_API_KEY`)
  - Python 3.13+ and `uv` package manager are installed

- **Documents**: 
  - Documents are text-based. Supported formats are: PDF, DOCX, MD, TXT
  - Complex formatting, images, and tables may not be fully preserved

- **Usage**: 
  - Single-user, local development environment
  - No authentication or authorization required
  - Documents are processed synchronously
  - Single question format, no interactive chatting

### Limitations

- **Interface**: CLI-only interface; no web UI or API endpoints at this time
- **Document Management**: 
  - No document deletion or update functionality
  - No document versioning
  - No duplicate detection
  - Uploading the same document creates duplicate chunks unless cleanup is used
  
- **Retrieval**: 
  - Fixed retrieval of top 5 chunks per query
  - No query expansion or re-ranking

- **Citations**: 
  - Citations are based on LLM's self-reported chunk usage (via structured output)

- **Observability**: 
  - Basic logging only; no metrics dashboard
  - No cost tracking or budget limits
  - No evaluation metrics (groundedness, accuracy, etc.)

- **Scalability**: 
  - No multi-user support or session management
  - No rate limiting or request queuing
  - Synchronous processing may be slow for large documents

- **Security**: 
  - No authentication or authorization
  - API keys stored in environment variables (not in Key Vault)
  - No data encryption at rest beyond MongoDB's defaults

## Architecture

### High-Level Overview

DocTalk follows the simplest form of RAG architecture

### Core Components

**IngestionPipeline** (`src/doctalk/pipeline/ingestion.py`)
- Loads documents using LangChain loaders (PDF, DOCX, TXT, MD)
- Splits text into chunks using `RecursiveCharacterTextSplitter`
- Generates embeddings via `EmbeddingManager`
- Stores chunks in MongoDB Atlas via `VectorStoreManager`

**EmbeddingManager** (`src/doctalk/embedding/manager.py`)
- Wraps OpenAI embeddings API
- Supports single and batch embedding generation
- Uses `text-embedding-3-small` model by default
- Logs token usage for observability

**VectorStoreManager** (`src/doctalk/storage/manager.py`)
- Manages MongoDB Atlas Vector Search operations
- Stores chunks with embeddings and metadata
- Performs similarity search using vector embeddings
- Returns top-K relevant chunks with scores

**RAGAgent** (`src/doctalk/agents/rag_agent.py`)
- LangGraph-based agent with two-node workflow:
  1. **Retrieve Node**: Embeds query → vector search → returns top chunks
  2. **Generate Node**: Formats context → LLM call → extracts citations
- Uses structured output for citation tracking
- Returns `Answer` objects with text and citations

### Data Flow

**Document Upload Flow:**
1. User runs `upload_documents.py` with file paths
2. `IngestionPipeline` loads and chunks each document
3. `EmbeddingManager` generates embeddings for all chunks
4. `VectorStoreManager` stores chunks in MongoDB with embeddings

**Question Answering Flow:**
1. User runs `ask_question.py` and enters a question
2. `RAGAgent.ask()` invokes the LangGraph workflow:
   - **Retrieve**: Query is embedded → vector search returns top 5 chunks
   - **Generate**: Chunks are formatted → LLM generates answer with structured output
3. LLM returns answer text + list of referenced chunk numbers
4. Agent maps chunk numbers to actual chunks → builds citations
  5. `Answer` object is returned with text and citations

## Quick Start

### Prerequisites

- Python 3.13+
- `uv` package manager
- MongoDB Atlas cluster with vector search index configured
- OpenAI API key

### Setup

1. **Install dependencies:**
   ```bash
   uv sync --dev
   ```

2. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```bash
   MONGODB_URI=your_mongodb_atlas_connection_string
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Set up MongoDB vector search index:**
   Ensure your MongoDB Atlas cluster has a vector search index on the `chunks` collection with:
   - Index name: `vector_index`
   - Field: `embedding`
   - Dimensions: 1536 (for `text-embedding-3-small`)

### Usage

1. **Upload documents:**
   ```bash
   # Upload a single file
   uv run python scripts/upload_documents.py document.pdf

   # Upload all files from a directory
   uv run python scripts/upload_documents.py sample_documents/

   # Clear existing chunks and upload
   uv run python scripts/upload_documents.py --clear sample_documents/
   ```

2. **Ask questions:**
   ```bash
   uv run python scripts/ask_question.py
   ```
   Then enter your question when prompted.

### Example

```bash
# Upload sample documents
uv run python scripts/upload_documents.py --clear sample_documents/

# Ask a question
uv run python scripts/ask_question.py
# Enter: "What is a QFG-42?"

# Output:
# Answer: The QFG-42, or Quantum Flux Generator, is a device that...
# Citations:
#   [1] technical_guide.md
#       Chunk ID: 696d0fc969b2e9fa922e1bf3
#       Excerpt: "# Quantum Flux Generator Operations Manual..."
```

## Development

### Prerequisites

- Python 3.13+
- `uv` package manager

### Setup

```bash
uv sync --dev
```

### Development Workflow

The project uses the following tools for code quality and testing:

**Linting:**
```bash
uv run ruff check .
```

**Formatting:**
```bash
uv run ruff format .
```

**Testing:**
```bash
uv run pytest
```

## Future Work / If I Had More Time

Given additional time, I would focus on evaluation, experimentation, and optimization of the RAG system:

### Evaluation & Metrics

- **Conversation functionality**
  - Extend script to allow for multi-turn conversations instead of single question runs 

- **Groundedness Evaluation**:
  - Implement metrics to measure how well answers are grounded in source documents
  - Build a dataset to properly evaluate system performance 

- **Experimentation**:
    - Try alternative approaches to chunking, embeddings, retrieval, models, agent architectures...

