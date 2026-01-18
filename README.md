# DocTalk – Question Your Documents

DocTalk is a lightweight RAG (Retrieval-Augmented Generation) system that enables users to upload internal documents (PDF, DOCX, MD) and ask questions about their content. The system provides grounded answers with citations, ensuring users can verify the source of information.

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