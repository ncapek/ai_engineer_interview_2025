# AI Engineer Interview 2025

A basic document chatbot application built for the AI Engineer interview process.

## Overview

This project implements a basic document chatbot that allows users to interact with documents through natural language queries.

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