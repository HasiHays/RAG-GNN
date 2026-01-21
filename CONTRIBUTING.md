# Contributing to RAG-GNN

Thank you for your interest in contributing to RAG-GNN! This document provides guidelines for contributing to the project.

## Getting started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/RAG-GNN.git
   cd RAG-GNN
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development workflow

### Running tests

```bash
pytest tests/ -v
```

### Code formatting

We use Black for code formatting:

```bash
black rag_gnn/ tests/ examples/
```

### Linting

```bash
flake8 rag_gnn/ tests/
```

### Type checking

```bash
mypy rag_gnn/
```

## Making changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Add tests for new functionality

4. Run tests and linting:
   ```bash
   pytest tests/ -v
   black rag_gnn/ tests/
   flake8 rag_gnn/ tests/
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Open a Pull Request

## Pull Request guidelines

- Provide a clear description of the changes
- Include tests for new functionality
- Update documentation as needed
- Ensure all tests pass
- Follow the existing code style

## Reporting issues

When reporting issues, please include:

- Python version
- Package versions (`pip freeze`)
- Minimal code example to reproduce the issue
- Full error traceback

## Code of conduct

Please be respectful and constructive in all interactions. We welcome contributions from everyone.

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.
