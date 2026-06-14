# Contributing to optiVLSI

Contributions are welcome! Here's how you can help:

## Reporting Issues

Open an issue on [GitHub](https://github.com/rohankalbag/optiVLSI/issues) with:
- A clear description of the bug or feature request
- Steps to reproduce (for bugs)
- Expected vs actual behavior

## Development Setup

```bash
git clone https://github.com/rohankalbag/optiVLSI.git
cd optiVLSI
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
pytest --cov=optivlsi --cov-report=term
```

## Code Style

- Follow PEP 8
- Add docstrings to all public functions
- Ensure all tests pass before submitting a PR