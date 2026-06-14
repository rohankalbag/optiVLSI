.PHONY: install test test-cov lint build publish clean

# Install package in development mode
install:
	pip install -e .

# Install dev dependencies
install-dev:
	pip install -e ".[dev]"
	pip install -r requirements-dev.txt

# Run tests
test:
	pytest -v

# Run tests with coverage
test-cov:
	pytest --cov=optivlsi --cov-report=term --cov-report=html

# Run benchmarks
bench:
	pytest tests/test_benchmarks.py -v --benchmark-only

# Build distribution packages
build:
	python -m build

# Check distribution packages
check:
	twine check dist/*

# Publish to Test PyPI
publish-test:
	twine upload --repository testpypi dist/*

# Publish to PyPI
publish:
	twine upload dist/*

# Clean build artifacts
clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache/ __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true