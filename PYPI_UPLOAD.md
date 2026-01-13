# Uploading to Test PyPI

## Prerequisites

1. Install build tools:

```bash
pip install build twine
```

2. Create an account on https://test.pypi.org/

3. Get an API token from https://test.pypi.org/manage/account/token/

## Build and Upload Steps

1. **Build the package:**

```bash
python -m build
```

2. **Upload to Test PyPI:**

```bash
python -m twine upload --repository testpypi dist/*
```

When prompted, use:

- Username: `__token__`
- Password: Your API token (including the `pypi-` prefix)

## Usage After Upload

**Option 1: Install with fallback to main PyPI (recommended)**

```bash
pip install --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ pure-visual-grounder
```

**Option 2: Test PyPI only (may fail due to missing dependencies)**

```bash
pip install -i https://test.pypi.org/simple/ pure-visual-grounder
```

**Note**: Test PyPI doesn't contain all packages from main PyPI. Dependencies like `langsmith`, `langchain`, etc. may not be available on Test PyPI, causing installation to fail with Option 2.

Then import the function:

```python
from app.parsing_strategies.pure_visual_grounding import process_pdf_with_vision
```

## Package Structure

The package exposes only the `process_pdf_with_vision` function from the `pure_visual_grounding` module as requested.

## Configuration Files

The package now uses modern Python packaging with:

- **`pyproject.toml`**: Modern standard configuration (recommended)
- **`setup.py`**: Legacy configuration (still works, but pyproject.toml takes precedence)

## Notes

- Update the version in `pyproject.toml` for each new upload (line 7: `version = "1.0.0"`)
- Update author name and email in `pyproject.toml` if needed
- The package is configured with MIT license
- Dependencies are managed in `requirements.txt` and dynamically loaded via pyproject.toml
- Both `setup.py` and `pyproject.toml` are included for compatibility
