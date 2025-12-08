"""Tests for core package utility functions."""
import pytest
from pure_visual_grounding.utils import clean_metadata, ensure_utf8, clean_empty_fields_inplace


def test_ensure_utf8():
    """Test UTF-8 encoding helper."""
    assert ensure_utf8("test") == "test"
    assert ensure_utf8("café") == "café"
    assert ensure_utf8(123) == 123  # Non-string unchanged


def test_clean_metadata():
    """Test metadata cleaning function."""
    metadata = {
        "pdf_name": "test.pdf",
        "page_number": 1,
        "error": "none",
        "unicode": "café"
    }
    cleaned = clean_metadata(metadata)
    assert cleaned == metadata
    assert isinstance(cleaned["unicode"], str)


def test_clean_empty_fields_inplace():
    """Test recursive empty field removal."""
    data = {
        "keep": "value",
        "empty_str": "",
        "empty_list": [],
        "empty_dict": {},
        "nested": {
            "keep": "value",
            "empty": None
        },
        "list_with_empty": [1, "", None, {"empty": ""}]
    }
    
    result = clean_empty_fields_inplace(data)
    
    assert "keep" in data
    assert "empty_str" not in data
    assert "empty_list" not in data
    assert "empty_dict" not in data
    assert "empty" not in data["nested"]
    assert len(data["list_with_empty"]) == 1  # Only [1] remains

