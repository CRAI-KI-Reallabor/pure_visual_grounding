"""Tests for JSON parsing utilities."""
import pytest
from pure_visual_grounding.json_cleaner import parse_generated_report


def test_parse_generated_report_dict():
    """Test parsing when input is already a dict."""
    data = {"key": "value"}
    assert parse_generated_report(data) == data


def test_parse_generated_report_valid_json():
    """Test parsing valid JSON string."""
    json_str = '{"key": "value", "number": 42}'
    result = parse_generated_report(json_str)
    assert result == {"key": "value", "number": 42}


def test_parse_generated_report_markdown_fenced():
    """Test parsing JSON from markdown code blocks."""
    markdown = "```json\n{\"key\": \"value\"}\n```"
    result = parse_generated_report(markdown)
    assert result == {"key": "value"}


def test_parse_generated_report_with_extra_text():
    """Test parsing JSON with surrounding text."""
    text = "Here is the result: {\"key\": \"value\"} End of result."
    result = parse_generated_report(text)
    assert result == {"key": "value"}


def test_parse_generated_report_invalid_returns_string():
    """Test that invalid JSON returns original string."""
    invalid = "This is not JSON at all!"
    result = parse_generated_report(invalid)
    assert isinstance(result, str)
    assert result == invalid

