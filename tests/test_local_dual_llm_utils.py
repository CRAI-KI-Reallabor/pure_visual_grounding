"""Tests for local_dual_llm utility functions."""
import pytest
import json
import tempfile
import os


@pytest.fixture
def utils_module():
    """Fixture providing utils module."""
    pytest.importorskip("torch")
    from local_dual_llm import utils
    return utils


def test_parse_ocr_json_valid(utils_module):
    """Test parsing valid OCR JSON."""
    valid_json = '{"extracted_information": {"key": "value"}}'
    result = utils_module.parse_ocr_json(valid_json, "test.png")
    assert result["ok"] is True
    assert "data" in result
    assert result["data"]["extracted_information"]["key"] == "value"


def test_parse_ocr_json_markdown_fenced(utils_module):
    """Test parsing JSON from markdown code block."""
    fenced = "```json\n{\"extracted_information\": {}}\n```"
    result = utils_module.parse_ocr_json(fenced, "test.png")
    assert result["ok"] is True


def test_parse_ocr_json_invalid(utils_module):
    """Test parsing invalid JSON returns error dict."""
    invalid = "This is not JSON!"
    result = utils_module.parse_ocr_json(invalid, "test.png")
    assert result["ok"] is False
    assert "raw" in result


def test_flatten_extracted_info(utils_module):
    """Test flattening extracted info structure."""
    extracted = {
        "Topic_and_context_information": {
            "technical_identifier": "TEST-123",
            "topic_description": "Test Topic",
            "context_information": "Test context"
        },
        "product_component_information": [{"item": "value"}],
        "embedded_table_chart": []
    }
    
    flattened = utils_module.flatten_extracted_info(extracted)
    
    assert flattened["technical_identifier"] == "TEST-123"
    assert flattened["topic_description"] == "Test Topic"
    assert isinstance(flattened["product_component_information"], list)


def test_parse_json_garbage_valid(utils_module):
    """Test parse_json_garbage with valid JSON."""
    valid = '{"key": "value"}'
    result = utils_module.parse_json_garbage(valid)
    assert result == {"key": "value"}


def test_parse_json_garbage_markdown(utils_module):
    """Test parse_json_garbage with markdown fence."""
    fenced = "```json\n{\"key\": \"value\"}\n```"
    result = utils_module.parse_json_garbage(fenced)
    assert result == {"key": "value"}


def test_parse_json_garbage_invalid(utils_module):
    """Test parse_json_garbage with invalid input."""
    invalid = "Not JSON!"
    result = utils_module.parse_json_garbage(invalid)
    assert "error" in result
    assert "raw" in result


def test_convert_pdf_to_images_nonexistent(utils_module):
    """Test PDF conversion with non-existent file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = utils_module.convert_pdf_to_images("/nonexistent/file.pdf", tmpdir)
        assert result == []


def test_cleanup_folder_using_os(utils_module):
    """Test folder cleanup utility."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_folder = os.path.join(tmpdir, "test_folder")
        os.makedirs(test_folder)
        
        # Create a test file
        test_file = os.path.join(test_folder, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        
        # Cleanup should remove the folder
        utils_module.cleanup_folder_using_os(test_folder)
        assert not os.path.exists(test_folder)

