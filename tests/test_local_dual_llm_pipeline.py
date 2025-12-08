"""Tests for local_dual_llm pipeline functions (signature and basic logic)."""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock


@pytest.fixture
def pipeline_module():
    """Fixture providing pipeline module."""
    pytest.importorskip("torch")
    from local_dual_llm import pipeline
    return pipeline


@pytest.fixture
def mock_config():
    """Fixture providing mock config."""
    pytest.importorskip("torch")
    from local_dual_llm.config import LocalDualLLMConfig
    
    config = LocalDualLLMConfig()
    config.dataset_dir = "./test_dataset"
    config.results_dir = "./test_results"
    config.cache_dir = "./test_cache"
    return config


def test_inference_pdf_signature(pipeline_module):
    """Test that inference_pdf has correct signature."""
    import inspect
    sig = inspect.signature(pipeline_module.inference_pdf)
    params = list(sig.parameters.keys())
    
    assert "pdf_path" in params
    assert "config" in params
    assert "engine" in params
    assert "custom_output_dir" in params


def test_batched_inference_signature(pipeline_module):
    """Test that batched_inference has correct signature."""
    import inspect
    sig = inspect.signature(pipeline_module.batched_inference)
    params = list(sig.parameters.keys())
    
    assert "input_folder" in params
    assert "config" in params
    assert "engine" in params


def test_recursive_batched_inference_signature(pipeline_module):
    """Test that recursive_batched_inference has correct signature."""
    import inspect
    sig = inspect.signature(pipeline_module.recursive_batched_inference)
    params = list(sig.parameters.keys())
    
    assert "input_root_dir" in params
    assert "config" in params
    assert "engine" in params


def test_batched_inference_nonexistent_folder(pipeline_module, mock_config):
    """Test batched_inference with non-existent folder."""
    result = pipeline_module.batched_inference("/nonexistent/folder", config=mock_config)
    assert result["total"] == 0
    assert result["processed"] == []
    assert result["failed"] == []


def test_batched_inference_empty_folder(pipeline_module, mock_config):
    """Test batched_inference with empty folder."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = pipeline_module.batched_inference(tmpdir, config=mock_config)
        # Should return None or empty result for no PDFs
        assert result is None or result.get("total", 0) == 0

