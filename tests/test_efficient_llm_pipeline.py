"""Tests for efficient_llm pipeline functions (signature and basic logic)."""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock


@pytest.fixture
def pipeline_module():
    """Fixture providing pipeline module."""
    pytest.importorskip("torch")
    from efficient_llm import pipeline
    return pipeline


@pytest.fixture
def mock_config():
    """Fixture providing mock config."""
    pytest.importorskip("torch")
    from efficient_llm.config import PipelineConfig
    
    return PipelineConfig(
        dots_model_path="./dots_model",
        image_folder="./images",
        pdf_path="./test.pdf",
        crops_dir="./crops",
        reports_dir="./reports"
    )


def test_run_pipeline_signature(pipeline_module):
    """Test that run_pipeline has correct signature."""
    import inspect
    sig = inspect.signature(pipeline_module.run_pipeline)
    params = list(sig.parameters.keys())
    
    assert "cfg" in params


def test_run_dots_layout_on_folder_signature(pipeline_module):
    """Test that run_dots_layout_on_folder has correct signature."""
    import inspect
    sig = inspect.signature(pipeline_module.run_dots_layout_on_folder)
    params = list(sig.parameters.keys())
    
    assert "engine" in params
    assert "image_folder" in params
    assert "prompt" in params
    assert "batch_size" in params


def test_run_dots_layout_on_folder_no_pngs(pipeline_module):
    """Test run_dots_layout_on_folder with folder containing no PNGs."""
    import tempfile
    
    # Mock engine
    mock_engine = MagicMock()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            pipeline_module.run_dots_layout_on_folder(
                engine=mock_engine,
                image_folder=tmpdir
            )
