"""Tests for efficient_llm package imports."""
import pytest
import importlib


def test_import_efficient_llm_optional():
    """Test that efficient_llm can be imported (if dependencies available)."""
    pytest.importorskip("torch")
    
    mod = importlib.import_module("efficient_llm")
    assert hasattr(mod, "PipelineConfig")
    assert hasattr(mod, "run_pipeline")


def test_import_efficient_llm_submodules():
    """Test that efficient_llm submodules can be imported."""
    pytest.importorskip("torch")
    
    from efficient_llm import config, engine, pipeline, utils, prompts
    
    assert hasattr(config, "PipelineConfig")
    assert hasattr(engine, "DotsLayoutEngine")
    assert hasattr(pipeline, "run_pipeline")
    assert hasattr(utils, "convert_pdf_to_images")
    assert hasattr(prompts, "DOTS_LAYOUT_PROMPT")
