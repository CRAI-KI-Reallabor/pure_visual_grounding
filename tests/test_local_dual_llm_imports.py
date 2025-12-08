"""Tests for local_dual_llm package imports."""
import pytest
import importlib


def test_import_local_dual_llm_optional():
    """Test that local_dual_llm can be imported (if dependencies available)."""
    pytest.importorskip("torch")
    
    mod = importlib.import_module("local_dual_llm")
    assert hasattr(mod, "inference_pdf")
    assert hasattr(mod, "batched_inference")
    assert hasattr(mod, "recursive_batched_inference")
    assert hasattr(mod, "QwenEngine")
    assert hasattr(mod, "LocalDualLLMConfig")


def test_import_local_dual_llm_submodules():
    """Test that local_dual_llm submodules can be imported."""
    pytest.importorskip("torch")
    
    from local_dual_llm import config, engine, pipeline, utils, prompts
    
    assert hasattr(config, "LocalDualLLMConfig")
    assert hasattr(engine, "QwenEngine")
    assert hasattr(pipeline, "inference_pdf")
    assert hasattr(utils, "convert_pdf_to_images")
    assert hasattr(prompts, "OCR_PROMPT")

