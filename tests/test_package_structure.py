"""Tests for package structure and metadata."""
import pytest
import importlib


def test_core_package_version():
    """Test that core package has version."""
    mod = importlib.import_module("pure_visual_grounding")
    assert hasattr(mod, "__version__")
    assert isinstance(mod.__version__, str)
    assert len(mod.__version__) > 0


def test_core_package_author():
    """Test that core package has author info."""
    mod = importlib.import_module("pure_visual_grounding")
    assert hasattr(mod, "__author__")
    assert hasattr(mod, "__email__")


def test_local_dual_llm_package_version():
    """Test that local_dual_llm package has version (if available)."""
    pytest.importorskip("torch")
    
    mod = importlib.import_module("local_dual_llm")
    assert hasattr(mod, "__version__")
    assert isinstance(mod.__version__, str)


def test_local_dual_llm_package_exports():
    """Test that local_dual_llm exports expected items."""
    pytest.importorskip("torch")
    
    mod = importlib.import_module("local_dual_llm")
    assert hasattr(mod, "__all__")
    
    expected_exports = [
        "QwenEngine",
        "LocalDualLLMConfig",
        "inference_pdf",
        "batched_inference",
        "recursive_batched_inference"
    ]
    
    for export in expected_exports:
        assert export in mod.__all__


