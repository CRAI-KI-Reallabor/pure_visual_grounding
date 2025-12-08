import importlib
import pytest


def test_import_core():
    mod = importlib.import_module("pure_visual_grounding")
    assert hasattr(mod, "process_pdf_with_vision")


def test_import_local_dual_llm_optional():
    # Skip if heavy optional deps are not installed (torch/gpu stack).
    pytest.importorskip("torch")
    mod = importlib.import_module("local_dual_llm")
    assert hasattr(mod, "inference_pdf")

