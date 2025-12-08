"""Tests for core pure_visual_grounding package imports and basic functionality."""
import pytest
import importlib


def test_import_core_package():
    """Test that core package imports successfully."""
    mod = importlib.import_module("pure_visual_grounding")
    assert hasattr(mod, "process_pdf_with_vision")
    assert hasattr(mod, "__version__")


def test_import_core_submodules():
    """Test that core submodules can be imported."""
    from pure_visual_grounding import core
    from pure_visual_grounding import utils
    from pure_visual_grounding import json_cleaner
    
    assert hasattr(core, "process_pdf_with_vision")
    assert hasattr(utils, "clean_metadata")
    assert hasattr(json_cleaner, "parse_generated_report")

