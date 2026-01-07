"""Tests for PipelineConfig."""
import pytest
import os


@pytest.fixture
def config():
    """Fixture providing PipelineConfig instance."""
    pytest.importorskip("torch")
    from efficient_llm.config import PipelineConfig
    return PipelineConfig(
        dots_model_path="./dots_model",
        image_folder="./images",
        pdf_path="./test.pdf",
        crops_dir="./crops",
        reports_dir="./reports"
    )


def test_config_defaults(config):
    """Test that config has expected default values."""
    assert config.padding == 12
    assert config.max_side == 1280
    assert config.batch_size == 18
    assert config.base_max_new_tokens == 12000
    assert config.extra_max_new_tokens == 15000
    assert config.gemma_model_id == "google/gemma-3n-e4b-it"
    assert config.crop_upscale == 2.0
    assert config.attn_impl == "flash_attention_2"
    assert config.device_map == "auto"
    assert config.torch_dtype == "bfloat16"


def test_config_custom_values():
    """Test creating config with custom values."""
    pytest.importorskip("torch")
    from efficient_llm.config import PipelineConfig
    
    cfg = PipelineConfig(
        dots_model_path="./custom_dots",
        image_folder="./custom_images",
        pdf_path=None,
        crops_dir="./custom_crops",
        reports_dir="./custom_reports",
        batch_size=4,
        torch_dtype="float16"
    )
    assert cfg.dots_model_path == "./custom_dots"
    assert cfg.batch_size == 4
    assert cfg.torch_dtype == "float16"
    assert cfg.pdf_path is None
