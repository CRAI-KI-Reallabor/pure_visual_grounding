"""Tests for LocalDualLLMConfig."""
import pytest
import os


@pytest.fixture
def config():
    """Fixture providing LocalDualLLMConfig instance."""
    pytest.importorskip("torch")
    from local_dual_llm.config import LocalDualLLMConfig
    return LocalDualLLMConfig()


def test_config_defaults(config):
    """Test that config has expected default values."""
    assert config.dataset_dir == "./dataset"
    assert config.results_dir == "./results"
    assert config.model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert config.device == "auto"
    assert config.gen_max_new_tokens_ocr == 2086
    assert config.gen_max_new_tokens_report == 2286


def test_config_custom_values():
    """Test creating config with custom values."""
    pytest.importorskip("torch")
    from local_dual_llm.config import LocalDualLLMConfig
    
    cfg = LocalDualLLMConfig(
        dataset_dir="./custom_dataset",
        results_dir="./custom_results",
        device="cpu"
    )
    assert cfg.dataset_dir == "./custom_dataset"
    assert cfg.results_dir == "./custom_results"
    assert cfg.device == "cpu"


def test_config_debug_log_dir_property(config):
    """Test debug_log_dir property."""
    debug_dir = config.debug_log_dir
    assert debug_dir == os.path.join(config.results_dir, "debug_logs")


