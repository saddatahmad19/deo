from deo.config import DeoConfig, ModelConfig


def test_freellmapi_is_the_default_model():
    model = ModelConfig()
    assert model.provider == "freellmapi"
    assert model.name == "auto:default"
    assert model.model_string == "freellmapi:auto:default"


def test_model_override_preserves_freellmapi_auto_profile():
    model = ModelConfig()
    model.override("auto:default")
    assert model.provider == "freellmapi"
    assert model.name == "auto:default"


def test_model_override_accepts_explicit_provider():
    model = ModelConfig()
    model.override("openai:gpt-4o")
    assert model.provider == "openai"
    assert model.name == "gpt-4o"


def test_config_example_uses_freellmapi():
    example = DeoConfig().to_example_toml()
    assert 'provider = "freellmapi"' in example
    assert 'name = "auto:default"' in example
