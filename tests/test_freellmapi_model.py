import os
from unittest.mock import patch

from deo.agents.factory import _build_model
from deo.config import DeoConfig


def test_build_model_uses_freellmapi_openai_compatible_gateway():
    config = DeoConfig()
    with patch.dict(os.environ, {"FREELLMAPI_API_KEY": "test-key"}, clear=True):
        with patch("langchain_openai.ChatOpenAI") as chat:
            _build_model(config)
    kwargs = chat.call_args.kwargs
    assert kwargs["model"] == "auto:default"
    assert kwargs["api_key"] == "test-key"
    assert kwargs["base_url"] == "http://localhost:3001/v1"


def test_build_model_honors_freellmapi_base_url_override():
    config = DeoConfig()
    with patch.dict(
        os.environ,
        {
            "FREELLMAPI_API_KEY": "test-key",
            "FREELLMAPI_BASE_URL": "https://gateway.example/v1",
        },
        clear=True,
    ):
        with patch("langchain_openai.ChatOpenAI") as chat:
            _build_model(config)
    assert chat.call_args.kwargs["base_url"] == "https://gateway.example/v1"
