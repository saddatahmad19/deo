"""Test that factory passes per-role model overrides to build_subagents."""
from unittest.mock import patch
from deo.config import DeoConfig


def test_factory_passes_per_role_models_to_build_subagents():
    """Verify that create_deo_agent passes per-role models from config to build_subagents."""
    config = DeoConfig()
    config.subagents.implementer_model = "anthropic:claude-opus-4-7"
    config.subagents.reviewer_model = "anthropic:claude-haiku-4-5-20251001"

    with patch("deo.agents.factory.build_subagents") as mock_build, \
         patch("deo.agents.factory._build_model"), \
         patch("deo.agents.factory.create_deep_agent"), \
         patch("deo.agents.factory.configure_git_tools"), \
         patch("deo.agents.factory.configure_chroma_tools"), \
         patch("deo.agents.factory.build_orchestrator_middleware"):
        mock_build.return_value = []
        from deo.agents.factory import create_deo_agent
        create_deo_agent(config, manager=None, indexer=None)
        call_kwargs = mock_build.call_args.kwargs
        assert call_kwargs["implementer_model"] == "anthropic:claude-opus-4-7"
        assert call_kwargs["reviewer_model"] == "anthropic:claude-haiku-4-5-20251001"
