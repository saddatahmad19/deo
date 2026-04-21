# tests/test_config_models.py
from deo.config import SubagentToggles, DeoConfig

def test_subagent_toggles_has_model_fields():
    t = SubagentToggles()
    assert hasattr(t, "planner_model")
    assert hasattr(t, "implementer_model")
    assert hasattr(t, "reviewer_model")
    assert hasattr(t, "tester_model")
    assert hasattr(t, "documenter_model")

def test_model_fields_default_to_none():
    t = SubagentToggles()
    assert t.planner_model is None
    assert t.implementer_model is None
    assert t.reviewer_model is None
    assert t.tester_model is None
    assert t.documenter_model is None

def test_apply_subagent_model_from_toml():
    config = DeoConfig()
    config._apply({
        "subagents": {
            "implementer_model": "anthropic:claude-opus-4-7",
            "reviewer_model": "anthropic:claude-haiku-4-5-20251001",
        }
    })
    assert config.subagents.implementer_model == "anthropic:claude-opus-4-7"
    assert config.subagents.reviewer_model == "anthropic:claude-haiku-4-5-20251001"
    assert config.subagents.planner_model is None
