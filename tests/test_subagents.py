# tests/test_subagents.py
from deo.agents.subagents import (
    build_planner_subagent, build_implementer_subagent,
    build_reviewer_subagent, build_tester_subagent,
    build_documenter_subagent, build_subagents,
)
from deo.agents.schemas import (
    PlanResult, ImplementResult, ReviewResult, RunResult, DocResult,
)

def test_planner_has_response_format():
    sa = build_planner_subagent()
    assert "response_format" in sa
    assert sa["response_format"] is PlanResult

def test_implementer_has_response_format():
    sa = build_implementer_subagent()
    assert sa["response_format"] is ImplementResult

def test_reviewer_has_response_format():
    sa = build_reviewer_subagent()
    assert sa["response_format"] is ReviewResult

def test_tester_has_response_format():
    sa = build_tester_subagent()
    assert sa["response_format"] is RunResult

def test_documenter_has_response_format():
    sa = build_documenter_subagent()
    assert sa["response_format"] is DocResult

def test_per_role_model_override():
    sa = build_implementer_subagent(model="anthropic:claude-opus-4-7")
    assert sa["model"] == "anthropic:claude-opus-4-7"

def test_build_subagents_passes_per_role_models():
    sas = build_subagents(
        implementer_model="anthropic:claude-opus-4-7",
        reviewer_model="anthropic:claude-haiku-4-5-20251001",
        enable_tester=False,
        enable_documenter=False,
    )
    names = {s["name"]: s for s in sas}
    assert names["implementer-agent"]["model"] == "anthropic:claude-opus-4-7"
    assert names["reviewer-agent"]["model"] == "anthropic:claude-haiku-4-5-20251001"
    assert "tester-agent" not in names
