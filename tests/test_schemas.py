# tests/test_schemas.py
from deo.agents.schemas import (
    PlanStep, PlanResult,
    FileChange, ImplementResult,
    ReviewIssue, ReviewResult,
    TestResult, DocResult,
)

def test_plan_result_roundtrip():
    data = PlanResult(
        steps=[PlanStep(step="add util", file="deo/util.py", action="create")],
        files_affected=["deo/util.py"],
        risks=[],
        estimated_complexity="low",
    )
    assert data.steps[0].file == "deo/util.py"
    assert data.estimated_complexity == "low"

def test_review_result_approved_false_when_critical():
    result = ReviewResult(
        issues=[ReviewIssue(severity="critical", file="a.py", line=10,
                            message="null deref", suggestion="guard it")],
        approved=False,
        verdict="critical issues found",
    )
    assert result.approved is False

def test_implement_result_empty_caveats():
    r = ImplementResult(
        files_changed=[FileChange(filepath="x.py", action="modified", summary="added fn")],
        summary="done",
        caveats=[],
    )
    assert r.caveats == []

def test_test_result_all_pass():
    r = TestResult(test_files=["tests/test_x.py"], coverage_estimate="87%",
                   all_pass=True, failures=[])
    assert r.all_pass is True

def test_doc_result():
    r = DocResult(files_updated=["README.md"], summary="updated usage")
    assert "README.md" in r.files_updated
