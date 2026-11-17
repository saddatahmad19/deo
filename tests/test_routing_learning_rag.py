"""Coverage for FreeLLMAPI routing, harness probing, learning and hybrid RAG."""
import os
from unittest.mock import patch

from deo.config import DeoConfig
from deo.learning.experience import Episode, PresetBandit, SessionLearner, Signals, infer_outcome
from deo.memory.chunking import chunk_source, identifier_terms
from deo.memory.lexical import LexicalIndex
from deo.routing.freellmapi import DecisionAnswer, RouteState, RoutedChatOpenAI
from deo.routing.router import DecisionRouter, heuristic_route


def _state():
    cfg = DeoConfig().routing
    return RouteState(presets=dict(cfg.presets), role_presets=dict(cfg.role_presets), session_id="s1")


def test_route_state_resolves_role_and_turn_presets():
    rs = _state()
    assert rs.model_for("orchestrator") == "auto:default"   # no turn decision yet
    rs.turn_preset = "fastest"
    assert rs.model_for("orchestrator") == "auto:fast"
    assert rs.model_for("tester") == "auto:reliable"
    rs.role_models["reviewer"] = "auto:my-chain"
    assert rs.model_for("reviewer") == "auto:my-chain"
    assert rs.headers("planner")["X-Session-Id"] == "deo-s1-planner"


def test_routed_chat_openai_injects_model_and_headers():
    rs = _state()
    rs.turn_preset, rs.task_type = "smartest", "code"
    llm = RoutedChatOpenAI(model="auto:default", api_key="k", base_url="http://x/v1",
                           route_state=rs, route_role="orchestrator")
    payload = llm._get_request_payload([("user", "hi")])
    assert payload["model"] == "auto:smart"
    assert payload["extra_headers"]["X-FreeLLM-Task-Type"] == "code"


def test_heuristic_categories():
    assert heuristic_route("review the auth module for security issues").category == "review"
    assert heuristic_route("why does the indexer skip files?").category == "question"
    assert heuristic_route("fix the crash when chroma is down").category == "debug"


def test_decision_router_uses_gateway_answers_when_confident():
    router = DecisionRouter(DeoConfig().routing)
    answers = {"category": DecisionAnswer("tests", {"tests": 0.9}),
               "preset": DecisionAnswer("reliable", {"reliable": 0.8})}
    with patch("deo.routing.router.ask_decisions", return_value=answers):
        d = router.decide("please make this better")
    assert (d.category, d.preset, d.profile, d.source) == ("tests", "reliable", "write-tests", "decision")


def test_decision_router_falls_back_to_heuristics_on_error():
    from deo.routing.freellmapi import DecisionError
    router = DecisionRouter(DeoConfig().routing)
    with patch("deo.routing.router.ask_decisions", side_effect=DecisionError("down")):
        d = router.decide("review this diff")
    assert d.source == "heuristic" and router.decisions_ok is False


def test_subagent_freellmapi_model_strings_become_model_objects():
    from deo.agents.factory import _resolve_subagent_models, make_route_state
    config = DeoConfig()
    rs = make_route_state(config, "s")
    specs = [{"name": "reviewer-agent", "model": "freellmapi:auto:smart"}, {"name": "tester-agent"}]
    with patch.dict(os.environ, {"FREELLMAPI_API_KEY": "k"}):
        _resolve_subagent_models(config, specs, rs)
    assert isinstance(specs[0]["model"], RoutedChatOpenAI) and rs.role_models["reviewer"] == "auto:smart"
    assert specs[1]["model"].route_role == "tester"


def test_signals_and_outcome():
    s = Signals()
    s.observe("===== 12 passed in 0.3s =====")
    assert infer_outcome(s, None, edited=True) == "success"
    s.observe("FAILED tests/test_x.py::t - AssertionError\n1 failed, 11 passed")
    assert infer_outcome(s, None, edited=True) == "failure"


def test_bandit_overrides_only_with_evidence():
    b = PresetBandit(seed=1)
    assert b.choose_preset("debug", "smartest", ("smartest", "fastest"))[0] == "smartest"
    for _ in range(30):
        b.update("debug", "smartest", 0.0)
        b.update("debug", "fastest", 1.0)
    assert b.choose_preset("debug", "smartest", ("smartest", "fastest"))[0] == "fastest"


def test_session_learner_records_and_learns(tmp_path):
    learner = SessionLearner(tmp_path, seed=0)
    d = heuristic_route("add tests for config")
    learner.begin_turn("add tests for config", d, "auto:reliable")
    learner.observe_tool_call("edit_file", {"file_path": "/tests/test_config.py"})
    learner.observe_tool_output("5 passed in 0.1s")
    ep = learner.end_turn(["groq/llama"])
    assert ep.outcome == "success" and ep.files_edited == ["tests/test_config.py"]
    learner.feedback("bad", "run tests with uv run pytest")
    reloaded = SessionLearner(tmp_path)
    assert reloaded.store.load()[-1].feedback == "bad"
    assert reloaded.lessons.recall("how do I run pytest", k=1)[0].text.endswith("uv run pytest")
    assert reloaded.files.prior("tests/test_config.py") >= 0


def test_python_chunks_follow_ast_and_methods():
    src = "import os\n\n" + "class Big:\n" + "".join(
        f"    def m{i}(self):\n        return {'x' * 80!r}\n\n" for i in range(12)) + "def f():\n    pass\n"
    chunks = chunk_source(src, "python", 400, 50)
    syms = [s for c in chunks for s in c.symbols]
    assert "Big.m0" in syms and "f" in syms
    assert identifier_terms("getUserById") == ["get", "user", "by", "id"]


def test_lexical_index_bm25_and_symbols(tmp_path):
    lex = LexicalIndex(tmp_path / "lex.sqlite")
    lex.replace_file("a.py", [{"id": "1", "document": "def parse_config(path): ...", "language": "python",
                               "symbols": "parse_config", "chunk_idx": 0, "start_line": 1, "end_line": 1}])
    lex.replace_file("b.py", [{"id": "2", "document": "def render(): ...", "language": "python",
                               "symbols": "render", "chunk_idx": 0, "start_line": 1, "end_line": 1}])
    assert lex.search("where is config parsed")[0]["filepath"] == "a.py"
    assert lex.find_symbol("parse_config")[0]["filepath"] == "a.py"
    assert lex.get_chunks("a.py", [0])[0]["id"] == "1"
    lex.delete_file("a.py")
    assert lex.count() == 1
