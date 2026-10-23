"""LLM plumbing: strict JSON, one retry, router model config (plan.md 3.2.12)."""

import pytest

from hagent.config import ModelConfig
from hagent.exceptions import ModelError
from hagent.models import OpenAICompatModel, ScriptedModel

from router.llm import (LLMCallFailed, LLMFailure, LLMOutputError, LLMRetriesExhausted, RouterModels, build_model,
                        extract_json_object, load_router_models, query_json, router_models_from_dict)

MSGS = [{"role": "user", "content": "go"}]


def parse_ok(obj):
    if obj.get("ok") is not True:
        raise LLMOutputError("ok must be true")
    return obj


def test_extract_json_object():
    assert extract_json_object(' {"a": 1} ') == {"a": 1}
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('```\n{"a": 1}\n```') == {"a": 1}
    for bad in ("", "nope", '{"a": 1} trailing', 'prefix {"a": 1}', "[1, 2]", '"str"', "```json\n{bad}\n```"):
        with pytest.raises(LLMOutputError):
            extract_json_object(bad)


def test_query_json_first_try():
    m = ScriptedModel(['{"ok": true}'], tokens_per_call=(10, 5), cost_per_call=0.5)
    q = query_json(m, MSGS, parse_ok)
    assert q.value == {"ok": True} and q.attempts == 1 and q.errors == () and (q.tokens, q.cost) == (15, 0.5)


def test_query_json_retries_once_with_the_reason():
    m = ScriptedModel(["oops", '{"ok": true}'])
    q = query_json(m, MSGS, parse_ok)
    assert q.attempts == 2 and len(q.errors) == 1 and len(m.calls) == 2
    assert [x["role"] for x in m.calls[1]] == ["user", "assistant", "user"] and "rejected" in m.calls[1][-1]["content"]


def test_query_json_never_makes_a_third_call():
    m = ScriptedModel(["oops", '{"ok": false}', '{"ok": true}'])
    with pytest.raises(LLMRetriesExhausted) as e:
        query_json(m, MSGS, parse_ok)
    assert len(m.calls) == 2 and len(e.value.errors) == 2 and len(e.value.replies) == 2
    assert isinstance(e.value, LLMFailure)


def test_provider_error_is_wrapped_and_keeps_earlier_replies():
    def boom(_):
        raise ModelError("HTTP 500")
    with pytest.raises(LLMCallFailed) as e:
        query_json(ScriptedModel(["oops", boom]), MSGS, parse_ok)
    assert isinstance(e.value, ModelError) and len(e.value.errors) == 1 and len(e.value.replies) == 1


def test_router_models_config(tmp_path):
    p = tmp_path / "router.toml"
    p.write_text('[report]\nmodel = "m1"\nmax_tokens = 100\n[mapper]\nmodel = "m2"\n')
    rm = load_router_models(p)
    assert rm.report.model == "m1" and rm.report.max_tokens == 100 and rm.mapper.model == "m2"
    assert rm.factor_search is None
    assert isinstance(build_model(rm.require("report"), "report"), OpenAICompatModel)
    with pytest.raises(ValueError, match="factor_search"):
        rm.require("factor_search")


def test_router_models_config_rejects_unknown_names():
    with pytest.raises(ValueError, match="top-level"):
        router_models_from_dict({"reporter": {}})
    with pytest.raises(ValueError, match=r"\[report\]"):
        router_models_from_dict({"report": {"modle": "x"}})


def test_example_config_loads():
    import pathlib
    rm = load_router_models(pathlib.Path(__file__).parents[1] / "router_models.example.toml")
    assert all(isinstance(c, ModelConfig) for c in (rm.report, rm.mapper, rm.factor_search))
