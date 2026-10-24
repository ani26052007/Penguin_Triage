"""The code key of logged traces (ISSUES #52): a source serves only logs of the code under test.

``TraceSource.load`` / ``BehaviorSource.load`` are template methods that check ``code_key``; a source
implements ``_load``. A source that returns the original code's logs for a patched incident, or logs
that carry no key, raises ``TraceUnavailable`` (a recorded skip), so a fixed fault cannot keep firing.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import replace

import pytest

import router
import tests
from router.behavior_synthetic import (CodeKeyedBehaviorSource, make_behavior_source,
                                       make_behavior_traces)
from router.prereq.behavior_trace import BehaviorSource, InMemoryBehaviorSource
from router.prereq.trace import InMemoryTraceSource, TraceSource, TraceUnavailable
from router.router import route
from router.schemas import CodeRef, Mode
from router.stage1.synthetic import CodeKeyedTraceSource, make_incident, make_source, make_trace, make_traces
from tests.router_cases import default_config, router_cases

CASES = {c.name: c for c in router_cases()}


def _import_all(package) -> None:
    for m in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(m.name)


def _subclasses(cls) -> set:
    out = set()
    for sub in cls.__subclasses__():
        out |= {sub} | _subclasses(sub)
    return out


def test_no_source_overrides_load():
    """Structural: ``load`` is the template method; every source (library and test doubles) implements ``_load``."""
    _import_all(router)
    _import_all(tests)
    for base in (TraceSource, BehaviorSource):
        subs = _subclasses(base)
        assert subs >= ({InMemoryTraceSource, CodeKeyedTraceSource} if base is TraceSource
                        else {InMemoryBehaviorSource, CodeKeyedBehaviorSource})
        for cls in subs:
            assert "load" not in vars(cls), f"{cls.__module__}.{cls.__qualname__} overrides load(); implement _load()"
            assert "_load" in vars(cls) or any("_load" in vars(k) for k in cls.__mro__[1:-1] if k is not base), cls
        assert base.__dict__["_load"].__isabstractmethod__


def test_traces_of_other_code_are_unavailable():
    inc = make_incident()
    src = make_source(code_key="c9+cfg:other")
    with pytest.raises(TraceUnavailable, match="not for the code under test"):
        src.load(inc, Mode.CLEAN)
    assert len(make_source(code_key=inc.code_ref.key()).load(inc, Mode.CLEAN)) == 8


def test_traces_without_a_key_are_unavailable_not_trusted():
    inc = make_incident()
    src = InMemoryTraceSource({"clean": [make_trace(i, Mode.CLEAN) for i in range(4)]})
    with pytest.raises(TraceUnavailable, match="no code key"):
        src.load(inc, Mode.CLEAN)
    mixed = InMemoryTraceSource({"clean": make_traces(3, Mode.CLEAN) + [make_trace(9, Mode.CLEAN)]})
    with pytest.raises(TraceUnavailable):
        mixed.load(inc, Mode.CLEAN)


def test_behavior_source_checks_the_key_too():
    inc = make_incident()
    with pytest.raises(TraceUnavailable, match="not for the code under test"):
        make_behavior_source(code_key="c9+cfg:other").load(inc, Mode.CLEAN)
    bare = InMemoryBehaviorSource({"clean": make_behavior_traces(4, Mode.CLEAN)})
    with pytest.raises(TraceUnavailable, match="no code key"):
        bare.load(inc, Mode.CLEAN)
    assert make_behavior_source().load(inc, Mode.CLEAN)


def test_code_key_round_trips_through_json():
    t = make_trace(1, Mode.CLEAN, code_key="k")
    assert type(t).from_dict(t.to_dict()).code_key == "k"
    b = make_behavior_traces(1, code_key="k")[0]
    assert type(b).from_dict(b.to_dict()).code_key == "k"


def test_a_source_that_keeps_serving_the_original_logs_cannot_make_the_fix_look_unfixed():
    """After a patch the incident's code key changes. A source that returns the ORIGINAL code's logs
    (here: planted unnormalised inputs) is refused, so the old fault does not fire for the new code;
    the checks are recorded as skipped instead."""
    case = CASES["combined_code_plus_light"]
    original = case.incident()
    patched = replace(original, code_ref=CodeRef("c1", {}))

    stale = case.inputs()                          # logs keyed on the original code: unnormalised inputs
    ok = route(original, case.quota_backend(), default_config(), stale)
    assert any(e.source == "contract_obs" for e in ok.stage1_evidence)        # the planted fault is seen on c0

    class Replaying(CodeKeyedTraceSource):         # ignores the incident's code: always the original's logs
        def _load(self, incident, mode):
            return super()._load(original, mode)

    res = route(patched, case.quota_backend(), default_config(),
                replace(stale, stage1_checks=tuple(_with_source(c, Replaying(case.trace_faults, case.trace_default))
                                                   for c in stale.stage1_checks)))
    skipped = {s for s, _ in res.skipped}
    assert {"contract_obs", "contract_action", "staleness", "sanity"} <= skipped
    assert not [e for e in res.stage1_evidence if e.source in skipped]
    assert "not for the code under test" in res.verdict.notes


def _with_source(check, source):
    if hasattr(check, "traces"):
        import copy
        c = copy.copy(check)
        c.traces = source
        return c
    return check
