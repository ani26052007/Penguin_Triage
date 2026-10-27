"""The source contract suite on the router's in-memory sources and on the Isaac sources."""

from __future__ import annotations

from dataclasses import replace

import pytest

from router.behavior_synthetic import BehaviorFaults, make_behavior_traces
from router.prereq.behavior_trace import BehaviorSource, InMemoryBehaviorSource
from router.prereq.trace import InMemoryTraceSource, TraceSource
from router.schemas import Mode
from router.stage1.synthetic import default_code_key, make_incident, make_traces

from vla_isaac.traces.sources import IsaacBehaviorSource, IsaacTraceSource

from tests.contracts.source_contract import SourceUnderTest, TraceSourceContract
from tests.test_traces import LAYOUT, traced
from tests.trace_fixtures import PW

needs_pw = pytest.mark.skipif(not (PW / "policy_wrapper" / "instrument.py").is_file(),
                              reason=f"no policy-wrapper checkout at {PW}")


class TestInMemoryTraceSourceContract(TraceSourceContract):
    def make(self, tmp_path):
        src = InMemoryTraceSource({"clean": make_traces(4, Mode.CLEAN), "incident": make_traces(4, Mode.INCIDENT)})
        inc = make_incident()
        return SourceUnderTest(src, TraceSource, inc, Mode.CLEAN,
                               (replace(inc, log_refs={"incident": "clean"}), Mode.INCIDENT))


class TestInMemoryBehaviorSourceContract(TraceSourceContract):
    def make(self, tmp_path):
        f = BehaviorFaults(fail_phase="grasp")
        key = default_code_key()
        src = InMemoryBehaviorSource({"clean": make_behavior_traces(4, Mode.CLEAN, f, code_key=key),
                                      "incident": make_behavior_traces(4, Mode.INCIDENT, f, code_key=key)})
        inc = make_incident()
        return SourceUnderTest(src, BehaviorSource, inc, Mode.CLEAN,
                               (replace(inc, log_refs={"incident": "clean"}), Mode.INCIDENT))


def _isaac(tmp_path, source, base) -> SourceUnderTest:
    _, _, _, inc = traced(tmp_path, seeds=range(4))
    return SourceUnderTest(source, base, inc, Mode.CLEAN,
                           (replace(inc, log_refs={"incident": inc.log_refs["clean"]}), Mode.INCIDENT))


@needs_pw
class TestIsaacTraceSourceContract(TraceSourceContract):
    def make(self, tmp_path):
        return _isaac(tmp_path, IsaacTraceSource(LAYOUT), TraceSource)


@needs_pw
class TestIsaacBehaviorSourceContract(TraceSourceContract):
    def make(self, tmp_path):
        return _isaac(tmp_path, IsaacBehaviorSource(LAYOUT), BehaviorSource)
