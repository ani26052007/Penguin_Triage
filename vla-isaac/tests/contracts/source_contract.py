"""The ``TraceSource`` / ``BehaviorSource`` contract as a reusable suite (IMPL_PLAN D1).

vla-router states the contract in ``R:router/prereq/trace.py`` and ``behavior_trace.py`` (template
``load`` that a subclass never overrides; every trace logged with ``incident.code_ref``; a missing log
is ``TraceUnavailable``, never "checked, nothing found") and checks it on its in-memory sources
(``R:tests/test_trace_code_key.py``). This suite turns it into one class that any source's test
subclasses; ``tests/test_source_contract.py`` runs it on the router's ``InMemory*Source`` (to show it is
the same contract) and on ``IsaacTraceSource`` / ``IsaacBehaviorSource`` over traced fake runs.

A subclass implements ``make(tmp_path) -> SourceUnderTest``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from router.prereq.behavior_trace import BehaviorSource
from router.prereq.trace import TraceSource, TraceUnavailable
from router.schemas import CodeRef, Incident, Mode


@dataclass
class SourceUnderTest:
    source: TraceSource | BehaviorSource
    base: type                           # TraceSource or BehaviorSource
    incident: Incident                   # log_refs[mode.value] points at logs of incident.code_ref
    mode: Mode
    wrong_mode: tuple[Incident, Mode]    # a reference to logs of the other mode: a ValueError


class TraceSourceContract:
    def make(self, tmp_path) -> SourceUnderTest:             # pragma: no cover - abstract
        raise NotImplementedError

    def test_load_is_the_template_and_is_not_overridden(self, tmp_path):
        s = self.make(tmp_path)
        assert isinstance(s.source, s.base) and type(s.source).load is s.base.load
        assert type(s.source)._load is not s.base._load

    def test_traces_are_in_the_mode_asked_for_and_carry_the_code_key(self, tmp_path):
        s = self.make(tmp_path)
        traces = s.source.load(s.incident, s.mode)
        assert traces and all(t.mode is s.mode for t in traces)
        assert {t.code_key for t in traces} == {s.incident.code_ref.key()}
        assert len({t.seed for t in traces}) == len(traces)

    def test_loading_twice_gives_the_same_traces_and_they_round_trip(self, tmp_path):
        s = self.make(tmp_path)
        first, again = s.source.load(s.incident, s.mode), s.source.load(s.incident, s.mode)
        assert first == again
        assert [type(t).from_dict(t.to_dict()) for t in first] == first

    def test_logs_of_other_code_are_refused(self, tmp_path):
        s = self.make(tmp_path)
        other = replace(s.incident, code_ref=CodeRef("other-code", {"server": {"denoising_steps": 99}}))
        with pytest.raises(TraceUnavailable):
            s.source.load(other, s.mode)

    def test_a_missing_reference_is_unavailable_not_clean(self, tmp_path):
        s = self.make(tmp_path)
        with pytest.raises(TraceUnavailable):
            s.source.load(replace(s.incident, log_refs={}), s.mode)

    def test_a_reference_to_the_other_modes_logs_is_an_error(self, tmp_path):
        s = self.make(tmp_path)
        incident, mode = s.wrong_mode
        with pytest.raises(ValueError):
            s.source.load(incident, mode)
