"""Stage 1 check 2: observation contract (plan.md 3.2.4)."""

import pytest

from router import vocab as V
from router.prereq.trace import TraceUnavailable
from router.schemas import Mode
from router.stage1.nominal import MissingNominal, NominalStats
from router.stage1.obs_contract import (M_OPEN_LOOP_RATIO, ObservationContractCheck, OpenLoopProbe, PreprocessingPath,
                                        PolicyOpenLoopProbe)
from router.stage1.synthetic import CAMERAS, Faults, make_incident, make_nominal, make_source

NOMINAL = make_nominal()
INCIDENT = make_incident()


def check(faults=Faults(), probe=None, **kw):
    return ObservationContractCheck(NOMINAL, make_source(faults), CAMERAS, probe=probe, **kw)


def ids(found):
    return sorted(e.id for e in found)


def test_clean_traces_have_no_evidence():
    assert check().run(INCIDENT) == []


def test_clean_traces_stay_clean_across_many_seeds():
    for base in range(5):
        src = make_source(Faults(), n=6)
        for traces in src._by_ref.values():          # reshuffle which episodes are used
            traces[:] = traces[base % 3:] or traces
        assert ObservationContractCheck(NOMINAL, src, CAMERAS).run(INCIDENT) == []


def test_channel_swap_is_found_and_named():
    found = check(Faults(swap_rb=True)).run(INCIDENT)
    assert "s1.obs.channel_order.front" in ids(found)
    e = next(e for e in found if e.id == "s1.obs.channel_order.front")
    assert e.source == "contract_obs" and e.signal == V.SIG_CONTRACT_VIOLATION
    assert e.location == "obs.channel_order" and "BGR" in e.detail and "RGB" in e.detail
    assert 0.5 <= e.strength <= 1.0
    # specific: a channel swap does not look like a range or orientation fault
    assert not any(i.startswith(("s1.obs.value_range", "s1.obs.orientation", "s1.obs.image_size")) for i in ids(found))


@pytest.mark.parametrize("faults, words", [
    (Faults(vflip=True), "vertical flip"),
    (Faults(hflip=True), "horizontal flip"),
    (Faults(rot180=True), "180"),
])
def test_orientation_faults_are_found_and_named(faults, words):
    found = check(faults).run(INCIDENT)
    e = next(e for e in found if e.id == "s1.obs.orientation.front")
    assert e.location == "obs.orientation" and words in e.detail and e.signal == V.SIG_CONTRACT_VIOLATION
    assert not any("channel_order" in i or "value_range" in i for i in ids(found))


def test_unnormalized_values_are_a_range_violation():
    found = check(Faults(unnormalized=True)).run(INCIDENT)
    e = next(e for e in found if e.id == "s1.obs.value_range.front")
    assert e.location == "obs.normalization" and e.strength > 0.99
    assert not any("channel_order" in i or "orientation" in i for i in ids(found))


def test_wrong_resolution_is_exact_and_full_strength():
    found = check(Faults(wrong_resolution=True)).run(INCIDENT)
    e = next(e for e in found if e.id == "s1.obs.image_size")
    assert e.strength == 1.0 and e.location == "obs.image_size" and "(240, 320)" in e.detail
    assert "(224, 224)" in e.detail


def test_only_the_planted_cameras_fields_fire_per_fault():
    # every fault leaves the other sub-checks quiet
    for faults, prefix in [(Faults(swap_rb=True), "s1.obs.channel_order"), (Faults(vflip=True), "s1.obs.orientation"),
                           (Faults(unnormalized=True), "s1.obs.value_range")]:
        assert {i.rsplit(".", 1)[0] for i in ids(check(faults).run(INCIDENT))} == {prefix}


class FixedProbe(OpenLoopProbe):
    def __init__(self, current, reference):
        self.errors = {PreprocessingPath.CURRENT: current, PreprocessingPath.REFERENCE: reference}
        self.calls = []

    def action_error(self, incident, path):
        self.calls.append(path)
        return self.errors[path]


def test_open_loop_probe_flags_a_broken_observation_path():
    probe = FixedProbe(current=0.9, reference=0.1)                              # 9x the reference error
    found = check(probe=probe).run(INCIDENT)
    e = next(e for e in found if e.id == "s1.obs.open_loop")
    assert e.location == "obs.preprocessing" and e.signal == V.SIG_CONTRACT_VIOLATION and "9.00x" in e.detail
    assert set(probe.calls) == {PreprocessingPath.CURRENT, PreprocessingPath.REFERENCE}


def test_open_loop_probe_quiet_when_preprocessing_matches_reference():
    assert check(probe=FixedProbe(current=0.105, reference=0.10)).run(INCIDENT) == []
    assert check(probe=FixedProbe(current=0.05, reference=0.10)).run(INCIDENT) == []   # better is not a fault
    with pytest.raises(ValueError):
        check(probe=FixedProbe(current=0.1, reference=0.0)).run(INCIDENT)


def test_open_loop_probe_is_optional_and_the_real_one_is_a_stub():
    assert not any(e.id == "s1.obs.open_loop" for e in check().run(INCIDENT))
    with pytest.raises(NotImplementedError):
        PolicyOpenLoopProbe().action_error(INCIDENT, PreprocessingPath.CURRENT)


def test_missing_traces_raise_instead_of_reading_as_clean():
    with pytest.raises(TraceUnavailable):
        check().run(make_incident(log_refs={}))


def test_check_is_built_with_all_its_nominal_inputs_or_not_at_all():
    bare = NominalStats({}, fire_z=4.0)
    with pytest.raises(MissingNominal):
        ObservationContractCheck(bare, make_source(), CAMERAS)
    with pytest.raises(MissingNominal):
        ObservationContractCheck(NOMINAL, make_source(), ("front", "overhead"))
    no_ratio = NominalStats({k: v for k, v in NOMINAL.metrics.items() if k != M_OPEN_LOOP_RATIO}, NOMINAL.vectors)
    ObservationContractCheck(no_ratio, make_source(), CAMERAS)                          # fine without a probe
    with pytest.raises(MissingNominal):
        ObservationContractCheck(no_ratio, make_source(), CAMERAS, probe=FixedProbe(1, 1))


def test_reads_the_requested_inference_mode():
    src = make_source(clean=Faults(), incident=Faults(vflip=True))
    inc = make_incident()
    assert ObservationContractCheck(NOMINAL, src, CAMERAS, mode=Mode.CLEAN).run(inc) == []
    assert ObservationContractCheck(NOMINAL, src, CAMERAS, mode=Mode.INCIDENT).run(inc) != []


# --- partial skips (ISSUES #59, F3 review): a sub-check that cannot run says so -------------------

from dataclasses import replace as _replace


def run_partial(spec_edit=None, probe=FixedProbe(1.0, 1.0)):
    incident = INCIDENT if spec_edit is None else _replace(INCIDENT, training_spec=_replace(INCIDENT.training_spec, **spec_edit))
    return check(probe=probe).run_partial(incident)


def test_a_complete_spec_and_a_probe_give_no_partial_skip():
    assert run_partial()[1] == []


@pytest.mark.parametrize("edit,words", [({"image_size": None}, "image size not checked"),
                                        ({"channel_order": None}, "channel order not checked")])
def test_a_missing_spec_field_is_a_partial_skip_on_contract_violation(edit, words):
    found, skips = run_partial(edit)
    assert [s.signals for s in skips] == [{V.SIG_CONTRACT_VIOLATION}] and words in skips[0].reason


def test_no_open_loop_probe_is_a_note_not_a_partial_skip():
    c = check(probe=None)
    assert c.run_partial(INCIDENT)[1] == []
    assert c.notes(INCIDENT) == ["open-loop probe not configured"]
    assert check(probe=FixedProbe(1.0, 1.0)).notes(INCIDENT) == []


def test_no_probe_with_a_missing_image_size_is_still_a_partial_skip():
    found, skips = run_partial({"image_size": None}, probe=None)
    assert [s.signals for s in skips] == [{V.SIG_CONTRACT_VIOLATION}] and "image size" in skips[0].reason
