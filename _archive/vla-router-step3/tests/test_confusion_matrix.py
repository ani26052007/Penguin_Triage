"""The confusion-matrix script runs on the 11 mocked fault classes (plan.md 6.5)."""

from __future__ import annotations

from router.schemas import VerdictLabel
from scripts.confusion_matrix import build_cases, format_matrix, main, run_matrix, summarize


def test_eleven_fault_classes_cover_every_non_degenerate_label():
    cases = build_cases()
    assert len(cases) == 11 and len({c.name for c in cases}) == 11
    assert {c.expected for c in cases} == set(VerdictLabel) - {VerdictLabel.PERCEPTION_UNRESOLVED, VerdictLabel.UNDETERMINED}


def test_matrix_is_mostly_diagonal_with_sampling_noise_reported():
    results = run_matrix(build_cases(), n_seeds=30, repeats=10)
    s = summarize(results)
    assert s["total"] == 110
    assert s["accuracy_when_committing"] >= 0.95
    assert s["undetermined_rate"] <= 0.10
    for r in results:
        assert r.correct >= 8, f"{r.case.name}: {dict(r.labels())}"      # noise may cost a few repeats
    text = format_matrix(results)
    assert all(r.case.name in text for r in results)


def test_noise_free_classes_are_exact_and_carry_the_right_flags():
    results = {r.case.name: r for r in run_matrix(build_cases(), n_seeds=30, repeats=10)}
    for name in ("no_fault", "vision_glossy", "rendering_mismatch", "sim_asset", "infra_stale_server",
                 "code_global", "code_condition_specific", "code_partial_offset"):
        assert results[name].correct == 10, name
        assert results[name].flag_mismatches() == [], name


def test_main_runs(capsys):
    assert main(["--seeds", "20", "--repeats", "2"]) == 0
    assert "fault class" in capsys.readouterr().out
