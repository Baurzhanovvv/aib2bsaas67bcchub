"""Сборка отчёта: итог, fix_first, паспорт, сводка."""

import pytest

from reqcheck import version
from reqcheck.llm import LLMUsage
from reqcheck.models import (
    Document,
    DocumentFormat,
    Finding,
    GroundingResult,
    RunMode,
    Severity,
    Verdict,
)
from reqcheck.stages.report import build_report, decide_verdict
from tests.factories import SHA, make_document, make_finding, make_requirement

USAGE = LLMUsage(model="test-model", requests=4, cache_hits=1, tokens_in=100, tokens_out=50)


def finding(fid: str, severity: Severity) -> Finding:
    return make_finding(fid).model_copy(update={"severity": severity})


def build(
    findings: tuple[Finding, ...], usage: LLMUsage | None = USAGE, doc: Document | None = None
):
    grounded = GroundingResult(
        findings=findings, quotes_checked=5, quotes_confirmed=4, findings_dropped=2
    )
    return build_report(doc or make_document(), (make_requirement(),), grounded, None, usage, 1.5)


@pytest.mark.parametrize(
    ("severities", "expected"),
    [
        ((), Verdict.GO),
        ((Severity.MAJOR, Severity.MINOR), Verdict.GO),
        ((Severity.CRITICAL,), Verdict.CONDITIONAL_GO),
        ((Severity.CRITICAL, Severity.MAJOR), Verdict.CONDITIONAL_GO),
        ((Severity.BLOCKER,), Verdict.NO_GO),
        ((Severity.BLOCKER, Severity.CRITICAL), Verdict.NO_GO),
    ],
)
def test_verdict_full_run(severities: tuple[Severity, ...], expected: Verdict) -> None:
    findings = tuple(finding(f"D-{i:03d}", s) for i, s in enumerate(severities, 1))
    assert build(findings).verdict is expected


def test_rules_only_clean_is_not_go() -> None:
    report = build((), usage=None)
    assert report.verdict is Verdict.CONDITIONAL_GO
    assert report.set_check_done is False


def test_rules_only_blocker_stays_no_go() -> None:
    assert build((finding("D-001", Severity.BLOCKER),), usage=None).verdict is Verdict.NO_GO


def test_unread_pages_cap_go_to_conditional() -> None:
    doc = make_document().model_copy(update={"unread_pages": (4,)})
    assert build((), doc=doc).verdict is Verdict.CONDITIONAL_GO
    assert build((finding("D-001", Severity.BLOCKER),), doc=doc).verdict is Verdict.NO_GO


def test_decide_verdict_incomplete_flag() -> None:
    assert decide_verdict((), complete=True) is Verdict.GO
    assert decide_verdict((), complete=False) is Verdict.CONDITIONAL_GO


def test_fix_first_blockers_then_criticals_by_id_max_seven() -> None:
    findings = (
        finding("D-001", Severity.CRITICAL),
        finding("D-002", Severity.MAJOR),
        finding("D-003", Severity.BLOCKER),
        finding("D-004", Severity.CRITICAL),
        finding("D-005", Severity.BLOCKER),
    )
    assert build(findings).fix_first == ("D-003", "D-005", "D-001", "D-004")
    many = tuple(finding(f"D-{i:03d}", Severity.CRITICAL) for i in range(1, 12))
    assert build(many).fix_first == tuple(f"D-{i:03d}" for i in range(1, 8))


def test_summary_counts() -> None:
    findings = (
        finding("D-001", Severity.CRITICAL),
        finding("D-002", Severity.MINOR),
        finding("D-003", Severity.BLOCKER),
    )
    summary = build(findings).summary
    assert summary.requirements_total == 1
    assert summary.findings_total == 3
    assert summary.by_severity[Severity.MINOR] == 1
    assert summary.by_severity[Severity.MAJOR] == 0
    assert summary.critical_finding_ids == ("D-001", "D-003")


def test_passport_full_mode() -> None:
    p = build(()).passport
    assert p.mode is RunMode.FULL
    assert (p.model, p.llm_requests, p.cache_hits, p.tokens_in, p.tokens_out) == (
        "test-model",
        4,
        1,
        100,
        50,
    )
    assert (p.quotes_checked, p.quotes_confirmed, p.findings_dropped) == (5, 4, 2)
    assert p.duration_s == 1.5
    assert p.run_at.tzinfo is not None
    assert p.document_sha256 == SHA
    assert p.agent_version


def test_passport_without_usage() -> None:
    p = build((), usage=None).passport
    assert p.mode is RunMode.RULES_ONLY
    assert p.model is None
    assert p.llm_requests == 0


def test_passport_unread_pages() -> None:
    doc = Document(
        name="a.pdf", sha256=SHA, format=DocumentFormat.PDF, fragments=(), unread_pages=(2, 5)
    )
    assert build((), doc=doc).passport.unread_pages == (2, 5)


def test_version_falls_back_outside_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(version, "_git", lambda *args: None)
    assert version.agent_version() == (version.__version__, False)


def test_version_dirty_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = {"describe": "v1.2-3-gabc1234", "status": " M file.py"}
    monkeypatch.setattr(version, "_git", lambda *args: answers[args[0]])
    assert version.agent_version() == ("v1.2-3-gabc1234", True)
    answers["status"] = ""
    assert version.agent_version() == ("v1.2-3-gabc1234", False)


def test_model_error_branch() -> None:
    grounded = GroundingResult()
    report = build_report(
        make_document(), (make_requirement(),), grounded, None, USAGE, 1.0, model_error="квота"
    )
    assert report.passport.model_error == "квота"
    assert report.passport.mode is RunMode.RULES_ONLY
    assert report.passport.model == "test-model"
    assert report.passport.llm_requests == 4
    assert report.set_check_done is False
    assert report.verdict is Verdict.CONDITIONAL_GO


def test_model_error_keeps_no_go() -> None:
    grounded = GroundingResult(findings=(finding("D-001", Severity.BLOCKER),))
    report = build_report(
        make_document(), (make_requirement(),), grounded, None, USAGE, 1.0, model_error="сеть"
    )
    assert report.verdict is Verdict.NO_GO


def test_set_check_requires_trace() -> None:
    assert build(()).set_check_done is False  # trace=None
    from reqcheck.models import TraceReport

    report = build_report(
        make_document(), (make_requirement(),), GroundingResult(), TraceReport(), USAGE, 1.0
    )
    assert report.set_check_done is True
