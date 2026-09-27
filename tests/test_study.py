"""Every study stage and the model arm, end to end on the fixture benchmarks, with a
fake model. Checks that each stage runs and writes the fields the README quotes."""

import json

import pytest

from agent_memory.llm import FakeLLM
from agent_memory.study import sweep
from agent_memory.study.common import Run
from agent_memory.study.diagnostics import diagnostics_stage
from agent_memory.study.forgetting import forgetting_stage
from agent_memory.study.models import dry_run, stage_answer, stage_extract, stage_retrieval
from agent_memory.study.recall import main_stage


@pytest.fixture
def run(tmp_path, locomo_file, lme_file, monkeypatch):
    assert locomo_file.parent == lme_file.parent
    monkeypatch.setenv("AGENT_MEMORY_DATA", str(locomo_file.parent))
    return Run(tmp_path / "results")


def read(run, name):
    return json.loads((run.results / name).read_text("utf-8"))


def test_retrieval_study_stages(run, monkeypatch):
    monkeypatch.setattr(sweep, "split_of", lambda q: "dev")  # the fixtures are tiny
    monkeypatch.setattr(sweep, "GRID", {"session_weight": (0.0, 0.3), "neighbours": (0,)})
    monkeypatch.setattr(sweep, "WINDOW_SHARES", (0.1, 0.5))
    sweep.dev_sweep(run)
    dev = read(run, "dev_sweep.json")
    assert dev["store_best"]["config"]["neighbours"] == 0
    assert dev["window_share_best"]["recent_share"] in (0.1, 0.5)
    main_stage(run)
    out = read(run, "retrieval.json")
    strategies = {r["strategy"] for r in out["by_budget"]}
    assert {"sliding_window", "window_bm25", "store", "random"} <= strategies
    assert any("session_recall" in r for r in out["by_budget"])
    assert any("recall_strict" in r for r in out["by_budget"])
    assert "window_bm25" not in {r["strategy"] for r in out["by_k"]}  # no k for split plans
    assert out["by_evidence_position"] and "subsets" in out
    assert (run.results / "rows" / "locomo.jsonl.gz").exists()
    forgetting_stage(run)
    assert {p["policy"] for p in read(run, "forgetting.json")["policies"]} >= {"none", "phatic:3"}
    diagnostics_stage(run)
    diag = {d["dataset"]: d for d in read(run, "diagnostics.json")}
    assert diag["longmemeval"]["histories"] == 3
    assert diag["longmemeval"]["data_quality_flags"]["partial_key"]["all types"] == 1
    assert diag["locomo"]["evidence_id_audit"]["unresolvable"] == 1
    assert diag["locomo"]["counts"]["questions_without_evidence_turns"] == 1
    assert diag["longmemeval"]["counts"]["empty_turns_skipped_at_ingest"] == 0
    out = read(run, "retrieval.json")
    assert {d["credit"] for d in out["by_evidence_position_differences"]} == {"lenient", "strict"}
    assert out["subsets"]["turn_key_complete"]["comparisons"]
    late = diag["longmemeval"]["sessions_dated_after_the_question"]
    assert late["questions_with_future_sessions"] == 1 and late["future_answer_sessions"] == 0
    vis = diag["locomo"]["lexical_visibility"]["all types"]
    assert set(vis) == {
        "evidence_turns", "evidence_turns_sharing_no_word", "questions",
        "questions_with_no_visible_evidence",
    }  # fmt: skip
    sweep.window_share_sensitivity(run)
    assert read(run, "window_share.json")["versus_bm25_turns"]
    sweep.recency_sensitivity(run)
    assert read(run, "recency_sensitivity.json")["split"] == "test"


def test_model_arm_stages_with_a_fake(run, tmp_path, monkeypatch):
    run.results.mkdir(parents=True)
    (run.results / "dev_sweep.json").write_text(
        json.dumps(
            {
                "store_best": {"config": {}},
                "recency_best": {"half_life_days": 365.0},
                "window_share_best": {"recent_share": 0.1},
            }
        )
    )
    fake = FakeLLM({"Is the response correct?": "yes"}, default="[]")
    stage_retrieval(run, fake)
    assert read(run, "model_retrieval.json")["by_budget"]
    stores = tmp_path / "stores"
    stage_extract(run, fake, stores)
    assert (stores / "locomo_conv-x.db").exists()
    monkeypatch.setattr("agent_memory.study.models.split_of", lambda q: "test")
    stage_answer(run, fake, stores, per_type=5, judge="judge-model")
    answers = read(run, "model_answers.json")
    assert answers["self_judged"] is False and answers["rows"]
    assert all(r["unparsed"] == 0 for r in answers["summary"])
    lines = dry_run(run, per_type=5)
    assert lines[-1].startswith("total:")
