import json
from datetime import datetime

import pytest

from agent_memory import retrieval as R
from agent_memory.analysis import answer_location, lexical_visibility
from agent_memory.datasets import data_dir, iter_json_array, iter_longmemeval, load_locomo
from agent_memory.evaluate import cluster_of, evaluate, score, split_of


def test_locomo_loader(locomo_file):
    qs = load_locomo(locomo_file)
    assert [q.qtype for q in qs] == ["temporal", "multi-hop", "abstention", "single-hop"]
    temporal, multi, adv, broken = qs
    assert temporal.evidence_turns == {"conv-x:D1:1"}
    assert multi.evidence_turns == {"conv-x:D1:1", "conv-x:D2:1"}  # "D1:1; D2:1" split
    assert multi.evidence_sessions == {"conv-x:S1", "conv-x:S2"}
    assert adv.answer.startswith("Not answerable")  # the adversarial answer is the trap
    assert broken.evidence_turns == frozenset()  # evidence pointing nowhere is dropped
    assert temporal.now == datetime(2023, 6, 20, 10)
    turns = [t for s in temporal.history for t in s.turns]
    assert "[shares a photo: a photo of runners]" in turns[1].text
    assert all(q.history is temporal.history for q in qs)


def test_longmemeval_loader(lme_file):
    ku, abst = list(iter_longmemeval(lme_file))
    assert ku.qtype == "knowledge-update" and abst.qtype == "abstention"
    assert [s.id for s in ku.history] == ["a", "c", "b"]  # sorted by date, not file order
    assert ku.evidence_turns == {"q_ku:b:0", "q_ku:a:0"}
    assert ku.now == datetime(2023, 9, 1, 10)
    assert abst.evidence_turns == frozenset()


@pytest.mark.parametrize("chunk", [1, 3, 7, 1 << 22])
def test_iter_json_array_across_chunk_boundaries(tmp_path, chunk):
    p = tmp_path / "a.json"
    items = [{"a": 1, "s": "café \U0001f600"}, {"b": [2, 3], "c": {"d": "]"}}]
    p.write_text(" \n[ " + " ,\n ".join(json.dumps(i) for i in items) + " ]\n", "utf-8")
    assert list(iter_json_array(p, chunk)) == items
    p.write_text("[]", "utf-8")
    assert list(iter_json_array(p, chunk)) == []
    for broken in ('[{"a": 1},', '[{"a": 1', "no array"):
        p.write_text(broken, "utf-8")
        with pytest.raises(ValueError):
            list(iter_json_array(p, chunk))


def test_missing_data_points_at_the_fetch_script(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_DATA", str(tmp_path))
    assert data_dir() == tmp_path
    with pytest.raises(FileNotFoundError, match=r"fetch_data\.py"):
        load_locomo()


def test_score_recall_all_and_knowledge_update(lme_file):
    ku, abst = list(iter_longmemeval(lme_file))
    assert score(abst, frozenset()) is None
    only_old = score(ku, frozenset({"q_ku:a:0"}))
    assert (only_old.recall, only_old.all, only_old.newest, only_old.stale_only) == (
        0.5,
        False,
        False,
        True,
    )
    both = score(ku, frozenset({"q_ku:a:0", "q_ku:b:0", "q_ku:c:0"}))
    assert (both.recall, both.all, both.newest, both.stale_only) == (1.0, True, True, False)
    none = score(ku, frozenset())
    assert (none.recall, none.newest, none.stale_only) == (0.0, False, False)


def test_splits_are_stable_and_by_conversation(locomo_file, lme_file):
    qs = load_locomo(locomo_file)
    assert len({split_of(q) for q in qs}) == 1  # one conversation never straddles splits
    assert {cluster_of(q) for q in qs} == {"conv-x"}
    ku, _ = list(iter_longmemeval(lme_file))
    assert split_of(ku) == split_of(ku) and cluster_of(ku) == "q_ku"


def test_evaluate_rows(locomo_file):
    qs = load_locomo(locomo_file)
    rows = list(
        evaluate(
            qs, {"bm25": R.bm25_turns, "window": R.sliding_window}, budgets=(20, 10_000), top_k=(1,)
        )
    )
    assert len(rows) == len(qs) * 2 * 3
    full = [r for r in rows if r.get("budget") == 10_000 and r["measurable"]]
    assert all(r["tokens"] <= 10_000 for r in rows)
    window_full = [r for r in full if r["strategy"] == "window"]
    assert all(r["recall"] == 1.0 for r in window_full)  # everything fits: all evidence found
    assert sum(not r["measurable"] for r in rows) == 6  # the broken-evidence question


def test_diagnostics(locomo_file):
    temporal, multi, adv, broken = load_locomo(locomo_file)
    assert answer_location(temporal) == "needs_date"  # "7 May 2023" vs "yesterday"
    assert answer_location(multi) == "in_text"  # "a bowl" -> "bowl"
    assert answer_location(broken) is None
    assert lexical_visibility(temporal) == 1.0  # shares "pottery"/"class"
    assert lexical_visibility(adv) == 1.0  # "race"


def test_malformed_locomo_evidence_ids_are_normalised():
    from agent_memory.datasets import _locomo_evidence

    got = _locomo_evidence(["D:11:26", "D30:05", "D", "D9:1 D4:4", "D8:6; D9:17"])
    assert got == ["D11:26", "D30:5", "D9:1", "D4:4", "D8:6", "D9:17"]
