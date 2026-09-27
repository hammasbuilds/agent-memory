"""The model arm, exercised with a deterministic fake and a local stub HTTP server -
no Ollama, no network, no model."""

import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from agent_memory.answer import answer_and_judge, answer_prompt, judge_prompt, parse_verdict
from agent_memory.context import Context
from agent_memory.datasets import iter_longmemeval
from agent_memory.embedding import make_dense, make_hybrid
from agent_memory.extract import build_prompt, extract_into, parse
from agent_memory.llm import DiskCache, FakeLLM, Ollama, OllamaError
from agent_memory.retrieval import History
from agent_memory.store import MemoryStore

NOW = datetime(2024, 7, 1)


def test_parse_tolerates_fences_and_drops_bad_items():
    reply = """Here you go:
```json
[{"subject": "user", "attribute": "employer", "value": "Riverside", "turn": 0},
 {"subject": "user", "attribute": "", "value": "x", "turn": 0},
 {"subject": "user", "attribute": "pet", "value": "dog", "turn": 99},
 {"subject": "user", "attribute": "age", "value": 34, "turn": 1},
 "junk"]
```"""
    got = parse(reply, n_turns=2)
    assert [(f.attribute, f.value, f.turn) for f in got] == [
        ("employer", "Riverside", 0),
        ("age", "34", 1),
    ]
    assert parse("no json here", 3) == []
    assert parse("[not valid json]", 3) == []
    assert parse('{"subject": "a"}', 3) == []


def test_extract_into_supersedes_across_sessions(history):
    fake = FakeLLM(
        {
            "Riverside": '[{"subject": "user", "attribute": "employer", "value": "Riverside Hospital", "turn": 0}]',
            "St Mary": '[{"subject": "user", "attribute": "employer", "value": "St Mary\'s clinic", "turn": 0}]',
        },
        default="[]",
    )
    with MemoryStore() as m:
        m.ingest(history)
        counts = extract_into(m, history, fake, model="fake")
        assert counts == {"sessions": 3, "facts_parsed": 2, "facts_stored": 2}
        versions = m.history_of("user", "employer")
        assert [f.value for f in versions] == ["Riverside Hospital", "St Mary's clinic"]
        assert versions[1].source_turn == "s3:0" and versions[1].extractor == "fake"
    assert "2024-01-10" in build_prompt(history[0])


def test_answer_and_judge_prompts(lme_file):
    ku, abst, _ = list(iter_longmemeval(lme_file))
    ctx = Context((), 0, 100)
    assert "(nothing retrieved)" in answer_prompt(ku, ctx)
    assert "2023-09-01" in answer_prompt(ku, ctx)
    assert "updated value" in judge_prompt(ku, "St Mary's")
    assert "not available" in judge_prompt(abst, "I don't know")
    fake = FakeLLM({"Is the response correct?": "Yes."}, default="St Mary's")
    out = answer_and_judge(ku, ctx, fake)
    assert (out["response"], out["correct"]) == ("St Mary's", True)
    fake.calls.clear()
    answer_and_judge(ku, ctx, fake, model="a", judge="b")
    assert len(fake.calls) == 2
    shrug = FakeLLM({"Is the response correct?": "It depends."}, default="St Mary's")
    assert answer_and_judge(ku, ctx, shrug)["correct"] is None


def test_unparsed_verdicts_count_as_wrong_and_are_reported():
    from agent_memory.report import summarise_answers

    rows = [
        {
            "dataset": "d",
            "strategy": "s",
            "qtype": "t",
            "cluster": str(i),
            "recall": 1.0,
            "correct": c,
        }
        for i, c in enumerate([True, True, None, False])
    ]
    rec = next(r for r in summarise_answers(rows, b=100) if r["qtype"] == "t")
    assert rec["n"] == 4 and rec["unparsed"] == 1
    assert rec["accuracy"]["mean"] == 0.5  # 2 of 4, not 2 of 3


@pytest.mark.parametrize(
    ("reply", "verdict"), [("yes", True), ("No.", False), ("  YES, it is", True), ("unsure", None)]
)
def test_parse_verdict(reply, verdict):
    assert parse_verdict(reply) is verdict


def test_dense_and_hybrid_strategies(history):
    h = History(history)
    fake = FakeLLM()
    dense = make_dense(fake)
    assert dense(h, "beagle puppy Biscuit", NOW, "s").units[0][0].id == "s1:2"
    dense(h, "nurse", NOW, "s")  # same history: documents are not embedded again
    n_embed_calls = sum(c[0] == "embed" for c in fake.calls)
    hybrid = make_hybrid(fake)(h, "beagle puppy", NOW, "s")
    assert hybrid.units[0][0].id == "s1:2"
    assert {t.id for u in hybrid.units for t in u} <= {t.id for t in h.turns}
    assert n_embed_calls == 3  # one batch of documents, two queries


class _Stub(BaseHTTPRequestHandler):
    hits: ClassVar[list[str]] = []
    digests: ClassVar[dict[str, str]] = {}
    fail_next: ClassVar[list[int]] = []  # HTTP codes to answer the next requests with

    def _send(self, code: int, out: dict) -> None:
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        _Stub.hits.append(self.path)
        models = [{"name": n, "digest": d} for n, d in _Stub.digests.items()]
        self._send(200, {"models": models})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.hits.append(self.path)
        if _Stub.fail_next:
            self._send(_Stub.fail_next.pop(0), {"error": "busy"})
        elif self.path == "/api/generate":
            self._send(
                200, {"response": f"echo {body['prompt'][:5]} t={body['options']['temperature']}"}
            )
        elif self.path == "/api/embed":
            self._send(200, {"embeddings": [[float(len(t)), 1.0] for t in body["input"]]})
        else:
            self._send(404, {"error": "no such endpoint"})

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_url():
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _Stub.hits = []
    _Stub.digests = {"m:latest": "sha-m-1", "e:latest": "sha-e-1"}
    _Stub.fail_next = []
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def fast(url, cache=None):
    return Ollama(url, cache, timeout=5, backoff=0.0)


def test_ollama_client_caches_generations_and_embeddings(stub_url, tmp_path):
    client = fast(stub_url, DiskCache(tmp_path / "cache.sqlite"))
    assert client.generate("m", "hello world") == "echo hello t=0.0"
    assert client.generate("m", "hello world") == "echo hello t=0.0"
    assert client.generate("m", "hello world", {"temperature": 0.5}).endswith("t=0.5")
    assert _Stub.hits.count("/api/generate") == 2  # the repeat came from disk
    assert client.embed("e", ["ab", "abc"]) == [[2.0, 1.0], [3.0, 1.0]]
    assert client.embed("e", ["abc", "abcd"]) == [[3.0, 1.0], [4.0, 1.0]]
    assert _Stub.hits.count("/api/embed") == 2  # second call sent only "abcd"
    # a fresh client over the same cache directory resumes without calling out
    again = fast(stub_url, DiskCache(tmp_path / "cache.sqlite"))
    assert again.generate("m", "hello world") == "echo hello t=0.0"
    assert _Stub.hits.count("/api/generate") == 2


def test_a_re_pulled_model_misses_the_cache(stub_url, tmp_path):
    fast(stub_url, DiskCache(tmp_path / "c.sqlite")).generate("m", "hello world")
    _Stub.digests["m:latest"] = "sha-m-2"  # same tag, new weights
    fast(stub_url, DiskCache(tmp_path / "c.sqlite")).generate("m", "hello world")
    assert _Stub.hits.count("/api/generate") == 2


def test_server_errors_are_retried_client_errors_are_not(stub_url):
    _Stub.fail_next = [503, 500]
    assert fast(stub_url).generate("m", "hello world") == "echo hello t=0.0"
    assert _Stub.hits.count("/api/generate") == 3
    _Stub.fail_next = [400]
    with pytest.raises(OllamaError, match="HTTP 400"):
        fast(stub_url).generate("m", "again")
    _Stub.fail_next = [503] * 10
    with pytest.raises(OllamaError, match="HTTP 503"):
        Ollama(stub_url, retries=2, backoff=0.0).generate("m", "third")


def test_proxy_environment_is_ignored_for_local_ollama(stub_url, monkeypatch):
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")  # nothing listens there
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert fast(stub_url).generate("m", "hello world") == "echo hello t=0.0"


def test_ollama_errors_are_clear(stub_url):
    with pytest.raises(OllamaError, match="HTTP 404"):
        fast(stub_url)._post("/api/nothing", {})
    with pytest.raises(OllamaError, match="not pulled"):
        fast(stub_url).generate("missing-model", "x")
    with pytest.raises(OllamaError, match="cannot reach Ollama"):
        Ollama("http://127.0.0.1:9", timeout=2, retries=1, backoff=0.0).generate("m", "x")


def test_run_models_sh_matches_model_tags_exactly():
    import re
    import shutil
    import subprocess
    from pathlib import Path

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_models.sh").read_text()
    pulled = re.search(r"^pulled\(\) \{.*?^\}", script, re.S | re.M).group(0)
    tags = (
        '{"models":[{"name":"qwen2.5:14b-instruct-q8_0"},{"name":"nomic-embed-text:latest"},'
        '{"name":"qwen2.5-coder:14b"}]}'
    )
    checks = "; ".join(
        f'pulled "{m}" && echo "{m} yes" || echo "{m} no"'
        for m in ("qwen2.5:14b-instruct", "nomic-embed-text", "qwen2.5-coder:14b", "qwen2.5-coder")
    )
    out = subprocess.run(
        [bash, "-c", f"tags='{tags}'\n{pulled}\n{checks}"], capture_output=True, text=True
    ).stdout.split("\n")
    assert out[:4] == [
        "qwen2.5:14b-instruct no",  # only a longer tag with the same prefix is pulled
        "nomic-embed-text yes",  # :latest
        "qwen2.5-coder:14b yes",
        "qwen2.5-coder no",
    ]
