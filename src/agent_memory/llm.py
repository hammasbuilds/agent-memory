"""A minimal Ollama client with a disk cache, and a deterministic fake for tests.

Every generation and embedding is cached on disk under a key of (model, prompt hash,
options), so an interrupted model run resumes where it stopped and a re-run of the
report costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import urllib.error
import urllib.request
from array import array
from pathlib import Path
from typing import Protocol

DEFAULT_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen2.5:14b-instruct"
EMBED_MODEL = "nomic-embed-text"
GREEDY = {"temperature": 0.0, "seed": 0, "num_ctx": 8192}


class LLM(Protocol):
    def generate(self, model: str, prompt: str, options: dict | None = None) -> str: ...

    def embed(self, model: str, texts: list[str]) -> list[list[float]]: ...


class OllamaError(RuntimeError):
    pass


def _key(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


class DiskCache:
    """One SQLite file of cached results. Generations are stored as text, embeddings
    as packed float32 (LongMemEval_S alone is ~195k turn embeddings; as JSON files
    they would take ~3 GB). Every put commits, so a killed run loses at most one call.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, text TEXT, vector BLOB)"
        )

    def get_text(self, key: str) -> str | None:
        row = self.db.execute("SELECT text FROM cache WHERE key=?", (key,)).fetchone()
        return None if row is None else row[0]

    def get_vector(self, key: str) -> list[float] | None:
        row = self.db.execute("SELECT vector FROM cache WHERE key=?", (key,)).fetchone()
        return None if row is None else array("f", row[0]).tolist()

    def put_text(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO cache(key, text) VALUES (?, ?)", (key, value))

    def put_vector(self, key: str, value: list[float]) -> None:
        blob = array("f", value).tobytes()
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO cache(key, vector) VALUES (?, ?)", (key, blob))


class Ollama:
    """Talks to Ollama's HTTP API. Nothing is sent until `generate`/`embed` is called."""

    def __init__(self, url: str = DEFAULT_URL, cache: DiskCache | None = None, timeout: int = 600):
        self.url = url.rstrip("/")
        self.cache = cache
        self.timeout = timeout

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise OllamaError(f"{path} -> HTTP {e.code}: {e.read()[:300]!r}") from e
        except urllib.error.URLError as e:
            raise OllamaError(f"cannot reach Ollama at {self.url}: {e.reason}") from e

    def generate(self, model: str, prompt: str, options: dict | None = None) -> str:
        opts = {**GREEDY, **(options or {})}
        key = _key("generate", model, prompt, opts)
        if self.cache and (hit := self.cache.get_text(key)) is not None:
            return hit
        body = {"model": model, "prompt": prompt, "options": opts, "stream": False}
        out = self._post("/api/generate", body)["response"]
        if self.cache:
            self.cache.put_text(key, out)
        return out

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Embeddings in input order; cached per text, only misses are sent."""
        keys = [_key("embed", model, t) for t in texts]
        out: list[list[float] | None] = [
            self.cache.get_vector(k) if self.cache else None for k in keys
        ]
        missing = [i for i, v in enumerate(out) if v is None]
        for start in range(0, len(missing), 64):
            batch = missing[start : start + 64]
            vecs = self._post("/api/embed", {"model": model, "input": [texts[i] for i in batch]})[
                "embeddings"
            ]
            for i, v in zip(batch, vecs, strict=True):
                out[i] = v
                if self.cache:
                    self.cache.put_vector(keys[i], v)
        return [v for v in out if v is not None]


class FakeLLM:
    """Deterministic stand-in: answers from a script, embeds by hashed bag of words."""

    def __init__(self, replies: dict[str, str] | None = None, default: str = "", dim: int = 64):
        self.replies = replies or {}
        self.default = default
        self.dim = dim
        self.calls: list[tuple[str, str]] = []

    def generate(self, model: str, prompt: str, options: dict | None = None) -> str:
        self.calls.append(("generate", prompt))
        for needle, reply in self.replies.items():
            if needle in prompt:
                return reply
        return self.default

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        self.calls.append(("embed", f"{len(texts)} texts"))
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            out.append(v)
        return out
