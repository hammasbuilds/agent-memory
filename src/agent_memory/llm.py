"""A minimal Ollama client with a disk cache, and a deterministic fake for tests.

Every generation and embedding is cached on disk under a key of (model, prompt hash,
options), so an interrupted model run resumes where it stopped and a re-run of the
report costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
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
    """JSON values in files named by their key, sharded by the key's first two chars."""

    def __init__(self, root: Path):
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> object | None:
        p = self._path(key)
        return json.loads(p.read_text("utf-8")) if p.exists() else None

    def put(self, key: str, value: object) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value), "utf-8")
        os.replace(tmp, p)  # atomic, so a killed run never leaves half a file


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
        if self.cache and (hit := self.cache.get(key)) is not None:
            return str(hit)
        body = {"model": model, "prompt": prompt, "options": opts, "stream": False}
        out = self._post("/api/generate", body)["response"]
        if self.cache:
            self.cache.put(key, out)
        return out

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Embeddings in input order; cached per text, only misses are sent."""
        keys = [_key("embed", model, t) for t in texts]
        out: list[list[float] | None] = [
            self.cache.get(k) if self.cache else None  # type: ignore[misc]
            for k in keys
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
                    self.cache.put(keys[i], v)
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
