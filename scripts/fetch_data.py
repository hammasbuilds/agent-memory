"""Download the two benchmark files, resumably, in parallel byte-range chunks.

The network this was built on throttles Hugging Face to ~20 KB/s per connection,
so a single `curl` of the 277 MB LongMemEval_s file would take hours. Each file is
split into 5 MB ranges; every range is written to its own part file, so an
interrupted run resumes from the parts already on disk. The assembled file is
only accepted if its SHA-256 matches the one published by the host.

    uv run python scripts/fetch_data.py                 # locomo + longmemeval oracle + s
    uv run python scripts/fetch_data.py --only locomo oracle
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

# name -> (url, size in bytes, sha256). Sizes and hashes come from the hosts' own
# metadata (the Hugging Face tree API's LFS oid; the GitHub blob for LoCoMo).
FILES = {
    "locomo": (
        "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
        2_805_274,
        "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
        "locomo10.json",
    ),
    "oracle": (
        f"{HF}/longmemeval_oracle.json",
        15_388_478,
        "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c",
        "longmemeval_oracle.json",
    ),
    "s": (
        f"{HF}/longmemeval_s_cleaned.json",
        277_383_467,
        "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
        "longmemeval_s_cleaned.json",
    ),
}


def _fetch_range(url: str, start: int, end: int, part: Path, retries: int = 30) -> int:
    """Fetch bytes [start, end] into `part`, resuming a partial part file."""
    want = end - start + 1
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            return want
        if have > want:
            part.unlink()
            have = 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, part.open("ab") as fh:
                if resp.status != 206:
                    raise OSError(f"server ignored the Range header (HTTP {resp.status})")
                while block := resp.read(65536):
                    fh.write(block)
        except OSError as exc:
            print(f"  retry {attempt + 1} for {part.name}: {exc}", file=sys.stderr)
            time.sleep(min(2 * (attempt + 1), 20))
    raise RuntimeError(f"gave up on {part.name} after {retries} attempts")


def fetch(name: str, out_dir: Path, workers: int, chunk_mb: int = 5) -> Path:
    url, size, sha, filename = FILES[name]
    dest = out_dir / filename
    if dest.exists() and dest.stat().st_size == size and _sha256(dest) == sha:
        print(f"{filename}: already present and verified")
        return dest
    chunk = chunk_mb * 1024 * 1024
    parts_dir = out_dir / f".{filename}.{chunk_mb}mb.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_range, url, s, e, parts_dir / f"{i:05d}"): i
            for i, (s, e) in enumerate(ranges)
        }
        for done, fut in enumerate(as_completed(futures), 1):
            fut.result()
            mb = done * chunk / 1e6
            print(f"  {filename}: {done}/{len(ranges)} chunks ({mb / (time.time() - t0):.2f} MB/s)")
    with dest.open("wb") as out:
        for i in range(len(ranges)):
            out.write((parts_dir / f"{i:05d}").read_bytes())
    got = _sha256(dest)
    if got != sha:
        dest.unlink()
        raise RuntimeError(f"{filename}: sha256 {got} != expected {sha}; parts kept for retry")
    for p in parts_dir.iterdir():
        p.unlink()
    parts_dir.rmdir()
    print(f"{filename}: verified sha256 {sha[:12]}")
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=sorted(FILES), default=sorted(FILES))
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--workers", type=int, default=16, help="parallel range requests")
    ap.add_argument("--chunk-mb", type=int, default=5, help="size of each range request")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        fetch(name, args.out, args.workers, args.chunk_mb)


if __name__ == "__main__":
    main()
