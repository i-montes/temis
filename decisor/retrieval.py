"""Evidence retrieval for the decisor (RAG). Pluggable:

1. The caller brings its own RAG: it passes `context` (a string or a list of passages) in the
   request, and nothing here runs.
2. Otherwise, Clio Memory: if `clio-memory` is on the PATH it is used; if not, the release binary is
   downloaded into ./bin (sha256-checked against the release's SHA256SUMS). It never runs `setup`,
   so no agent configuration is touched.

The memory file is the one Clio shares (~/.local/share/clio/memory.db) when it exists, else the
decisor's own (data/clio-memory/memory.db). Reads go through `clio-memory --json recall`, which opens
the file read-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = "i-montes/clio-memory"
VERSION = os.environ.get("DECISOR_CLIO_MEMORY_VERSION", "v0.7.1")
BIN_DIR = Path("bin")
SHARED_DB = Path(os.environ.get("CLIO_MEMORY_DATA", Path.home() / ".local/share/clio")) / "memory.db"
OWN_DB = Path("data/clio-memory/memory.db")


@dataclass
class Passage:
    id: str
    text: str
    score: float
    source: str


def _asset() -> str:
    sysname, arch = platform.system(), platform.machine().lower()
    arch = "aarch64" if arch in ("arm64", "aarch64") else "x86_64"
    if sysname == "Darwin":
        return f"clio-memory-{arch}-apple-darwin"
    if sysname == "Linux":
        return f"clio-memory-{arch}-unknown-linux-gnu"
    raise RuntimeError(f"clio-memory has no release binary for {sysname}")


def find_or_download_binary() -> Path:
    found = shutil.which("clio-memory")
    if found:
        return Path(found)
    local = BIN_DIR / "clio-memory"
    if local.exists():
        return local
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    asset = _asset()
    base = f"https://github.com/{REPO}/releases/download/{VERSION}"
    sums = urllib.request.urlopen(f"{base}/SHA256SUMS", timeout=60).read().decode()
    want = next((ln.split()[0] for ln in sums.splitlines() if ln.strip().endswith(asset)), None)
    if not want:
        raise RuntimeError(f"{asset} not listed in SHA256SUMS")
    data = urllib.request.urlopen(f"{base}/{asset}", timeout=300).read()
    if hashlib.sha256(data).hexdigest() != want:
        raise RuntimeError("clio-memory download failed its sha256 check")
    local.write_bytes(data)
    local.chmod(0o755)
    return local


class ClioMemoryRetriever:
    def __init__(self, db: Path | None = None, scope: str = "clio") -> None:
        self.bin = find_or_download_binary()
        self.db = db or (SHARED_DB if SHARED_DB.exists() else OWN_DB)
        self.scope = scope

    @property
    def available(self) -> bool:
        return self.db.exists()

    def search(self, query: str, k: int = 5) -> list[Passage]:
        if not self.available or not query.strip():
            return []
        try:
            out = subprocess.run([str(self.bin), "--db", str(self.db), "--json", "recall", self.scope, query[:2000]],
                                 capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            return []
        if out.returncode != 0:
            return []
        try:
            hits = json.loads(out.stdout or "[]")
        except json.JSONDecodeError:
            return []
        res = []
        for h in hits[:k]:
            m = h.get("memory") or {}
            res.append(Passage(id=f"m{m.get('id')}", text=str(m.get("content", ""))[:1200],
                               score=float(h.get("score", 0.0)), source=f"clio-memory:{m.get('source') or self.scope}"))
        return res


def format_context(passages: list[Passage] | list[str] | str | None) -> str | None:
    """Passages -> the context text fed to the model (marked as data, numbered)."""
    if not passages:
        return None
    if isinstance(passages, str):
        return passages
    lines = []
    for i, p in enumerate(passages, 1):
        text = p.text if isinstance(p, Passage) else str(p)
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)
