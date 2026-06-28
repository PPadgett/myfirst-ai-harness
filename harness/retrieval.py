from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing import Any


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    metadata: dict[str, Any]
    score: float = 0.0


class DirectoryCorpusRetriever:
    def __init__(self, corpus_dir: Path) -> None:
        self.corpus_dir = corpus_dir
        self._docs = self._load_docs()

    def _load_docs(self) -> list[RetrievedDoc]:
        docs: list[RetrievedDoc] = []
        if not self.corpus_dir.exists():
            return docs
        for path in sorted(self.corpus_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".txt", ".md", ".mdx", ".csv", ".jsonl", ".json"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            docs.append(
                RetrievedDoc(
                    doc_id=str(path),
                    text=text,
                    metadata={"source": str(path), "suffix": path.suffix.lower()},
                )
            )
        return docs

    def search(self, query: str, k: int = 20) -> list[RetrievedDoc]:
        if not self._docs:
            return []
        q = query.lower().split()
        qset = {w for w in q if len(w) > 2}
        scored: list[RetrievedDoc] = []
        for doc in self._docs:
            dtext = doc.text.lower()
            hits = sum(1 for w in qset if w in dtext)
            # simple term-overlap scorer
            score = float(hits)
            if score > 0:
                scored.append(RetrievedDoc(doc.doc_id, doc.text, doc.metadata, score))
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[: max(0, k)]


class SimpleReranker:
    @staticmethod
    def rank(query: str, docs: list[RetrievedDoc], k: int = 8) -> list[RetrievedDoc]:
        if not docs:
            return []
        qset = {w for w in query.lower().split() if len(w) > 2}
        for doc in docs:
            dtext = doc.text.lower()
            bonus = sum(1 for w in qset if w in dtext)
            doc.score += 0.25 * bonus
        return sorted(docs, key=lambda d: d.score, reverse=True)[:k]


def pack_context(docs: list[RetrievedDoc], max_tokens: int) -> list[RetrievedDoc]:
    out: list[RetrievedDoc] = []
    used = 0
    for doc in docs:
        doc_tokens = max(1, len(doc.text.split()))
        if used + doc_tokens > max_tokens:
            if not out:
                out.append(doc)
            break
        out.append(doc)
        used += doc_tokens
    return out

