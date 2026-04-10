import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFIndex:
    def __init__(self, max_features: int = 10000, ngram_range: tuple = (1, 2)):
        self._vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self._matrix: Any = None

    def fit(self, documents: list[str]) -> None:
        self._matrix = self._vectorizer.fit_transform(documents)

    def transform(self, text: str) -> np.ndarray:
        return self._vectorizer.transform([text])

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self._matrix is None:
            return []

        query_vec = self._vectorizer.transform([query])
        scores = (self._matrix @ query_vec.T).toarray().ravel()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"index": int(idx), "score": float(scores[idx])}
            for idx in top_indices
            if scores[idx] > 0
        ]

    def save(self, path: Path) -> None:
        joblib.dump({"vectorizer": self._vectorizer, "matrix": self._matrix}, path)

    def load(self, path: Path) -> None:
        data = joblib.load(path)
        self._vectorizer = data["vectorizer"]
        self._matrix = data["matrix"]


_TOKENIZE_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKENIZE_RE.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._documents: list[str] = []
        self._metadata: list[dict] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_term_freqs: list[Counter] = []
        self._doc_lengths: np.ndarray = np.array([])
        self._avg_dl: float = 0.0
        self._df: dict[str, int] = {}
        self._n_docs: int = 0

    @property
    def document_count(self) -> int:
        return self._n_docs

    def fit(self, documents: list[str], metadata: list[dict] | None = None) -> None:
        self._documents = documents
        self._metadata = metadata if metadata is not None else [{} for _ in documents]
        self._n_docs = len(documents)
        self._doc_tokens = [_tokenize(doc) for doc in documents]
        self._doc_lengths = np.array([len(tokens) for tokens in self._doc_tokens])
        self._avg_dl = float(self._doc_lengths.mean()) if self._n_docs > 0 else 0.0
        self._doc_term_freqs: list[Counter] = [Counter(tokens) for tokens in self._doc_tokens]

        doc_freqs: dict[str, int] = {}
        for tokens in self._doc_tokens:
            for term in set(tokens):
                doc_freqs[term] = doc_freqs.get(term, 0) + 1
        self._df = doc_freqs

    def _idf(self, term: str) -> float:
        doc_freq = self._df.get(term, 0)
        if doc_freq == 0:
            return 0.0
        return math.log((self._n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self._n_docs == 0:
            return []

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        scores = np.zeros(self._n_docs)

        for term in query_terms:
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for doc_idx, term_freqs in enumerate(self._doc_term_freqs):
                term_freq = term_freqs[term]
                if term_freq == 0:
                    continue
                doc_len = self._doc_lengths[doc_idx]
                numerator = term_freq * (self._k1 + 1)
                denominator = term_freq + self._k1 * (1 - self._b + self._b * doc_len / self._avg_dl)
                scores[i] += idf * numerator / denominator

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "text": self._documents[idx],
                "score": float(scores[idx]),
                "metadata": self._metadata[idx],
            }
            for idx in top_indices
            if scores[idx] > 0
        ]

    def save(self, path: Path) -> None:
        joblib.dump(
            {
                "documents": self._documents,
                "metadata": self._metadata,
                "doc_tokens": self._doc_tokens,
                "doc_term_freqs": self._doc_term_freqs,
                "doc_lengths": self._doc_lengths,
                "avg_dl": self._avg_dl,
                "df": self._df,
                "n_docs": self._n_docs,
                "k1": self._k1,
                "b": self._b,
            },
            path,
        )

    def load(self, path: Path) -> None:
        data = joblib.load(path)
        self._documents = data["documents"]
        self._metadata = data["metadata"]
        self._doc_tokens = data["doc_tokens"]
        self._doc_term_freqs = data.get(
            "doc_term_freqs",
            [Counter(tokens) for tokens in self._doc_tokens],
        )
        self._doc_lengths = data["doc_lengths"]
        self._avg_dl = data["avg_dl"]
        self._df = data["df"]
        self._n_docs = data["n_docs"]
        self._k1 = data["k1"]
        self._b = data["b"]
