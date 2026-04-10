import logging
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from biotech_ml.exceptions import InputValidationError

logger = logging.getLogger(__name__)

_VALID_BASES = re.compile(r"^[ACGTNacgtn]+$")


def _sanitize_sequence(sequence: str, strict: bool = False) -> str:
    cleaned = sequence.strip().upper()
    if not cleaned:
        return ""
    if not _VALID_BASES.match(cleaned):
        invalid_chars = set(cleaned) - set("ACGTN")
        if strict:
            raise InputValidationError(
                f"Invalid characters in sequence: {invalid_chars}"
            )
        logger.warning("Invalid characters in sequence: %s", invalid_chars)
        cleaned = re.sub(r"[^ACGTN]", "", cleaned)
    return cleaned


def extract_kmers(sequence: str, k: int = 6) -> dict[str, int]:
    if k < 1:
        raise ValueError("k must be >= 1")
    seq = _sanitize_sequence(sequence)
    if len(seq) < k:
        return {}
    return dict(Counter(seq[pos : pos + k] for pos in range(len(seq) - k + 1)))


def sequence_to_kmer_vector(sequence: str, k: int = 6, vocabulary: list[str] | None = None) -> np.ndarray:
    kmer_counts = extract_kmers(sequence, k)

    if vocabulary is None:
        import warnings
        warnings.warn(
            "No vocabulary provided - feature space is derived from this sequence only. "
            "Pass a fixed vocabulary for reproducible inference.",
            stacklevel=2,
        )
        vocabulary = sorted(kmer_counts.keys())

    if not vocabulary:
        return np.array([], dtype=np.float64)

    kmer_to_idx = {kmer: idx for idx, kmer in enumerate(vocabulary)}
    vector = np.zeros(len(vocabulary), dtype=np.float64)

    for kmer, count in kmer_counts.items():
        idx = kmer_to_idx.get(kmer)
        if idx is not None:
            vector[idx] = count

    return vector


def batch_sequences_to_features(sequences: list[str], k: int = 6) -> tuple[np.ndarray, DictVectorizer]:
    if not sequences:
        vectorizer = DictVectorizer(sparse=False, dtype=np.float64)
        return np.empty((0, 0), dtype=np.float64), vectorizer
    kmer_dicts = [extract_kmers(seq, k) for seq in sequences]
    vectorizer = DictVectorizer(sparse=False, dtype=np.float64)
    return vectorizer.fit_transform(kmer_dicts), vectorizer


def gc_content(sequence: str) -> float:
    seq = _sanitize_sequence(sequence)
    if not seq:
        return 0.0
    effective_length = len(seq) - seq.count("N")
    if effective_length == 0:
        return 0.0
    gc_count = seq.count("G") + seq.count("C")
    return gc_count / effective_length
