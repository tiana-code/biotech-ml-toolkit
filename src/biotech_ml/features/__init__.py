from biotech_ml.features.genomic import (
    batch_sequences_to_features,
    extract_kmers,
    gc_content,
    sequence_to_kmer_vector,
)
from biotech_ml.features.tabular import (
    compute_z_scores,
    encode_categorical,
    fill_missing,
    normalize_features,
)
from biotech_ml.features.text import BM25Index, TFIDFIndex


def __getattr__(name: str):
    """Lazy import for molecular features (requires rdkit)."""
    _molecular_names = {
        "batch_smiles_to_fingerprints",
        "smiles_to_descriptors",
        "smiles_to_maccs",
        "smiles_to_morgan",
        "smiles_to_morgan_with_validity",
    }
    if name in _molecular_names:
        from biotech_ml.features import molecular
        return getattr(molecular, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "batch_sequences_to_features",
    "batch_smiles_to_fingerprints",
    "BM25Index",
    "compute_z_scores",
    "encode_categorical",
    "extract_kmers",
    "fill_missing",
    "gc_content",
    "normalize_features",
    "sequence_to_kmer_vector",
    "smiles_to_descriptors",
    "smiles_to_maccs",
    "smiles_to_morgan",
    "smiles_to_morgan_with_validity",
    "TFIDFIndex",
]
