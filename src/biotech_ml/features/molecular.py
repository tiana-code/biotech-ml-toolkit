import logging

import numpy as np

from biotech_ml.exceptions import InputValidationError

logger = logging.getLogger(__name__)


_rdkit_cache: tuple | None = None


def _get_rdkit():
    global _rdkit_cache
    if _rdkit_cache is not None:
        return _rdkit_cache
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
        _rdkit_cache = (Chem, AllChem, Descriptors, MACCSkeys, rdMolDescriptors)
        return _rdkit_cache
    except ImportError:
        raise ImportError(
            "rdkit is required for molecular features. "
            "Install it with: pip install 'biotech-ml-toolkit[rdkit]'"
        )


def _validate_smiles(smiles: str):
    if not isinstance(smiles, str):
        raise InputValidationError(f"SMILES must be a string, got {type(smiles).__name__}")
    if not smiles.strip():
        logger.warning("Empty SMILES string")
        return None
    Chem, *_ = _get_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Invalid SMILES: %s", smiles)
    return mol


def smiles_to_morgan(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    Chem, AllChem, *_ = _get_rdkit()
    mol = _validate_smiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


def smiles_to_morgan_with_validity(smiles: str, radius: int = 2, n_bits: int = 2048) -> tuple[np.ndarray, bool]:
    """Returns (fingerprint_vector, is_valid). Zero vector if invalid."""
    Chem, AllChem, *_ = _get_rdkit()
    mol = _validate_smiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8), False
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr, True


def smiles_to_maccs(smiles: str) -> np.ndarray:
    Chem, _, _, MACCSkeys, _ = _get_rdkit()
    mol = _validate_smiles(smiles)
    if mol is None:
        return np.zeros(167, dtype=np.uint8)

    fingerprint = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(167, dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


def batch_smiles_to_fingerprints(smiles_list: list[str], fp_type: str = "morgan") -> np.ndarray:
    if not smiles_list:
        return np.empty((0, 0), dtype=np.uint8)

    if fp_type == "morgan":
        fingerprint_fn = smiles_to_morgan
        n_bits = 2048
    elif fp_type == "maccs":
        fingerprint_fn = smiles_to_maccs
        n_bits = 167
    else:
        raise ValueError(f"Unsupported fingerprint type: {fp_type}. Use 'morgan' or 'maccs'.")

    result = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for idx, smiles in enumerate(smiles_list):
        result[idx] = fingerprint_fn(smiles)
    return result


def smiles_to_descriptors(smiles: str) -> dict:
    _, _, Descriptors, _, rdMolDescriptors = _get_rdkit()
    mol = _validate_smiles(smiles)
    if mol is None:
        return {
            "molecular_weight": 0.0,
            "logp": 0.0,
            "num_h_donors": 0,
            "num_h_acceptors": 0,
            "num_rotatable_bonds": 0,
            "tpsa": 0.0,
        }

    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "num_h_donors": rdMolDescriptors.CalcNumHBD(mol),
        "num_h_acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "tpsa": Descriptors.TPSA(mol),
    }
