import logging

import numpy as np

from biotech_ml.exceptions import InputValidationError

logger = logging.getLogger(__name__)


def normalize_features(features: np.ndarray, method: str = "standard") -> tuple[np.ndarray, dict]:
    if features.size == 0:
        return features.copy(), {}

    if method == "standard":
        means = np.nanmean(features, axis=0)
        stds = np.nanstd(features, axis=0)
        nan_cols = np.isnan(means)
        if nan_cols.any():
            logger.warning("All-NaN columns detected at indices %s, filling with 0", np.where(nan_cols)[0].tolist())
            means[nan_cols] = 0.0
            stds[nan_cols] = 1.0
        stds[stds == 0] = 1.0
        result = (features - means) / stds
        return result, {"method": method, "means": means, "stds": stds}

    if method == "minmax":
        mins = np.nanmin(features, axis=0)
        maxs = np.nanmax(features, axis=0)
        nan_cols = np.isnan(mins)
        if nan_cols.any():
            logger.warning("All-NaN columns detected at indices %s, filling with 0", np.where(nan_cols)[0].tolist())
            mins[nan_cols] = 0.0
            maxs[nan_cols] = 1.0
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        result = (features - mins) / ranges
        return result, {"method": method, "mins": mins, "ranges": ranges}

    if method == "robust":
        medians = np.nanmedian(features, axis=0)
        percentile_25 = np.nanpercentile(features, 25, axis=0)
        percentile_75 = np.nanpercentile(features, 75, axis=0)
        nan_cols = np.isnan(medians)
        if nan_cols.any():
            logger.warning("All-NaN columns detected at indices %s, filling with 0", np.where(nan_cols)[0].tolist())
            medians[nan_cols] = 0.0
            percentile_25[nan_cols] = 0.0
            percentile_75[nan_cols] = 1.0
        iqr = percentile_75 - percentile_25
        iqr[iqr == 0] = 1.0
        result = (features - medians) / iqr
        return result, {"method": method, "medians": medians, "iqr": iqr}

    raise ValueError(f"Unsupported method: {method}. Use 'standard', 'minmax', or 'robust'.")


def fill_missing(features: np.ndarray, strategy: str = "median") -> np.ndarray:
    result = features.copy()

    if strategy == "median":
        fill_values = np.nanmedian(features, axis=0)
    elif strategy == "mean":
        fill_values = np.nanmean(features, axis=0)
    elif strategy == "zero":
        fill_values = np.zeros(features.shape[1] if features.ndim > 1 else 1)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}. Use 'median', 'mean', or 'zero'.")

    if result.ndim == 1:
        mask = np.isnan(result)
        result[mask] = fill_values[0] if isinstance(fill_values, np.ndarray) else fill_values
    else:
        for col_idx in range(result.shape[1]):
            mask = np.isnan(result[:, col_idx])
            result[mask, col_idx] = fill_values[col_idx]

    return result


def encode_categorical(
    values: list[str],
    categories: list[str] | None = None,
    on_unknown: str = "ignore",
) -> np.ndarray:
    if not values:
        return np.empty((0, 0), dtype=np.float64)

    if categories is None:
        categories = sorted(set(values))

    category_to_index = {category: idx for idx, category in enumerate(categories)}

    if on_unknown == "error":
        unknown = [value for value in values if value not in category_to_index]
        if unknown:
            raise InputValidationError(
                f"Unknown categories found: {sorted(set(unknown))}. "
                f"Expected one of: {categories}"
            )

    result = np.zeros((len(values), len(categories)), dtype=np.float64)

    for row_idx, value in enumerate(values):
        col_idx = category_to_index.get(value)
        if col_idx is not None:
            result[row_idx, col_idx] = 1.0

    return result


def compute_z_scores(values: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    if values.shape != means.shape or values.shape != stds.shape:
        raise ValueError(
            f"Shape mismatch: values={values.shape}, means={means.shape}, stds={stds.shape}"
        )
    safe_stds = stds.copy()
    safe_stds[safe_stds == 0] = 1.0
    return (values - means) / safe_stds
