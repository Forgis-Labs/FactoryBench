"""Anti-aliased downsampling for DataFrame-based robot time-series data.

Uses ``scipy.signal.decimate`` (IIR Chebyshev Type I low-pass filter followed
by subsampling) on continuous float columns and plain every-*q*-th-row
subsampling on categorical / integer / string columns.

Shared by all normalizer scripts in this package.
"""

import numpy as np
import pandas as pd
from typing import Optional, Set

try:
    from scipy.signal import decimate as _sp_decimate
except ImportError:
    _sp_decimate = None


def decimate_dataframe(
    df: pd.DataFrame,
    q: int,
    continuous_cols: Optional[Set[str]] = None) -> pd.DataFrame:
    """Downsample *df* by factor *q* with anti-alias filtering.

    Parameters
    ----------
    df : pd.DataFrame
        Input data at the original sampling rate.  Must be uniformly sampled.
    q : int
        Decimation factor (keep 1 out of every *q* samples).
    continuous_cols : set of str, optional
        Column names that represent continuous (float) signals and should be
        low-pass filtered before subsampling.  If *None*, every column whose
        dtype is a floating-point type is treated as continuous; everything
        else (integers, strings, booleans) is subsampled directly.

    Returns
    -------
    pd.DataFrame
        Decimated DataFrame with ``len(df) // q`` rows (index reset to 0-based).
    """
    if q <= 1:
        return df.reset_index(drop=True)

    if _sp_decimate is None:
        raise ImportError(
            "scipy is required for anti-aliased decimation.  "
            "Install with:  pip install scipy"
        )

    if continuous_cols is None:
        continuous_cols = set(df.select_dtypes(include=[np.floating]).columns)

    n_out = len(df) // q
    if n_out == 0:
        # Signal shorter than one decimation period, return the first row.
        return df.iloc[:1].reset_index(drop=True)

    out: dict = {}

    for col in df.columns:
        values = df[col].values

        if col in continuous_cols:
            numeric = pd.to_numeric(pd.Series(values), errors="coerce")
            valid_count = int(numeric.notna().sum())

            if valid_count < 2 * q:
                # Too few valid samples for meaningful filtering, subsample.
                out[col] = values[::q][:n_out]
            else:
                filled = (
                    numeric
                    .interpolate(limit_direction="both")
                    .bfill()
                    .ffill()
                    .fillna(0.0)
                    .values
                    .astype(np.float64)
                )
                try:
                    decimated = _sp_decimate(filled, q)
                    out[col] = decimated[:n_out]
                except ValueError:
                    # Signal too short for the anti-alias filter, fall back to subsample
                    out[col] = filled[::q][:n_out]
        else:
            # Categorical / integer / string, just pick every q-th value.
            out[col] = values[::q][:n_out]

    return pd.DataFrame(out)
