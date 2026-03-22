"""
Step 3: Generate monthly budget vectors per household.

Draws from a multivariate normal distribution parameterized by:
- Mean vector: per quintile × division (from epf_quintile_params.json)
- Covariance matrix: correlation structure (from epf_correlation.json) ×
  per-quintile standard deviations

Each household × month gets a 13-dimensional budget vector (one entry per
CCIF division) in CLP. Negative draws are clipped to zero.

Output: budgets DataFrame with columns
    household_id, month, division, budget_clp
in long format (one row per household × month × division).
"""
import json
import logging

import numpy as np
import pandas as pd

from config import (
    CCIF_DIVISIONS,
    DATA_DIR,
    LOOKUPS_DIR,
    N_MONTHS,
    SEED,
)

logger = logging.getLogger(__name__)


def load_quintile_params() -> dict[int, dict[str, dict[str, float]]]:
    """
    Load per-quintile, per-division mean and std from JSON.

    Returns { quintile(int): { division(str): { "mean": float, "std": float } } }
    """
    path = LOOKUPS_DIR / "epf_quintile_params.json"
    with open(path) as f:
        raw = json.load(f)
    # Convert string quintile keys to int
    return {int(q): divs for q, divs in raw.items()}


def load_correlation_matrix() -> tuple[list[str], np.ndarray]:
    """
    Load cross-division correlation matrix from JSON.

    Returns (division_order, correlation_matrix) where correlation_matrix
    is a 13×13 numpy array.
    """
    path = LOOKUPS_DIR / "epf_correlation.json"
    with open(path) as f:
        raw = json.load(f)
    div_order = raw["division_order"]
    corr = np.array(raw["correlation_matrix"], dtype=np.float64)

    # Validate symmetry and diagonal
    assert corr.shape == (len(div_order), len(div_order)), \
        f"Correlation matrix shape {corr.shape} does not match {len(div_order)} divisions"
    assert np.allclose(corr, corr.T, atol=1e-10), \
        "Correlation matrix is not symmetric"
    assert np.allclose(np.diag(corr), 1.0), \
        "Correlation matrix diagonal is not all ones"

    return div_order, corr


def build_covariance_matrix(
    corr: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """
    Construct covariance matrix from correlation matrix and standard deviations.

    cov = diag(std) @ corr @ diag(std)

    Parameters
    ----------
    corr : np.ndarray, shape (D, D)
        Correlation matrix.
    stds : np.ndarray, shape (D,)
        Standard deviations per dimension.

    Returns
    -------
    np.ndarray, shape (D, D)
        Positive semi-definite covariance matrix.
    """
    D = np.diag(stds)
    cov = D @ corr @ D

    # Force PSD: eigenvalue clipping
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Re-symmetrize (numerical hygiene)
    cov = (cov + cov.T) / 2.0

    return cov


def draw_budget_vectors(
    households: pd.DataFrame,
    n_months: int,
    quintile_params: dict[int, dict[str, dict[str, float]]],
    div_order: list[str],
    corr: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Draw monthly budget vectors for all households.

    For each quintile, constructs the covariance matrix once, then draws
    all (household × month) vectors in a single batch.

    Parameters
    ----------
    households : pd.DataFrame
        Must have columns: household_id, income_quintile.
    n_months : int
        Number of months to generate (e.g. 6).
    quintile_params : dict
        From load_quintile_params().
    div_order : list[str]
        Division ordering matching the correlation matrix.
    corr : np.ndarray
        Correlation matrix.
    rng : np.random.Generator

    Returns
    -------
    pd.DataFrame
        Long-format: household_id, month (1-indexed), division, budget_clp.
    """
    all_rows = []

    for quintile in sorted(quintile_params.keys()):
        hh_in_q = households[households["income_quintile"] == quintile]
        n_hh = len(hh_in_q)
        if n_hh == 0:
            continue

        # Build mean and std vectors in division_order
        params = quintile_params[quintile]
        means = np.array([params[d]["mean"] for d in div_order], dtype=np.float64)
        stds = np.array([params[d]["std"] for d in div_order], dtype=np.float64)

        # Build covariance
        cov = build_covariance_matrix(corr, stds)

        # Draw all vectors at once: shape (n_hh * n_months, n_divisions)
        n_draws = n_hh * n_months
        draws = rng.multivariate_normal(means, cov, size=n_draws)

        # Clip negatives to zero
        draws = np.maximum(draws, 0.0)

        # Round to integer CLP
        draws = np.round(draws).astype(np.int64)

        # Build long-format rows
        hh_ids = hh_in_q["household_id"].values
        draw_idx = 0
        for hh_id in hh_ids:
            for month in range(1, n_months + 1):
                budget_vec = draws[draw_idx]
                for div_idx, div_code in enumerate(div_order):
                    all_rows.append({
                        "household_id": hh_id,
                        "month": month,
                        "division": div_code,
                        "budget_clp": int(budget_vec[div_idx]),
                    })
                draw_idx += 1

    df = pd.DataFrame(all_rows)
    return df


def generate_budgets(
    households: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Main entry point: generate monthly budget vectors for all households.

    Pipeline:
    1. Load quintile parameters (mean, std per division)
    2. Load correlation matrix
    3. For each quintile batch, construct covariance and draw MVN vectors
    4. Clip to non-negative, round to integer CLP
    5. Return long-format table

    Parameters
    ----------
    households : pd.DataFrame
        Output from census.generate_households().

    Returns
    -------
    pd.DataFrame
        Columns: household_id, month, division, budget_clp.
        Shape: N_HOUSEHOLDS × N_MONTHS × 13 divisions.
    """
    rng = np.random.default_rng(seed)

    # Step 1
    quintile_params = load_quintile_params()
    logger.info("Loaded quintile params for %d quintiles", len(quintile_params))

    # Step 2
    div_order, corr = load_correlation_matrix()
    logger.info("Loaded %dx%d correlation matrix", corr.shape[0], corr.shape[1])

    # Step 3-4
    budgets = draw_budget_vectors(
        households, N_MONTHS, quintile_params, div_order, corr, rng
    )

    expected_rows = len(households) * N_MONTHS * len(div_order)
    logger.info(
        "Generated %d budget rows (expected %d)", len(budgets), expected_rows
    )
    if len(budgets) != expected_rows:
        logger.warning("Row count mismatch — check for missing quintiles in params")

    # Summary stats
    totals = budgets.groupby("household_id")["budget_clp"].sum()
    per_month = totals / N_MONTHS
    logger.info(
        "Monthly total per household — mean: %,.0f CLP, median: %,.0f CLP, "
        "Q1: %,.0f CLP, Q5 proxy (p90): %,.0f CLP",
        per_month.mean(),
        per_month.median(),
        per_month.quantile(0.2),
        per_month.quantile(0.9),
    )

    return budgets
