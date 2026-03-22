"""
Step 1: Sample synthetic households from Censo 2024 RM joint distribution.

Reads hogares + viviendas parquets from data/, extracts the empirical joint
distribution of (household_size, tenure_type, dwelling_type) for Region
Metropolitana, and samples N_HOUSEHOLDS from it. Assigns quintiles,
temporal fingerprints, billing DOMs, bank, and transport profiles.

If Censo parquets are not found in data/, falls back to hardcoded marginals.

Output: households DataFrame matching HOUSEHOLD_SCHEMA.
"""
import logging

import numpy as np
import pandas as pd

from config import (
    BANKS,
    DATA_DIR,
    HOUSEHOLD_SCHEMA,
    N_HOUSEHOLDS,
    SEED,
)
from constants import (
    CASH_BASE_PROB,
    CENSO_HOGARES_FILE,
    CENSO_JOIN_KEYS,
    CENSO_VIVIENDAS_FILE,
    COL_CANT_PER,
    COL_DWELLING_TYPE,
    COL_REGION_HOG,
    COL_REGION_VIV,
    COL_TENURE,
    DOM_RANGE,
    DWELLING_DEFAULT,
    DWELLING_MAP,
    FALLBACK_MARGINALS,
    FINGERPRINT_RANGES,
    LARGE_HH_BLEND_WEIGHT,
    LARGE_HH_QUINTILE_SHIFT,
    LARGE_HH_THRESHOLD,
    LATE_NIGHT_PROB,
    QUINTILE_PRIOR,
    QUINTILES,
    REGION_RM,
    TENURE_MAP,
    TRANSPORT_PROFILE_PROB,
)

logger = logging.getLogger(__name__)

# Cap household size at 8 for the distribution (7+ grouped)
MAX_HH_SIZE = 8


def load_censo_rm() -> pd.DataFrame | None:
    """
    Load Censo 2024 hogares + viviendas for RM, return joined DataFrame.

    Returns None if parquet files are not found in DATA_DIR.
    Returned DataFrame has columns: cant_per, tenure, dwelling.
    """
    hogares_path = DATA_DIR / CENSO_HOGARES_FILE
    viviendas_path = DATA_DIR / CENSO_VIVIENDAS_FILE

    if not hogares_path.exists() or not viviendas_path.exists():
        logger.warning(
            "Censo parquets not found in %s. Using fallback marginals.", DATA_DIR
        )
        return None

    logger.info("Loading Censo 2024 hogares from %s", hogares_path)
    hogares = pd.read_parquet(hogares_path)
    hogares = hogares[hogares[COL_REGION_HOG] == REGION_RM].copy()

    logger.info("Loading Censo 2024 viviendas from %s", viviendas_path)
    viviendas = pd.read_parquet(viviendas_path)
    viviendas = viviendas[viviendas[COL_REGION_VIV] == REGION_RM].copy()

    # Join hogares to viviendas on geographic hierarchy
    # Keep only the columns we need from viviendas to avoid collision
    viviendas_cols = CENSO_JOIN_KEYS + [COL_DWELLING_TYPE]
    viviendas_slim = viviendas[viviendas_cols].drop_duplicates(subset=CENSO_JOIN_KEYS)

    df = hogares.merge(viviendas_slim, on=CENSO_JOIN_KEYS, how="left")

    # Map coded values to internal labels
    df["tenure"] = df[COL_TENURE].map(TENURE_MAP).fillna("otro")
    df["dwelling"] = df[COL_DWELLING_TYPE].map(DWELLING_MAP).fillna(DWELLING_DEFAULT)
    df["cant_per"] = df[COL_CANT_PER].clip(upper=MAX_HH_SIZE).astype(int)

    return df[["cant_per", "tenure", "dwelling"]]


def extract_joint_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute empirical joint distribution from Censo data.

    Returns DataFrame with columns: cant_per, tenure, dwelling, weight.
    Weights are normalized to sum to 1.
    """
    counts = df.groupby(["cant_per", "tenure", "dwelling"]).size().reset_index(name="count")
    counts["weight"] = counts["count"] / counts["count"].sum()
    return counts.drop(columns=["count"])


def build_fallback_distribution() -> pd.DataFrame:
    """
    Build joint distribution from hardcoded fallback marginals.
    """
    rows = [
        {"cant_per": cp, "tenure": t, "dwelling": d, "weight": w}
        for cp, t, d, w in FALLBACK_MARGINALS
    ]
    df = pd.DataFrame(rows)
    df["weight"] = df["weight"] / df["weight"].sum()  # renormalize
    return df


def sample_household_demographics(
    joint_dist: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Sample n households from the joint distribution.

    Returns DataFrame with columns: household_id, n_personas, tenure_type, dwelling_type.
    """
    indices = rng.choice(
        len(joint_dist),
        size=n,
        replace=True,
        p=joint_dist["weight"].values,
    )
    sampled = joint_dist.iloc[indices].reset_index(drop=True)

    households = pd.DataFrame({
        "household_id": [f"HH-{i:04d}" for i in range(n)],
        "n_personas": sampled["cant_per"].values,
        "tenure_type": sampled["tenure"].values,
        "dwelling_type": sampled["dwelling"].values,
    })
    return households


def assign_quintiles(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Assign income quintile to each household based on (dwelling, tenure, hh_size).

    Uses QUINTILE_PRIOR conditional table with large-household adjustment.
    """
    quintiles = np.empty(len(df), dtype=int)

    for i, row in df.iterrows():
        key = (row["dwelling_type"], row["tenure_type"])
        base_probs = np.array(QUINTILE_PRIOR.get(key, [0.2, 0.2, 0.2, 0.2, 0.2]))

        # Large household adjustment
        if row["n_personas"] >= LARGE_HH_THRESHOLD:
            shift = np.array(LARGE_HH_QUINTILE_SHIFT)
            base_probs = (
                (1 - LARGE_HH_BLEND_WEIGHT) * base_probs
                + LARGE_HH_BLEND_WEIGHT * shift
            )
            base_probs = base_probs / base_probs.sum()

        quintiles[i] = rng.choice(QUINTILES, p=base_probs)

    df = df.copy()
    df["income_quintile"] = quintiles
    return df


def assign_bank(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Assign a primary bank to each household.

    Uniform draw from BANKS list. Could be refined with quintile weighting
    later (e.g., BANCO ESTADO heavier for Q1-Q2).
    """
    df = df.copy()
    df["bank"] = rng.choice(BANKS, size=len(df))
    return df


def assign_fingerprints(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Draw temporal fingerprints for each household.

    These are drawn once and remain fixed for all 6 months of generation.
    Includes: preferred meal/commute hours, shopping day preference,
    late-night flag, billing DOMs, and transport profile.
    """
    n = len(df)
    df = df.copy()

    # --- Temporal fingerprint ---
    for field, (lo, hi) in FINGERPRINT_RANGES.items():
        df[field] = rng.uniform(lo, hi, size=n)

    df["preferred_shop_day"] = rng.integers(0, 7, size=n)  # Mon=0 ... Sun=6
    df["late_night_shopper"] = rng.random(size=n) < LATE_NIGHT_PROB

    # --- Billing DOMs ---
    dom_lo, dom_hi = DOM_RANGE
    dom_fields = [
        "dom_enel", "dom_agua", "dom_gas", "dom_telefono",
        "dom_arriendo", "dom_gastos_comunes", "dom_isapre",
    ]
    for field in dom_fields:
        df[field] = rng.integers(dom_lo, dom_hi + 1, size=n)

    # --- Transport profile (conditioned on quintile) ---
    for transport_field, quintile_probs in TRANSPORT_PROFILE_PROB.items():
        probs = df["income_quintile"].map(quintile_probs).values
        df[transport_field] = rng.random(size=n) < probs

    return df


def generate_households(seed: int = SEED) -> pd.DataFrame:
    """
    Main entry point: generate the full households table.

    Pipeline:
    1. Load Censo RM joint distribution (or fallback)
    2. Sample N_HOUSEHOLDS demographic profiles
    3. Assign income quintiles
    4. Assign bank
    5. Draw temporal fingerprints, DOMs, transport profiles
    6. Validate schema and return

    Returns DataFrame with columns matching HOUSEHOLD_SCHEMA.
    """
    rng = np.random.default_rng(seed)

    # Step 1: joint distribution
    censo_df = load_censo_rm()
    if censo_df is not None:
        joint_dist = extract_joint_distribution(censo_df)
        logger.info(
            "Censo RM: %d unique (size, tenure, dwelling) combinations",
            len(joint_dist),
        )
    else:
        joint_dist = build_fallback_distribution()
        logger.info("Using fallback marginals: %d combinations", len(joint_dist))

    # Step 2: sample demographics
    households = sample_household_demographics(joint_dist, N_HOUSEHOLDS, rng)
    logger.info("Sampled %d households", len(households))

    # Step 3: assign quintiles
    households = assign_quintiles(households, rng)
    logger.info(
        "Quintile distribution:\n%s",
        households["income_quintile"].value_counts().sort_index().to_string(),
    )

    # Step 4: assign bank
    households = assign_bank(households, rng)

    # Step 5: fingerprints, DOMs, transport
    households = assign_fingerprints(households, rng)

    # Step 6: validate columns match schema
    expected_cols = list(HOUSEHOLD_SCHEMA.keys())
    actual_cols = list(households.columns)
    missing = set(expected_cols) - set(actual_cols)
    extra = set(actual_cols) - set(expected_cols)
    if missing:
        raise ValueError(f"Missing columns in households output: {missing}")
    if extra:
        logger.warning("Extra columns in households output (will be kept): %s", extra)

    # Reorder to match schema
    households = households[expected_cols]

    return households
