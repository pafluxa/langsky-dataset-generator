"""
Step 2: Build the merchant pool.

Tier 1 — shared chain merchants loaded from lookups/tier1_pool.json.
          Every household can draw from these. Segmented divisions (06, 07, 11)
          have premium/standard sublists selected by quintile.

Tier 2 — procedurally generated local merchant names, unique per household
          per division. Built from templates in lookups/tier2_templates.json.

Output: merchant_pool DataFrame matching MERCHANT_SCHEMA.
"""
import json
import logging

import numpy as np
import pandas as pd

from config import (
    CCIF_DIVISIONS,
    LOOKUPS_DIR,
    MERCHANT_SCHEMA,
    SEED,
)
from constants import (
    P_LOCAL,
    TIER1_QUINTILE_SEGMENTED,
)

logger = logging.getLogger(__name__)

# Number of tier 2 merchants to pre-generate per household per division.
# These form the household's "local neighborhood" — at transaction time,
# one is drawn uniformly from this pool.
TIER2_POOL_SIZE_PER_DIVISION = 5


def load_tier1_pool() -> dict[str, list[str] | dict[str, list[str]]]:
    """
    Load tier 1 merchant names from lookups/tier1_pool.json.

    Returns dict keyed by division code. Values are either:
    - list[str] for non-segmented divisions
    - dict with "standard" and "premium" keys for segmented divisions
    """
    path = LOOKUPS_DIR / "tier1_pool.json"
    with open(path) as f:
        pool = json.load(f)
    logger.info("Loaded tier 1 pool: %d divisions", len(pool))
    return pool


def load_tier2_templates() -> dict[str, dict]:
    """
    Load tier 2 name generation templates from lookups/tier2_templates.json.

    Returns dict keyed by division code. Each value has:
    prefixes, suffixes, numbered, p_suffix, p_number.
    """
    path = LOOKUPS_DIR / "tier2_templates.json"
    with open(path) as f:
        templates = json.load(f)
    logger.info("Loaded tier 2 templates: %d divisions", len(templates))
    return templates


def get_tier1_merchants_for_household(
    tier1_pool: dict,
    division: str,
    quintile: int,
) -> list[str]:
    """
    Return the tier 1 merchant list for a given division and quintile.

    For segmented divisions (06, 07, 11), selects premium or standard
    sublist based on the household's quintile.
    For non-segmented divisions, returns the full list.
    Returns empty list if division has no tier 1 pool.
    """
    entry = tier1_pool.get(division)
    if entry is None:
        return []

    if isinstance(entry, list):
        return entry

    # Segmented division — determine which segment this quintile belongs to
    segments = TIER1_QUINTILE_SEGMENTED.get(division, {})
    for segment_name, quintile_list in segments.items():
        if quintile in quintile_list:
            return entry.get(segment_name, [])

    # Fallback: if quintile not found in any segment, use all values
    all_merchants = []
    for sublist in entry.values():
        if isinstance(sublist, list):
            all_merchants.extend(sublist)
    return all_merchants


def generate_tier2_name(
    template: dict,
    rng: np.random.Generator,
) -> str:
    """
    Generate a single tier 2 merchant name from a template.

    Rule:
        name = choice(prefixes)
        if random < p_suffix: name += " " + choice(suffixes)
        if numbered and random < p_number: name += " N°" + randint(1,999)
        return name[:64].upper()
    """
    name = rng.choice(template["prefixes"])

    if template["suffixes"] and rng.random() < template["p_suffix"]:
        name = name + " " + rng.choice(template["suffixes"])

    if template["numbered"] and rng.random() < template["p_number"]:
        num = rng.integers(1, 1000)
        name = name + f" N\u00b0{num}"

    return name[:64].upper()


def generate_tier2_pool_for_household(
    household_id: str,
    quintile: int,
    templates: dict[str, dict],
    rng: np.random.Generator,
) -> list[dict]:
    """
    Generate tier 2 local merchants for one household across all eligible divisions.

    Only generates for divisions that:
    1. Have a template in tier2_templates.json
    2. Have p_local > 0 for this quintile

    Returns list of merchant dicts (not yet a DataFrame).
    """
    merchants = []
    merchant_counter = 0

    for division, template in templates.items():
        # Check if this division has any local probability for this quintile
        div_p_local = P_LOCAL.get(division, {})
        if div_p_local.get(quintile, 0.0) <= 0.0:
            continue

        # Generate a pool of unique local names for this household × division
        generated_names: set[str] = set()
        attempts = 0
        max_attempts = TIER2_POOL_SIZE_PER_DIVISION * 5  # safety valve

        while len(generated_names) < TIER2_POOL_SIZE_PER_DIVISION and attempts < max_attempts:
            name = generate_tier2_name(template, rng)
            generated_names.add(name)
            attempts += 1

        for name in generated_names:
            merchants.append({
                "merchant_id": f"{household_id}-T2-{merchant_counter:03d}",
                "merchant_name": name,
                "ccif_division": division,
                "tier": 2,
                "household_id": household_id,
            })
            merchant_counter += 1

    return merchants


def build_tier1_table(tier1_pool: dict) -> pd.DataFrame:
    """
    Flatten the tier 1 pool into a merchant table.

    Tier 1 merchants are shared across all households (household_id = None).
    For segmented divisions, all sublists are included — selection by quintile
    happens at transaction time, not here.
    Deduplicates by (merchant_name, division) to avoid repeated entries
    when a name appears in both premium and standard.
    """
    rows = []
    counter = 0

    for division, entry in tier1_pool.items():
        names: list[str] = []
        if isinstance(entry, list):
            names = entry
        elif isinstance(entry, dict):
            for sublist in entry.values():
                names.extend(sublist)

        # Deduplicate within this division
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            rows.append({
                "merchant_id": f"T1-{counter:04d}",
                "merchant_name": name,
                "ccif_division": division,
                "tier": 1,
                "household_id": None,
            })
            counter += 1

    df = pd.DataFrame(rows)
    logger.info("Tier 1 table: %d unique merchants across %d divisions",
                len(df), df["ccif_division"].nunique())
    return df


def build_merchant_pool(
    households: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Main entry point: build the complete merchant pool.

    Pipeline:
    1. Load tier 1 pool from JSON → flatten into shared merchant table
    2. Load tier 2 templates from JSON
    3. For each household, generate tier 2 local merchants
    4. Concatenate tier 1 + all tier 2 tables
    5. Validate schema and return

    Parameters
    ----------
    households : pd.DataFrame
        Output from census.generate_households(). Must have columns:
        household_id, income_quintile.

    Returns
    -------
    pd.DataFrame
        Merchant pool matching MERCHANT_SCHEMA.
    """
    rng = np.random.default_rng(seed)

    # Step 1: tier 1
    tier1_pool = load_tier1_pool()
    tier1_df = build_tier1_table(tier1_pool)

    # Step 2: tier 2 templates
    templates = load_tier2_templates()

    # Step 3: generate tier 2 for each household
    tier2_rows: list[dict] = []
    for _, hh in households.iterrows():
        hh_merchants = generate_tier2_pool_for_household(
            household_id=hh["household_id"],
            quintile=hh["income_quintile"],
            templates=templates,
            rng=rng,
        )
        tier2_rows.extend(hh_merchants)

    tier2_df = pd.DataFrame(tier2_rows)
    logger.info(
        "Tier 2 table: %d merchants across %d households",
        len(tier2_df),
        households["household_id"].nunique(),
    )

    # Step 4: concatenate
    merchant_pool = pd.concat([tier1_df, tier2_df], ignore_index=True)

    # Step 5: validate
    expected_cols = list(MERCHANT_SCHEMA.keys())
    actual_cols = list(merchant_pool.columns)
    missing = set(expected_cols) - set(actual_cols)
    if missing:
        raise ValueError(f"Missing columns in merchant pool: {missing}")

    merchant_pool = merchant_pool[expected_cols]

    logger.info(
        "Merchant pool complete: %d total (%d tier1, %d tier2)",
        len(merchant_pool),
        len(tier1_df),
        len(tier2_df),
    )

    return merchant_pool
