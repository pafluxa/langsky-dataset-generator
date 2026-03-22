"""
Behavioral parameters for the dataset generator.

All generation-controlling tables: p_local gradients, temporal profiles,
cash-vs-card gradients, transport sub-streams, recurring billing parameters,
and household fingerprint ranges.

Structural definitions (paths, schemas, enums) live in config.py.
"""

# ---------------------
# p_local: probability of tier 2 (local) merchant by division × quintile
# ---------------------
# Rows: CCIF division code (str). Columns: quintile 1-5.
# Higher p_local = more local/informal purchases.
P_LOCAL: dict[str, dict[int, float]] = {
    "01": {1: 0.6, 2: 0.5, 3: 0.4, 4: 0.3, 5: 0.2},   # Alimentos
    "02": {1: 0.5, 2: 0.4, 3: 0.3, 4: 0.2, 5: 0.1},   # Bebidas/tabaco
    "03": {1: 0.5, 2: 0.4, 3: 0.3, 4: 0.2, 5: 0.1},   # Vestuario
    "04": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},   # Vivienda (monopoly)
    "05": {1: 0.5, 2: 0.4, 3: 0.3, 4: 0.2, 5: 0.1},   # Muebles
    "06": {1: 0.5, 2: 0.4, 3: 0.3, 4: 0.2, 5: 0.1},   # Salud
    "07": {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1},  # Transporte
    "08": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},   # Info/comunicación
    "09": {1: 0.7, 2: 0.6, 3: 0.5, 4: 0.4, 5: 0.3},   # Recreación
    "10": {1: 0.8, 2: 0.7, 3: 0.6, 4: 0.5, 5: 0.4},   # Educación
    "11": {1: 0.8, 2: 0.7, 3: 0.55, 4: 0.4, 5: 0.3},  # Restaurantes
    "12": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},   # Seguros/financieros
    "13": {1: 0.7, 2: 0.6, 3: 0.45, 4: 0.3, 5: 0.2},  # Cuidado personal
}

# ---------------------
# Cash vs card gradient by quintile
# ---------------------
# Base probability of paying cash (for eligible transactions).
# Modulated per division and amount range at generation time.
CASH_BASE_PROB: dict[int, float] = {
    1: 0.70,
    2: 0.55,
    3: 0.35,
    4: 0.20,
    5: 0.08,
}

# Divisions where cash is common even at higher quintiles
CASH_HEAVY_DIVISIONS: list[str] = ["01", "07", "11"]

# Amount threshold (CLP): below this, cash probability is boosted
CASH_AMOUNT_THRESHOLD: int = 50_000

# ---------------------
# Temporal meta-groups
# ---------------------
# Each meta-group defines the temporal generation profile.
# tod = time-of-day parameters (mean hour, std hours) for a normal distribution
# dow = day-of-week weights (Mon=0 ... Sun=6), relative (will be normalized)
# amount_range = (min_clp, max_clp) typical range
# is_recurring = whether transactions have fixed DOM

TEMPORAL_PROFILES: dict[str, dict] = {
    "recurring": {
        "divisions": ["04", "08", "12"],  # + div 06 isapre, div 09 gym (handled specially)
        "tod": {"mean": 12.0, "std": 3.0},  # business hours
        "dow": [1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 0.1],  # weekday dominant
        "is_recurring": True,
    },
    "daily_necessities": {
        "divisions": ["01", "07"],  # almacén + transport
        "tod_commute_morning": {"mean": 7.5, "std": 0.7},
        "tod_commute_evening": {"mean": 18.0, "std": 0.7},
        "tod_lunch": {"mean": 13.0, "std": 0.5},
        "dow": [1.2, 1.2, 1.2, 1.2, 1.2, 0.8, 0.5],  # weekday peak
        "is_recurring": False,
    },
    "weekly_shopping": {
        "divisions": ["01", "11"],  # supermarket + restaurant
        "tod": {"mean": 13.0, "std": 3.0},  # midday/evening
        "dow": [0.5, 0.5, 0.5, 1.0, 0.8, 1.5, 1.5],  # Thu + weekend peak
        "is_recurring": False,
    },
    "irregular_leisure": {
        "divisions": ["03", "05", "09", "11", "13"],
        "tod": {"mean": 15.0, "std": 4.0},  # broad window
        "dow": [0.4, 0.4, 0.5, 0.6, 0.8, 1.5, 1.5],  # weekend-biased
        "is_recurring": False,
    },
}

# ---------------------
# Transport sub-streams (div 07)
# ---------------------
TRANSPORT_SUBSTREAMS: dict[str, dict] = {
    "fuel": {
        "tod": {"mean": 8.0, "std": 2.0},
        "dow": [1.5, 0.8, 0.8, 0.8, 1.5, 0.5, 0.3],  # Mon+Fri
        "amount_range": (20_000, 60_000),
        "source_type": "debit",  # mostly card
        "requires": "has_car",
    },
    "public_transport": {
        "tod_morning": {"mean": 7.5, "std": 0.5},
        "tod_evening": {"mean": 18.0, "std": 0.5},
        "dow": [1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.1],  # weekday only
        "amount_range": (800, 3_500),
        "cash_override": {1: 0.8, 2: 0.6, 3: 0.3, 4: 0.1, 5: 0.05},
        "requires": "uses_public_transport",
    },
    "rideshare": {
        "tod_night": {"mean": 23.0, "std": 1.5},
        "tod_lunch": {"mean": 13.0, "std": 1.0},
        "dow": [0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.0],  # Fri/Sat night + weekday lunch
        "amount_range": (3_000, 25_000),
        "source_type": "debit",
        "requires": "uses_rideshare",
    },
}

# ---------------------
# Recurring billing parameters
# ---------------------
# Divisions (or sub-categories) that generate monthly recurring transactions.
# dom_field = which household field provides the billing day.
RECURRING_BILLS: list[dict] = [
    {"division": "04", "label": "ENEL DISTRIBUCION", "dom_field": "dom_enel",
     "amount_range": (15_000, 120_000)},
    {"division": "04", "label": "AGUAS ANDINAS", "dom_field": "dom_agua",
     "amount_range": (8_000, 45_000)},
    {"division": "04", "label": "METROGAS", "dom_field": "dom_gas",
     "amount_range": (5_000, 50_000)},
    {"division": "08", "label": "ENTEL|MOVISTAR|WOM|CLARO", "dom_field": "dom_telefono",
     "amount_range": (10_000, 45_000)},
    {"division": "04", "label": "ARRIENDO|DIVIDENDO", "dom_field": "dom_arriendo",
     "amount_range": (150_000, 900_000), "tenure_filter": ["arrendatario", "propietario_deuda"]},
    {"division": "04", "label": "GASTOS COMUNES", "dom_field": "dom_gastos_comunes",
     "amount_range": (20_000, 150_000), "tenure_filter": ["propietario", "propietario_deuda"]},
    {"division": "06", "label": "ISAPRE|FONASA", "dom_field": "dom_isapre",
     "amount_range": (30_000, 250_000)},
]

# ---------------------
# Household fingerprint parameter ranges
# ---------------------
# Used when sampling household temporal fingerprints.
FINGERPRINT_RANGES: dict[str, tuple[float, float]] = {
    "preferred_lunch_hour": (12.0, 14.0),
    "preferred_commute_morning": (6.5, 8.5),
    "preferred_commute_evening": (17.0, 19.5),
}

# preferred_shop_day: drawn uniformly from 0-6 (Mon-Sun), no range needed
# late_night_shopper: Bernoulli with p=0.15

LATE_NIGHT_PROB: float = 0.15

# DOM (day of month) for billing: drawn uniformly from this range
DOM_RANGE: tuple[int, int] = (1, 28)

# ---------------------
# Transport profile probabilities by quintile
# ---------------------
# P(has_car), P(uses_rideshare), P(uses_public_transport) by quintile
TRANSPORT_PROFILE_PROB: dict[str, dict[int, float]] = {
    "has_car": {1: 0.15, 2: 0.30, 3: 0.50, 4: 0.70, 5: 0.85},
    "uses_rideshare": {1: 0.05, 2: 0.15, 3: 0.35, 4: 0.55, 5: 0.70},
    "uses_public_transport": {1: 0.90, 2: 0.80, 3: 0.60, 4: 0.35, 5: 0.15},
}

# ---------------------
# Quintile-segmented tier 1 pools
# ---------------------
# Divisions where tier 1 merchants differ by quintile bracket.
# Keys are division codes. Values map quintile ranges to merchant sublists.
# Actual merchant names live in lookups/tier1_pool.json; this controls the logic.
TIER1_QUINTILE_SEGMENTED: dict[str, dict[str, list[int]]] = {
    "06": {
        "premium": [4, 5],     # CLINICA LAS CONDES, CLINICA ALEMANA, etc.
        "standard": [1, 2, 3], # SALCOBRAND, CRUZ VERDE, AHUMADA, CESFAM
    },
    "07": {
        "premium": [4, 5],     # UBER, CABIFY, COPEC
        "standard": [1, 2, 3], # BIP/TRANSANTIAGO, COPEC
    },
    "11": {
        "premium": [4, 5],     # STARBUCKS, RAPPI, UBER EATS, DOMINOS
        "standard": [1, 2, 3], # local (handled by p_local), MCDONALDS
    },
}
