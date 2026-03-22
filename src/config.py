"""
Structural configuration for the dataset generator.

Paths, schemas, enums, and scale constants.
Behavioral parameters live in constants.py.
"""
from pathlib import Path
from dataclasses import dataclass, field

# ---------------------
# Paths
# ---------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOOKUPS_DIR = PROJECT_ROOT / "lookups"

# ---------------------
# Reproducibility
# ---------------------
SEED = 42

# ---------------------
# Scale
# ---------------------
N_HOUSEHOLDS = 1000
N_MONTHS = 6
APPROX_TXN_PER_MONTH = 90  # target average per household

# ---------------------
# CCIF 2018.CL — 13 divisions
# ---------------------
# Keys are zero-padded strings matching INE codes.
CCIF_DIVISIONS: dict[str, str] = {
    "01": "Alimentos y bebidas no alcohólicas",
    "02": "Bebidas alcohólicas, tabaco y estupefacientes",
    "03": "Vestuario y calzado",
    "04": "Vivienda, agua, electricidad, gas y otros combustibles",
    "05": "Muebles, artículos para el hogar y conservación del hogar",
    "06": "Salud",
    "07": "Transporte",
    "08": "Información y comunicación",
    "09": "Recreación, deportes y cultura",
    "10": "Servicios de educación",
    "11": "Servicios de restaurantes y alojamiento",
    "12": "Seguros y servicios financieros",
    "13": "Cuidado personal, protección social y bienes y servicios diversos",
}

# Quintile labels (INE standard)
QUINTILES = [1, 2, 3, 4, 5]

# ---------------------
# Banks (Chilean retail)
# ---------------------
BANKS: list[str] = [
    "BCI",
    "BANCO ESTADO",
    "SANTANDER",
    "BANCO DE CHILE",
    "SCOTIABANK",
    "ITAU",
    "FALABELLA",
    "RIPLEY",
    "SECURITY",
    "BICE",
]

# ---------------------
# Transaction schema
# ---------------------
# Field names and their intended types for transactions.parquet.
# This is the reference schema — generation code must produce exactly these columns.
TRANSACTION_SCHEMA: dict[str, str] = {
    "transaction_id": "str",
    "household_id": "str",
    "merchant_id": "str",
    "merchant_name": "str",          # max 64 chars, BPE input
    "amount_clp": "int",             # Chilean pesos, integer
    "iso_timestamp": "datetime64",   # timezone-naive, local Santiago time
    "bank_name": "str",
    "source_type": "str",            # debit | credit | cash
    "ccif_division": "str",          # ground truth label, NOT a training feature
    "ccif_group": "str",             # finer label for post-hoc eval
    "is_recurring": "bool",
}

# Source types
SOURCE_TYPES: list[str] = ["debit", "credit", "cash"]

# ---------------------
# Household schema
# ---------------------
HOUSEHOLD_SCHEMA: dict[str, str] = {
    "household_id": "str",
    "n_personas": "int",
    "tenure_type": "str",            # propietario | arrendatario | otro
    "dwelling_type": "str",          # casa | departamento
    "income_quintile": "int",        # 1-5
    "bank": "str",
    # Temporal fingerprint (drawn once, fixed for 6 months)
    "preferred_lunch_hour": "float",
    "preferred_commute_morning": "float",
    "preferred_commute_evening": "float",
    "preferred_shop_day": "int",     # 0=Monday ... 6=Sunday
    "late_night_shopper": "bool",
    # Fixed billing days (DOM = day of month)
    "dom_enel": "int",
    "dom_agua": "int",
    "dom_gas": "int",
    "dom_telefono": "int",
    "dom_arriendo": "int",           # rent or mortgage payment
    "dom_gastos_comunes": "int",
    "dom_isapre": "int",
    # Transport profile
    "has_car": "bool",
    "uses_rideshare": "bool",
    "uses_public_transport": "bool",
}

# ---------------------
# Merchant pool schema
# ---------------------
MERCHANT_SCHEMA: dict[str, str] = {
    "merchant_id": "str",
    "merchant_name": "str",
    "ccif_division": "str",
    "tier": "int",                   # 1 = shared chain, 2 = household-local
    "household_id": "str|null",      # null for tier 1
}
