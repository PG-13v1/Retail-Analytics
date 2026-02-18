"""
config.py

Central configuration file for the Retail Analytics Suite.
Keeps paths, constants, and global settings in one place.
"""

from pathlib import Path

# -----------------------------
# Base Directory
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# -----------------------------
# Data Directories
# -----------------------------
DATA_DIR = BASE_DIR / "data"

ONLINE_RETAIL_DIR = DATA_DIR / "online_retail"
WALMART_DIR = DATA_DIR / "walmart"
INSTACART_DIR = DATA_DIR / "instacart"

# -----------------------------
# Dataset Files
# -----------------------------
ONLINE_RETAIL_FILE = ONLINE_RETAIL_DIR / "online_retail.csv"

WALMART_TRAIN_FILE = WALMART_DIR / "train.csv"
WALMART_STORES_FILE = WALMART_DIR / "stores.csv"
WALMART_FEATURES_FILE = WALMART_DIR / "features.csv"

INSTACART_ORDERS_FILE = INSTACART_DIR / "orders.csv"
INSTACART_PRODUCTS_FILE = INSTACART_DIR / "products.csv"
INSTACART_ORDER_PRODUCTS_FILE = INSTACART_DIR / "order_products__train.csv"

# -----------------------------
# Model + Analytics Parameters
# -----------------------------
DEFAULT_FORECAST_DAYS = 30
DEFAULT_CLUSTER_COUNT = 4
ANOMALY_CONTAMINATION = 0.02
