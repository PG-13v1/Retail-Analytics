"""
utils.py

Utility/helper functions shared across dashboards and analytics modules.
"""

import pandas as pd
import numpy as np


# -----------------------------
# Data Loading Helpers
# -----------------------------
def safe_read_csv(path):
    """
    Safely load a CSV file with error handling.
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")


# -----------------------------
# Feature Engineering Helpers
# -----------------------------
def add_revenue_column(df, qty_col, price_col):
    """
    Adds Revenue = Quantity * Price column.
    """
    df["Revenue"] = df[qty_col] * df[price_col]
    return df


def convert_to_datetime(df, col):
    """
    Convert column to datetime safely.
    """
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# -----------------------------
# KPI Formatting Helpers
# -----------------------------
def format_currency(value):
    """
    Format numbers as currency string.
    """
    return f"${value:,.0f}"


def format_percent(value):
    """
    Format float as percentage string.
    """
    return f"{value:.2f}%"


# -----------------------------
# Forecast Evaluation Metrics
# -----------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Compute MAPE for forecasting evaluation.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# -----------------------------
# General Analytics Helpers
# -----------------------------
def top_n(df, col, n=10):
    """
    Return top N value counts of a column.
    """
    return df[col].value_counts().head(n)


def missing_summary(df):
    """
    Return missing value summary table.
    """
    return df.isnull().sum().sort_values(ascending=False)
