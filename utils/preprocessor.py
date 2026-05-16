"""
utils/preprocessor.py
Handles all data loading, cleaning, and preprocessing for AI-PFA.
"""

import pandas as pd
import numpy as np
import os
import re


# ── Keyword mapping ─────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "Food & Dining": ["mcdonald","kfc","pizza","biryani","cafe","grocery","imtiaz"],
    "Transport": ["uber","careem","indrive","petrol","fuel"],
    "Utilities": ["kesc","ssgc","internet","ptcl","electricity"],
    "Entertainment": ["netflix","spotify","cinema","youtube"],
    "Shopping": ["daraz","amazon","clothing","electronics"],
    "Healthcare": ["pharmacy","doctor","hospital"],
    "Education": ["university","course","books"],
    "Savings": ["investment","saving","deposit"],
    "Rent": ["rent"],
    "Miscellaneous": []
}


def categorise_transaction(description: str) -> str:
    desc = description.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in desc:
                return cat
    return "Miscellaneous"


# ✅ UPDATED SIGNATURE
def load_and_preprocess(filepath: str, selected_user_id=None, groq_key=None) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Normalize columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ── Multi-user filter ─────────────────────────────
    if selected_user_id is not None:
        for col in ["user_id", "userid", "user"]:
            if col in df.columns:
                df = df[df[col] == selected_user_id]
                break

    # Required columns
    required = {"date", "description", "amount"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    # Date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    # Amount
    df["amount"] = pd.to_numeric(
        df["amount"].astype(str).str.replace(r"[^\d.]", "", regex=True),
        errors="coerce"
    )
    df = df[df["amount"] > 0]

    # Description
    df["description"] = df["description"].astype(str)

    # Category
    if "category" not in df.columns:
        df["category"] = df["description"].apply(categorise_transaction)

    # Derived
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["day_of_week"] = df["date"].dt.day_name()

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def get_category_summary(df):
    return (
        df.groupby("category")["amount"]
        .sum()
        .reset_index(name="total_spent")
    )


def get_monthly_summary(df):
    return (
        df.groupby(["month", "category"])["amount"]
        .sum()
        .reset_index(name="total_spent")
    )


# ✅ NEW FUNCTION (fixes your error)
def get_user_ids(file_path: str):
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.lower() for c in df.columns]

        for col in ["user_id", "userid", "user"]:
            if col in df.columns:
                return sorted(df[col].dropna().unique().tolist())

    except Exception as e:
        print(f"Error: {e}")

    return []


# ── Profile ─────────────────────────────────────────────
def load_user_profile(path="data/raw/user_profile.json"):
    import json
    if os.path.exists(path):
        return json.load(open(path))

    return {
        "credit_score": 680,
        "savings_rate": 0.1,
        "debt_to_income_ratio": 0.35,
        "financial_stress_level": "Medium",
        "monthly_income_pkr": 80000,
    }


def extract_profile_from_df(df):
    return None