"""
app/dashboard.py — FIXED VERSION
Run: streamlit run app/dashboard.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

# ✅ FIXED IMPORTS
from utils.preprocessor import (
    load_and_preprocess,
    get_category_summary,
    get_monthly_summary,
    load_user_profile,
    extract_profile_from_df,
    get_user_ids,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="AI-PFA", layout="wide")

st.title("💎 AI Personal Finance Advisor")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.header("📂 Data")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_demo = st.sidebar.checkbox("Use demo dataset", value=True)

# ─────────────────────────────────────────────
# DATA SOURCE
# ─────────────────────────────────────────────
TMP_PATH = "data/tmp_upload.csv"

if uploaded and not use_demo:
    os.makedirs("data", exist_ok=True)
    with open(TMP_PATH, "wb") as f:
        f.write(uploaded.read())
    source_path = TMP_PATH
else:
    source_path = "data/raw/transactions.csv"

# ─────────────────────────────────────────────
# ✅ FIXED USER DETECTION
# ─────────────────────────────────────────────
try:
    user_ids = get_user_ids(source_path)
except:
    user_ids = []

selected_user = None

if user_ids and len(user_ids) > 0:
    st.sidebar.subheader("👤 Select User")
    selected_user = st.sidebar.selectbox(
        "User",
        options=user_ids,
        format_func=lambda x: f"User {x}"
    )

# ─────────────────────────────────────────────
# LOAD DATA (FIXED)
# ─────────────────────────────────────────────
@st.cache_data
def load_data(path, user):
    return load_and_preprocess(path, selected_user_id=user)

df = load_data(source_path, selected_user)

# ─────────────────────────────────────────────
# BASIC CHECK
# ─────────────────────────────────────────────
if df.empty:
    st.error("No data found.")
    st.stop()

# ─────────────────────────────────────────────
# PROFILE
# ─────────────────────────────────────────────
profile = extract_profile_from_df(df) or load_user_profile()

# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────
total_spent = df["amount"].sum()
monthly_avg = df.groupby("month")["amount"].sum().mean()

col1, col2 = st.columns(2)

col1.metric("💰 Total Spent", f"PKR {total_spent:,.0f}")
col2.metric("📅 Monthly Avg", f"PKR {monthly_avg:,.0f}")

# ─────────────────────────────────────────────
# CATEGORY BREAKDOWN
# ─────────────────────────────────────────────
st.subheader("📊 Category Breakdown")

cat = get_category_summary(df)

st.bar_chart(cat.set_index("category")["total_spent"])

# ─────────────────────────────────────────────
# MONTHLY TREND
# ─────────────────────────────────────────────
st.subheader("📈 Monthly Trend")

monthly = df.groupby("month")["amount"].sum()

st.line_chart(monthly)

# ─────────────────────────────────────────────
# TRANSACTIONS
# ─────────────────────────────────────────────
st.subheader("📋 Transactions")

st.dataframe(df.head(200), use_container_width=True)

# ─────────────────────────────────────────────
# SUCCESS MESSAGE
# ─────────────────────────────────────────────
st.success("✅ Dashboard running successfully!")