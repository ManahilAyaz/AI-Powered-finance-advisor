"""
data/generate_dataset.py
────────────────────────────────────────────────────────────────────────────────
Generates a 2,500-row transaction-level dataset enriched with financial health
profile columns drawn from the Kaggle Personal Finance Tracker schema:
  credit_score, debt_to_income_ratio, savings_rate, savings_goal_met,
  financial_stress_level, income_type, emergency_fund_months,
  subscription_services, loan_payment_pkr, investment_amount_pkr,
  cash_flow_status, financial_scenario, fraud_flag, financial_advice_score

These are attached at the user-profile level (one consistent profile per
generated dataset run) and repeated across all transactions — matching how
the Kaggle data works (per-user monthly snapshot) but embedded in our
transaction-level format so the dashboard can surface them.

Using the real Kaggle CSV:
  → The preprocessor only reads: date, description, amount, category columns.
  → The financial profile panel will use default/simulated profile values
    since the Kaggle CSV has no individual transactions.
  → For best results keep using this generated dataset.

Run: python data/generate_dataset.py
────────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os, json

np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────────
N_TRANSACTIONS  = 2500
START_DATE      = datetime(2023, 1, 1)
DATE_RANGE_DAYS = 730          # 2 full years

# Seasonal spend multipliers (month → multiplier)
SEASONAL = {
    1: 0.90, 2: 0.85, 3: 1.00, 4: 1.20,   # Apr: Eid / Ramadan
    5: 1.15, 6: 0.95, 7: 0.90, 8: 0.90,
    9: 1.00, 10: 1.05, 11: 1.10, 12: 1.30, # Dec: year-end spending
}

# ── Category config ────────────────────────────────────────────────────────────
CATEGORIES = {
    "Food & Dining":  {"min": 200,   "max": 3000,  "weight": 0.25},
    "Transport":      {"min": 50,    "max": 1500,  "weight": 0.15},
    "Utilities":      {"min": 500,   "max": 5000,  "weight": 0.10},
    "Entertainment":  {"min": 100,   "max": 2000,  "weight": 0.12},
    "Shopping":       {"min": 300,   "max": 8000,  "weight": 0.13},
    "Healthcare":     {"min": 200,   "max": 6000,  "weight": 0.08},
    "Education":      {"min": 500,   "max": 10000, "weight": 0.05},
    "Savings":        {"min": 1000,  "max": 20000, "weight": 0.05},
    "Rent":           {"min": 5000,  "max": 30000, "weight": 0.04},
    "Miscellaneous":  {"min": 50,    "max": 2000,  "weight": 0.03},
}

DESCRIPTIONS = {
    "Food & Dining":  ["McDonald's","KFC","Pizza Hut","Biryani House","Café Latte",
                       "Carrefour Grocery","Imtiaz Superstore","Kababjees","Hardee's",
                       "FoodPanda Order","Cheetay Delivery","Student Biryani","Burns Road"],
    "Transport":      ["Uber","Careem","InDrive","Petrol Station","Bykea",
                       "Parking Fee","Toll Tax","Metro BRT","CNG Station","Careem Bike"],
    "Utilities":      ["KESC Electricity Bill","SSGC Gas Bill","Nayatel Internet",
                       "PTCL Broadband","Jazz Internet","Telenor Postpaid","Water Bill"],
    "Entertainment":  ["Netflix","Spotify","Cinema Ticket","YouTube Premium",
                       "Amazon Prime","Video Game","Bowling","Concert Ticket","Daraz Entertainment"],
    "Shopping":       ["Daraz","Khaadi","Sana Safinaz","Al-Fatah","Chase Up",
                       "Electronics Store","Shoe Store","Clothing Store","OLX Purchase"],
    "Healthcare":     ["Pharmacy","Doctor Visit","Lab Test","Hospital Bill",
                       "Dentist","Optician","Dawaai Pharmacy","Health Supplement"],
    "Education":      ["University Fee","Coursera","Udemy","Books","Stationery",
                       "Tuition Fee","Online Course","Exam Fee"],
    "Savings":        ["Bank Transfer – Savings","Meezan Savings","Jazz Cash Saving",
                       "Prize Bond","Mutual Fund","Investment Transfer"],
    "Rent":           ["Monthly Rent","House Rent","Apartment Rent"],
    "Miscellaneous":  ["ATM Withdrawal","Bank Charges","Gift Purchase","Donation",
                       "Haircut","Laundry","Repair Service","Postage"],
}

PAYMENT_METHODS = ["Cash","Credit Card","Debit Card","JazzCash","EasyPaisa"]


# ══════════════════════════════════════════════════════════════════════════════
#  FINANCIAL PROFILE GENERATOR
#  Produces a realistic single-user financial health profile using the
#  distributions observed in the Kaggle Personal Finance Tracker dataset.
# ══════════════════════════════════════════════════════════════════════════════

def generate_financial_profile(monthly_income_pkr: float) -> dict:
    """
    Generate a correlated financial health profile for one user.
    Distributions calibrated from Kaggle dataset (converted to PKR where needed).
    USD→PKR rate used: 280 (approximate 2023–2024 average).
    """
    USD_TO_PKR = 280

    # Income type (Salary 72%, Freelance 18%, Mixed 11% — from Kaggle)
    income_type = np.random.choice(
        ["Salary", "Freelance", "Mixed"],
        p=[0.718, 0.176, 0.106],
    )

    # Savings rate: Kaggle mean=0.23, std=0.10, range [0.05, 0.40]
    savings_rate = float(np.clip(np.random.normal(0.23, 0.10), 0.05, 0.40))

    # Credit score: mean=680, std=50, range [515, 847]
    credit_score = int(np.clip(np.random.normal(680, 50), 515, 847))

    # Debt-to-income: mean=0.35, std=0.15, range [0.10, 0.60]
    debt_to_income_ratio = float(np.clip(np.random.normal(0.35, 0.15), 0.10, 0.60))

    # Financial stress: Low 51%, Medium 30%, High 20% — Kaggle proportions
    financial_stress_level = np.random.choice(
        ["Low", "Medium", "High"],
        p=[0.507, 0.295, 0.198],
    )

    # Stress affects spending behaviour — modulate category weights
    stress_multiplier = {"Low": 1.0, "Medium": 1.15, "High": 1.30}[financial_stress_level]

    # Cash flow status — correlated with stress
    cf_probs = {
        "Low":    [0.75, 0.20, 0.05],   # Positive, Neutral, Negative
        "Medium": [0.45, 0.35, 0.20],
        "High":   [0.20, 0.35, 0.45],
    }[financial_stress_level]
    cash_flow_status = np.random.choice(["Positive", "Neutral", "Negative"], p=cf_probs)

    # Emergency fund: Kaggle mean~1005 USD → 281,400 PKR, std~485 USD → 135,800 PKR
    emergency_fund_pkr = float(np.clip(
        np.random.normal(1005 * USD_TO_PKR, 485 * USD_TO_PKR), 0, 2585 * USD_TO_PKR
    ))
    emergency_fund_months = round(
        emergency_fund_pkr / (monthly_income_pkr * (1 - savings_rate)), 1
    ) if monthly_income_pkr > 0 else 0.0

    # Subscription services: Kaggle mean=5, std=2.6, range [1,9]
    subscription_services = int(np.clip(np.random.normal(5, 2.6), 1, 9))

    # Loan payment in PKR (Kaggle mean~509 USD → 142,520 PKR)
    loan_payment_pkr = float(np.clip(
        np.random.normal(509 * USD_TO_PKR, 200 * USD_TO_PKR), 0, 1177 * USD_TO_PKR
    ))

    # Investment amount PKR (Kaggle mean~401 USD → 112,280 PKR)
    investment_amount_pkr = float(np.clip(
        np.random.normal(401 * USD_TO_PKR, 235 * USD_TO_PKR), 0, 1292 * USD_TO_PKR
    ))

    # Savings goal met: ~9% met regardless of stress (Kaggle finding)
    savings_goal_met = int(np.random.random() < 0.09)

    # Financial scenario: normal 58%, inflation 22%, recession 20%
    financial_scenario = np.random.choice(
        ["normal", "inflation", "recession"],
        p=[0.580, 0.216, 0.204],
    )

    # Fraud flag: ~2.4% of users (Kaggle)
    fraud_flag = int(np.random.random() < 0.024)

    # Financial advice score: uniform [0.1, 100]
    financial_advice_score = float(np.round(np.random.uniform(0.1, 100.0), 1))

    return {
        "income_type":             income_type,
        "savings_rate":            round(savings_rate, 3),
        "credit_score":            credit_score,
        "debt_to_income_ratio":    round(debt_to_income_ratio, 3),
        "financial_stress_level":  financial_stress_level,
        "cash_flow_status":        cash_flow_status,
        "emergency_fund_pkr":      round(emergency_fund_pkr, 0),
        "emergency_fund_months":   emergency_fund_months,
        "subscription_services":   subscription_services,
        "loan_payment_pkr":        round(loan_payment_pkr, 0),
        "investment_amount_pkr":   round(investment_amount_pkr, 0),
        "savings_goal_met":        savings_goal_met,
        "financial_scenario":      financial_scenario,
        "fraud_flag":              fraud_flag,
        "financial_advice_score":  financial_advice_score,
        "stress_multiplier":       stress_multiplier,   # internal use only
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSACTION GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_transactions(profile: dict, monthly_income_pkr: float) -> pd.DataFrame:
    """Generate N_TRANSACTIONS rows with profile values stamped on every row."""
    categories = list(CATEGORIES.keys())
    weights    = np.array([CATEGORIES[c]["weight"] for c in categories])

    # Stress shifts weight toward essential categories
    if profile["financial_stress_level"] == "High":
        essential_idx = [categories.index(c) for c in ["Utilities","Rent","Healthcare","Food & Dining"]]
        weights[essential_idx] *= 1.3
        weights /= weights.sum()
    elif profile["financial_stress_level"] == "Low":
        discretionary_idx = [categories.index(c) for c in ["Entertainment","Shopping","Education"]]
        weights[discretionary_idx] *= 1.2
        weights /= weights.sum()

    stress_mult = profile["stress_multiplier"]
    rows = []

    for i in range(N_TRANSACTIONS):
        cat  = np.random.choice(categories, p=weights)
        cfg  = CATEGORIES[cat]
        date = START_DATE + timedelta(days=np.random.randint(0, DATE_RANGE_DAYS))

        # Amount: base × seasonal × stress × noise
        season = SEASONAL[date.month]
        base   = np.random.uniform(cfg["min"], cfg["max"])
        amount = round(base * season * stress_mult * np.random.uniform(0.85, 1.15), 2)

        # Fraud flag → occasional anomalously large transaction
        is_fraud = (profile["fraud_flag"] == 1 and np.random.random() < 0.008)
        if is_fraud:
            amount = round(amount * np.random.uniform(3.0, 6.0), 2)

        desc = np.random.choice(DESCRIPTIONS[cat])

        row = {
            # ── Core transaction fields ────────────────────────────────────────
            "transaction_id":       f"TXN{i+1:04d}",
            "date":                 date.strftime("%Y-%m-%d"),
            "description":          desc,
            "amount":               amount,
            "category":             cat,
            "payment_method":       np.random.choice(PAYMENT_METHODS),
            "month":                date.strftime("%Y-%m"),
            "day_of_week":          date.strftime("%A"),
            "is_fraud":             int(is_fraud),

            # ── Kaggle-sourced financial profile fields ────────────────────────
            "income_type":              profile["income_type"],
            "credit_score":             profile["credit_score"],
            "debt_to_income_ratio":     profile["debt_to_income_ratio"],
            "savings_rate":             profile["savings_rate"],
            "financial_stress_level":   profile["financial_stress_level"],
            "cash_flow_status":         profile["cash_flow_status"],
            "emergency_fund_pkr":       profile["emergency_fund_pkr"],
            "emergency_fund_months":    profile["emergency_fund_months"],
            "subscription_services":    profile["subscription_services"],
            "loan_payment_pkr":         profile["loan_payment_pkr"],
            "investment_amount_pkr":    profile["investment_amount_pkr"],
            "savings_goal_met":         profile["savings_goal_met"],
            "financial_scenario":       profile["financial_scenario"],
            "fraud_flag":               profile["fraud_flag"],
            "financial_advice_score":   profile["financial_advice_score"],
            "monthly_income_pkr":       monthly_income_pkr,
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    MONTHLY_INCOME_PKR = 80_000   # default user income

    out_path    = "data/raw/transactions.csv"
    profile_path = "data/raw/user_profile.json"
    os.makedirs("data/raw", exist_ok=True)

    # Don't overwrite a larger real file placed manually
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, nrows=5)
        if "credit_score" not in existing.columns:
            print("Existing CSV lacks enrichment columns — regenerating with enriched schema.")
        else:
            row_count = sum(1 for _ in open(out_path)) - 1
            if row_count > N_TRANSACTIONS:
                print(f"Existing dataset has {row_count} rows — skipping generation.")
                print("Delete data/raw/transactions.csv to force regeneration.")
                exit(0)

    # Generate profile then transactions
    profile = generate_financial_profile(MONTHLY_INCOME_PKR)
    df      = generate_transactions(profile, MONTHLY_INCOME_PKR)

    # Save transactions
    df.to_csv(out_path, index=False)

    # Save profile separately for quick dashboard loading
    profile_out = {k: v for k, v in profile.items() if k != "stress_multiplier"}
    profile_out["monthly_income_pkr"] = MONTHLY_INCOME_PKR
    with open(profile_path, "w") as f:
        json.dump(profile_out, f, indent=2)

    print(f"✅  Dataset : {len(df):,} transactions · {df['month'].nunique()} months → {out_path}")
    print(f"✅  Profile : → {profile_path}")
    print()
    print(f"  Income type          : {profile['income_type']}")
    print(f"  Credit score         : {profile['credit_score']}")
    print(f"  Debt-to-income       : {profile['debt_to_income_ratio']:.2f}")
    print(f"  Savings rate         : {profile['savings_rate']*100:.1f}%")
    print(f"  Financial stress     : {profile['financial_stress_level']}")
    print(f"  Cash flow            : {profile['cash_flow_status']}")
    print(f"  Emergency fund       : PKR {profile['emergency_fund_pkr']:,.0f} ({profile['emergency_fund_months']:.1f} months)")
    print(f"  Subscriptions        : {profile['subscription_services']}")
    print(f"  Loan payment/mo      : PKR {profile['loan_payment_pkr']:,.0f}")
    print(f"  Investment/mo        : PKR {profile['investment_amount_pkr']:,.0f}")
    print(f"  Savings goal met     : {'Yes' if profile['savings_goal_met'] else 'No'}")
    print(f"  Financial scenario   : {profile['financial_scenario']}")
    print(f"  Fraud flag           : {'Yes' if profile['fraud_flag'] else 'No'}")
    print(f"  Advice score         : {profile['financial_advice_score']}/100")
    print()
    print("  Kaggle columns integrated: credit_score, debt_to_income_ratio,")
    print("  savings_rate, savings_goal_met, financial_stress_level, income_type,")
    print("  emergency_fund_pkr, subscription_services, loan_payment_pkr,")
    print("  investment_amount_pkr, cash_flow_status, financial_scenario,")
    print("  fraud_flag, financial_advice_score")
    print()
    print("To use real Kaggle CSV:")
    print("  1. Download personal_finance_tracker_dataset.csv from Kaggle")
    print("  2. Place at data/raw/kaggle_reference.csv (for profile reference only)")
    print("  3. Our transaction CSV remains the primary data source")
