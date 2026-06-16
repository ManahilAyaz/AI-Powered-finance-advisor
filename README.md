# 💎 AI Personal Finance Advisor (AI-PFA)

> **FAST-NUCES Karachi · April 2026**  
> Kashmala Saghar (23K-3034) · Manahil Ayaz (23K-3018) · Syeda Tayyiba Fatima (23k-0853) . FSR-AI-PFA-2026-001

**Summary**
AI Personal Finance Advisor (AI-PFA) is a machine learning-powered personal finance management system that analyzes transaction data, categorizes expenses, detects overspending patterns, predicts future spending, and provides personalized financial recommendations. The platform also features an AI chatbot that helps users gain actionable insights and make smarter financial decisions.


AI-PFA analyses personal transaction data to deliver:

| Feature | Detail |
|---|---|
| **Auto-categorisation** | Rule-based keyword matching — 10 categories, ≥90% accuracy |
| **6 Smart Alert Types** | Burn rate · Acceleration · Merchant spike · Category creep · Savings erosion · Weekend drain |
| **Product-Level Recommendations** | Named Pakistani alternatives with real PKR prices (InDrive vs Careem, generic vs Panadol) |
| **Groq AI Chatbot** | Llama 3.3 70B — free, knows your full spending history |
| **Financial Health Profile** | Composite score + 15 Kaggle-enriched columns (credit score, DTI, stress level…) |
| **ML Predictions** | GradientBoosting R²=0.965 — next month forecast per category |

---

## Project Structure

```
AI-PFA/
├── app/
│   └── dashboard.py              ← Streamlit dashboard (7 tabs, ~1000 lines)
├── data/
│   ├── generate_dataset.py       ← Enriched 2,500-row generator (15 Kaggle columns)
│   ├── raw/
│   │   ├── transactions.csv      ← Generated dataset
│   │   └── user_profile.json     ← Financial health profile sidecar
│   └── processed/                ← Auto-generated clean CSV
├── models/
│   ├── spending_predictor.py     ← GradientBoosting + RandomForest (15 features)
│   └── best_model.pkl            ← Saved model (auto-generated)
├── utils/
│   ├── preprocessor.py           ← Data loading, cleaning, categorisation, profile loading
│   ├── overspending_detector.py  ← 6-type smart alert engine
│   └── recommender.py            ← Product-aware recommender + Groq chatbot context
├── notebooks/
│   └── 01_eda_and_training.ipynb
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate demo dataset
```bash
python data/generate_dataset.py
```

### 3. Get a free Groq API key (for chatbot)
1. Go to **console.groq.com** → sign up (no credit card)
2. Create API key → copy it
3. Either paste it in the sidebar when the app runs, or:
```bash
export GROQ_API_KEY=gsk_your_key_here
```

### 4. Launch dashboard
```bash
streamlit run app/dashboard.py
```
Open **http://localhost:8501**

---

##  Groq Chatbot — Free Setup

| Step | Action |
|---|---|
| 1 | Sign up at **console.groq.com** |
| 2 | Create API key (starts with `gsk_`) |
| 3 | Paste in sidebar **or** `export GROQ_API_KEY=gsk_...` |
| 4 | Ask anything — "What's cheaper than Careem?" / "I want fluoride-free toothpaste cheaper" |

**Model:** `llama-3.3-70b-versatile` · **Rate limit:** ~30 req/min free · **Context:** Your full spending history + financial profile

---

##  Alert System — 6 Types

| Type | What It Detects | Example |
|---|---|---|
| 🔥 **Burn Rate** | Projects month-end spend from current daily rate | "Food on track for PKR 92K — budget is 8K" |
| 📈 **Acceleration** | Spend >30% above your 3-month average | "Shopping up 68% vs your average" |
| 🛒 **Merchant Spike** | Single transaction 2.5σ above category mean | "Unusual PKR 8,400 at Daraz — 3.2× your norm" |
| 🐢 **Category Creep** | Category share grew ≥8 ppt over 3 months | "Entertainment grew from 6% → 19%" |
| 💸 **Savings Erosion** | Savings dropped 3+ consecutive months | "Savings down 3 months in a row — now 4%" |
| 📅 **Weekend Drain** | Weekend avg ≥1.8× weekday avg | "Weekend spend 2.8× higher — PKR 12K/month extra" |

Each alert includes a **specific, actionable step** — not just a number.

---

## ML Model Performance

| Model | R² Score | MAE | RMSE |
|---|---|---|---|
| **Gradient Boosting** ⭐ | **0.9648** | PKR 3,482 | PKR 5,146 |
| Random Forest | 0.9354 | PKR 4,338 | PKR 6,974 |

**15 features:** cyclical month encoding, 3-month rolling stats, lag-1/2/3, spend momentum, lag z-score, category share  
**Split:** Chronological 80/20 — no data leakage

---

## Kaggle Dataset Integration

The Kaggle Personal Finance Tracker dataset (monthly aggregates, USD, no descriptions) is **incompatible** with transaction-level analysis. Instead, we pulled all its financial health columns into our generator:

`credit_score` · `debt_to_income_ratio` · `savings_rate` · `financial_stress_level` · `income_type` · `emergency_fund_pkr` · `emergency_fund_months` · `subscription_services` · `loan_payment_pkr` · `investment_amount_pkr` · `savings_goal_met` · `financial_scenario` · `fraud_flag` · `financial_advice_score` · `cash_flow_status`

All distributions calibrated from real Kaggle data. Stress level affects category spending weights. Fraud flag generates realistic anomalous transactions.

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Core language |
| Streamlit | ≥1.28 | Dashboard |
| Plotly | ≥5.17 | Interactive charts |
| Scikit-learn | ≥1.3 | ML models |
| Pandas / NumPy | ≥2.0 / ≥1.24 | Data processing |
| Requests | ≥2.31 | Groq API calls |
| Groq API | Free tier | Llama 3.3 70B chatbot |

---

##  Future Work

| Area | Current | Next Step |
|---|---|---|
| Categorisation | Rule-based keywords | NLP classifier (TF-IDF + LR or DistilBERT) — plug-in replacement for `categorise_transaction()` |
| Bank integration | CSV upload only | 1Link / Plaid OAuth for live transaction sync |
| Multi-user | Single session | FastAPI backend + PostgreSQL + JWT auth |
| Mobile | Web only | React Native / Flutter consuming existing Python API |

---

##  Team

| Name | Role | ID |
|---|---|---|
| Kashmala Saghar | Team Lead · AI Model Dev · PM | 23K-3034 |
| Manahil Ayaz | Data Engineer · Testing · Dashboard | 23K-3018 |
| Syeda Tayyiba Fatima | Frontend · Testing · Dashboard | 23K-0853 |


**FAST-NUCES Karachi · AI Department · 2026 · MIT License**
