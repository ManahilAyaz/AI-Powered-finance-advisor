"""
utils/recommender.py — Smart Product-Aware Recommendation Engine
Two layers:
  1. Rule-based pattern analysis → SPECIFIC product/service alternatives
     based on what the user actually bought
  2. build_chatbot_context() → feeds user's full financial data to Claude chatbot
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from dataclasses import dataclass, field

# ── Product substitution database ─────────────────────────────────────────────
SUBSTITUTIONS: Dict[str, Dict] = {
    "netflix": {
        "label": "Netflix",
        "alternatives": [
            ("YouTube Premium Family", "PKR 540/mo shared", "split among 5 = PKR 108/person"),
            ("Tapmad TV", "PKR 250/mo", "local Pakistani content + sports"),
            ("Rotate + cancel model", "~50% saving", "subscribe 1 month, binge, cancel"),
        ],
        "saving_tip": "Downgrade to mobile-only plan or share a family plan to cut cost 75%."
    },
    "spotify": {
        "label": "Spotify",
        "alternatives": [
            ("YouTube Music free", "PKR 0", "same library with ads"),
            ("Spotify Student/Family", "PKR 230/mo", "50% off with family sharing"),
            ("JioSaavn", "PKR 99/mo", "large South Asian library"),
        ],
        "saving_tip": "Spotify free with data saver mode costs nothing. Family plan = PKR 58/person."
    },
    "youtube premium": {
        "label": "YouTube Premium",
        "alternatives": [
            ("NewPipe (Android)", "PKR 0", "open-source, no ads, no subscription needed"),
            ("uBlock Origin (desktop)", "PKR 0", "browser extension blocks all YouTube ads"),
        ],
        "saving_tip": "Premium is rarely worth it solo — only justified as a shared family plan."
    },
    "uber": {
        "label": "Uber",
        "alternatives": [
            ("InDrive", "~30% cheaper", "you negotiate the fare, no surge pricing"),
            ("Bykea", "PKR 30–100/trip", "motorcycle, fastest for short hops"),
            ("BRT / MetroBus pass", "PKR 1,500/mo", "unlimited rides — breaks even in ~5 Uber trips"),
        ],
        "saving_tip": "InDrive bids routinely come in 25–35% below Uber for the same route."
    },
    "careem": {
        "label": "Careem",
        "alternatives": [
            ("InDrive", "~25–35% cheaper", "bid-based, lower surge pricing"),
            ("Bykea", "PKR 30–100/trip", "motorcycle, fastest for distances under 5km"),
            ("MetroBus / BRT", "PKR 20–30/trip", "fixed route but very cheap"),
        ],
        "saving_tip": "Careem Go is significantly cheaper than Careem Go+ — always pick base tier."
    },
    "mcdonald": {
        "label": "McDonald's",
        "alternatives": [
            ("Local broast house", "40–60% cheaper", "comparable quality, no brand markup"),
            ("Student Biryani", "PKR 150–300/meal", "full meal vs PKR 800+ at McDonald's"),
            ("Meal prep at home", "PKR 50–100/meal", "batch cook on Sundays"),
        ],
        "saving_tip": "McDonald's app has daily deals — always order via app, prices are 15–20% lower."
    },
    "kfc": {
        "label": "KFC",
        "alternatives": [
            ("Local broast restaurant", "50% cheaper", "Karachi's local broast is genuinely excellent"),
            ("KFC Tuesday deal only", "PKR 299 combo", "cheapest day — plan meals around it"),
            ("Air-fryer broast at home", "PKR 80–150/meal", "replicate KFC results at home"),
        ],
        "saving_tip": "KFC Tuesday is the best value fast-food deal in Pakistan — limit visits to Tuesdays."
    },
    "pizza": {
        "label": "Pizza (Pizza Hut / Domino's)",
        "alternatives": [
            ("Broadway Pizza", "30% cheaper", "comparable quality, local brand"),
            ("Frozen pizza (Shan/National)", "PKR 350–500", "home-baked, feeds 2–3 people"),
            ("Order via app", "20–30% cheaper", "online prices always beat dine-in prices"),
        ],
        "saving_tip": "Pizza chain apps give 20–30% discounts vs counter ordering — always use the app."
    },
    "café": {
        "label": "Café / Coffee Shop",
        "alternatives": [
            ("Nescafé / Moccona at home", "PKR 15–30/cup", "vs PKR 500–800 at café"),
            ("Basic espresso machine", "PKR 5,000–15,000 one-time", "breaks even in 3–4 months"),
            ("Chaaye Khana / Paaye ki tapri", "PKR 30–80", "authentic ambiance at fraction of cost"),
        ],
        "saving_tip": "Making coffee at home 5×/week saves PKR 15,000–25,000/year."
    },
    "daraz": {
        "label": "Daraz",
        "alternatives": [
            ("OLX Pakistan", "used items 40–70% off", "electronics, furniture, clothes"),
            ("Daraz 11.11 / Big Friday", "up to 70% off", "plan big purchases around sales"),
            ("AliExpress direct", "10–30% cheaper", "same suppliers, no middleman"),
        ],
        "saving_tip": "Add to Daraz wishlist and wait for Flash Sale — items regularly drop 20–50%."
    },
    "khaadi": {
        "label": "Khaadi",
        "alternatives": [
            ("Khaadi End-of-Season Sale", "50–70% off", "twice yearly — stock up then"),
            ("Sapphire / Limelight", "15–25% cheaper", "comparable quality and aesthetic"),
            ("Wholesale garment district", "60–80% cheaper", "same fabrics, zero brand markup"),
        ],
        "saving_tip": "Khaadi outlet stores carry last-season stock at 40–60% discount year-round."
    },
    "jazz": {
        "label": "Jazz (Mobile/Internet)",
        "alternatives": [
            ("Jazz Giga Max weekly", "PKR 55/week", "14GB — better per-GB value than monthly"),
            ("Zong 4G unlimited", "PKR 500/mo", "compare speeds in your area first"),
            ("PTCL Fiber + Jazz SIM data saver", "combo approach", "fiber at home, minimal mobile data"),
        ],
        "saving_tip": "Jazz weekly bundles are consistently cheaper per GB than monthly packs."
    },
    "pharmacy": {
        "label": "Pharmacy",
        "alternatives": [
            ("Generic medicines", "40–80% cheaper", "same molecule — ask pharmacist explicitly"),
            ("Government dispensary", "PKR 0–50", "free or near-free for common medications"),
            ("Dawaai.pk online pharmacy", "10–25% cheaper", "delivery + verified generics"),
        ],
        "saving_tip": "Generic = branded in chemistry. Paracetamol (PKR 12) = Panadol (PKR 80). Always ask."
    },
}

CATEGORY_TIPS: Dict[str, List[str]] = {
    "Food & Dining": [
        "Meal prep Sundays: cooking 5 portions at once cuts per-meal cost 60–70%.",
        "FoodPanda/Cheetay Happy Hours (10pm–midnight) give 30–50% off — time orders around this.",
        "Buy staples from Imtiaz/Metro rather than neighbourhood kiryana — 20–30% cheaper.",
    ],
    "Transport": [
        "BRT/MetroBus monthly pass = PKR 1,500 unlimited — breaks even in ~5 Careem trips.",
        "Consolidate errands: one trip covering 3 tasks beats 3 separate ride-hail trips.",
        "Carpool with colleagues: splitting fuel 3 ways cuts your cost to PKR 33/trip.",
    ],
    "Entertainment": [
        "Audit all subscriptions right now — average person pays for 2–3 they forgot about.",
        "YouTube free + uBlock Origin = zero cost for most content. NewPipe on Android = no ads.",
        "Family plan sharing: split one Netflix/Spotify 4 ways = PKR 100–200/person/month.",
    ],
    "Shopping": [
        "72-hour rule: wait 3 days before any purchase over PKR 3,000. Eliminates 30–40% of impulse buys.",
        "OLX/Facebook Marketplace: 50–80% off lightly used electronics and furniture.",
        "Buy off-season: winter clothes in March, summer in October — 40–60% discounts.",
    ],
    "Utilities": [
        "LED bulbs cut electricity 70–80% per light point vs CFL/incandescent.",
        "Phantom load (standby devices) = 5–10% of your KESC bill — unplug everything not in use.",
        "KESC net metering: solar panels pay back in 3–5 years in Karachi's sunlight.",
    ],
    "Healthcare": [
        "Generic medicines are chemically identical to branded — always ask for generics.",
        "Annual check-up (PKR 3,000) prevents expensive emergency treatments.",
    ],
    "Savings": [
        "Automate savings: standing transfer on payday — before you can spend it.",
        "National Savings Certificates: 15–21% annual returns, government-backed.",
        "Meezan Bank Savings: ~13% profit rate, Shariah-compliant, easy access.",
    ],
}


@dataclass
class Recommendation:
    title: str
    detail: str
    category: str
    priority: str
    potential_saving: float
    substitutions: List[tuple] = field(default_factory=list)
    merchants_detected: List[str] = field(default_factory=list)


def _detect_merchants(descriptions: pd.Series) -> Dict[str, int]:
    counts = {}
    for key in SUBSTITUTIONS:
        n = int(descriptions.str.lower().str.contains(key, na=False).sum())
        if n > 0:
            counts[key] = n
    return dict(sorted(counts.items(), key=lambda x: -x[1]))

def generate_recommendations(
    df: pd.DataFrame,
    budgets: Dict[str, float] = None,
    monthly_income: float = 80000,
) -> List[Recommendation]:
    from utils.overspending_detector import DEFAULT_BUDGETS
    if budgets is None:
        budgets = DEFAULT_BUDGETS

    recs: List[Recommendation] = []
    total_spent = df["amount"].sum()
    months = max(df["month"].nunique(), 1)
    cat_totals = df.groupby("category")["amount"].sum()
    monthly_avg = df.groupby("month")["amount"].sum().mean()
    merchant_hits = _detect_merchants(df["description"])

    # R1: Merchant-specific substitutions — purely forward-looking
    for merchant_key, txn_count in merchant_hits.items():
        info = SUBSTITUTIONS[merchant_key]
        subset = df[df["description"].str.lower().str.contains(merchant_key, na=False)]
        cat = subset["category"].mode().iloc[0] if len(subset) else "Miscellaneous"
        avg_per_month = subset["amount"].sum() / months

        recs.append(Recommendation(
            title=f"Switch from {info['label']} to these cheaper alternatives",
            detail=info["saving_tip"],   # already forward-looking advice
            category=cat,
            priority="high" if avg_per_month > 1500 else "medium",
            potential_saving=avg_per_month * 0.40 * 12,
            substitutions=info["alternatives"],
            merchants_detected=[info["label"]],
        ))

    # R2: Category budget — forward-looking spend plan, not diagnosis
    for cat, budget in budgets.items():
        cat_monthly = df[df["category"] == cat].groupby("month")["amount"].sum()
        if len(cat_monthly) == 0:
            continue
        avg = cat_monthly.mean()
        if avg > budget and cat in CATEGORY_TIPS:
            tip = CATEGORY_TIPS[cat][len(recs) % len(CATEGORY_TIPS[cat])]
            recs.append(Recommendation(
                title=f"Bring {cat} under PKR {budget:,.0f}/month with these steps",
                detail=tip,   # tip is already actionable
                category=cat,
                priority="high",
                potential_saving=(avg - budget) * 12,
            ))

    # R3: Savings — forward-looking investment plan
    savings_total = cat_totals.get("Savings", 0)
    savings_rate = savings_total / (monthly_income * months)
    if savings_rate < 0.20:
        target = monthly_income * 0.20
        current = savings_total / months
        monthly_gap = target - current
        recs.append(Recommendation(
            title=f"Invest PKR {target:,.0f}/month to hit the 20% savings target",
            detail=(
                f"Start with PKR {min(monthly_gap, 5000):,.0f}/month via a standing transfer on payday "
                f"before discretionary spending kicks in. "
                f"NSCs pay 21% annually — PKR {target*12:,.0f}/year invested compounds to "
                f"PKR {target*12*1.21:,.0f} in 12 months."
            ),
            category="Savings",
            priority="high",
            potential_saving=monthly_gap * 12,
            substitutions=[
                ("National Savings Certificates", "21% p.a.", "safest, government-backed"),
                ("Meezan Savings Account", "~13% p.a.", "Shariah-compliant, easy access"),
                ("Meezan Mutual Fund", "12–18% p.a.", "slightly more risk, higher return"),
            ],
        ))

    # R4: Emergency fund — concrete build plan
    if savings_total < monthly_avg * 3:
        weekly_target = 1000
        weeks_to_goal = int((monthly_avg * 3 - savings_total) / (weekly_target * 4))
        recs.append(Recommendation(
            title=f"Build a 3-month emergency fund in ~{max(weeks_to_goal,1)} months",
            detail=(
                f"Transfer PKR {weekly_target:,.0f}/week automatically to a separate account. "
                f"Use JazzCash Saving Wallet — set up in 5 minutes, earns 8–10% annually, "
                f"and being separate removes the temptation to dip into it."
            ),
            category="Savings",
            priority="high",
            potential_saving=0,
            substitutions=[
                ("JazzCash Saving Wallet", "8–10% p.a.", "easiest setup, mobile-first"),
                ("EasyPaisa Account", "7–9% p.a.", "automatic round-up savings feature"),
            ],
        ))

    # R5: Subscription consolidation — forward-looking plan
    streaming_kws = ["netflix","spotify","youtube premium","amazon prime","hbo","apple tv"]
    streaming = df[df["description"].str.lower().apply(
        lambda d: any(k in d for k in streaming_kws))]
    if len(streaming) > 0:
        streaming_per_month = streaming["amount"].sum() / months
        unique_svcs = streaming["description"].str.lower().apply(
            lambda d: next((k for k in streaming_kws if k in d), "other")
        ).nunique()
        if unique_svcs >= 2:
            recs.append(Recommendation(
                title=f"Cut streaming costs by 50% with a rotate-and-cancel plan",
                detail=(
                    f"Subscribe to one service at a time for one month, binge, then cancel and move on. "
                    f"Or share a family plan — splits to PKR 100–200/person/month. "
                    f"Target: spend no more than PKR {streaming_per_month*0.5:,.0f}/month on streaming."
                ),
                category="Entertainment",
                priority="medium",
                potential_saving=streaming_per_month * 0.5 * 12,
                substitutions=[
                    ("Rotate + cancel model", "~50% saving", "1 service at a time"),
                    ("Family plan (4 people)", "~75% saving", "PKR 100–200/person/month"),
                ],
            ))

    # Fallbacks — always forward-looking
    fallbacks = [
        Recommendation(
            title="Apply the 72-hour rule to every purchase over PKR 3,000",
            detail=(
                "Add the item to a wishlist and wait 3 days before buying. "
                "This single habit eliminates 30–40% of impulse purchases with zero effort. "
                "Use Daraz wishlist or a notes app — review every Sunday."
            ),
            category="Shopping", priority="medium", potential_saving=15000,
        ),
        Recommendation(
            title="Ask your pharmacist for generics — same drug, 40–80% cheaper",
            detail=(
                "Request the generic name, not the brand. Paracetamol (PKR 12) = Panadol (PKR 80). "
                "Amoxicillin generic = PKR 50 vs PKR 250 branded. Works for 90% of common prescriptions. "
                "Dawaai.pk delivers generics with 10–25% additional discount."
            ),
            category="Healthcare", priority="low", potential_saving=8000,
            substitutions=[
                ("Generic paracetamol", "PKR 12", "identical to Panadol"),
                ("Generic amoxicillin", "PKR 50", "identical to branded antibiotics"),
                ("Dawaai.pk online", "10–25% off", "verified generics with home delivery"),
            ],
        ),
    ]
    i = 0
    while len(recs) < 3:
        recs.append(fallbacks[i % len(fallbacks)])
        i += 1

    order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: (order[r.priority], -r.potential_saving))
    return recs


def build_chatbot_context(df: pd.DataFrame, monthly_income: float = 80000,
                          profile: dict = None) -> str:
    """
    Build rich system prompt for Claude Finance Chatbot.
    Includes full spending history + Kaggle-enriched financial health profile.
    """
    cat_summary   = df.groupby("category")["amount"].agg(["sum", "count"])
    total_spent   = df["amount"].sum()
    months        = df["month"].nunique()
    top_merchants = df.groupby("description")["amount"].agg(["sum", "count"]).nlargest(10, "sum")
    merchant_hits = _detect_merchants(df["description"])

    lines = [
        "You are an expert personal finance advisor specialising in the Pakistani market (PKR).",
        "You have the user's COMPLETE spending history AND financial health profile. Use both.",
        "Give SPECIFIC, named alternatives — never generic advice.",
        "Always name real Pakistani products, brands, stores, apps with real PKR prices.",
        "If asked about a product substitute (e.g. 'cheaper toothpaste'), name the exact brand, price, where to buy in Pakistan.",
        "Be conversational and direct. Use bullet points for alternatives. Keep answers under 180 words.",
        "",
        "=== FINANCIAL HEALTH PROFILE ===",
        f"Monthly income     : PKR {monthly_income:,.0f}",
    ]

    if profile:
        cs   = profile.get("credit_score", 680)
        dti  = profile.get("debt_to_income_ratio", 0.35)
        sr   = profile.get("savings_rate", 0.10)
        sl   = profile.get("financial_stress_level", "Medium")
        cf   = profile.get("cash_flow_status", "Neutral")
        efm  = profile.get("emergency_fund_months", 1.0)
        it   = profile.get("income_type", "Salary")
        subs = profile.get("subscription_services", 3)
        loan = profile.get("loan_payment_pkr", 0)
        inv  = profile.get("investment_amount_pkr", 0)
        sgm  = profile.get("savings_goal_met", 0)
        scen = profile.get("financial_scenario", "normal")
        adv  = profile.get("financial_advice_score", 50)
        lines += [
            f"Income type        : {it}",
            f"Credit score       : {cs} ({'Good' if cs>=720 else 'Fair' if cs>=650 else 'Poor'})",
            f"Debt-to-income     : {dti:.0%} ({'OK' if dti<0.30 else 'High' if dti>0.45 else 'Moderate'})",
            f"Savings rate       : {sr:.0%} ({'On track' if sr>=0.20 else 'Below target'})",
            f"Financial stress   : {sl}",
            f"Cash flow          : {cf}",
            f"Emergency fund     : {efm:.1f} months ({'Sufficient' if efm>=3 else 'Insufficient'})",
            f"Active subscriptions: {subs}",
            f"Loan payment/mo    : PKR {loan:,.0f}",
            f"Investment/mo      : PKR {inv:,.0f}",
            f"Savings goal met   : {'Yes' if sgm else 'No'}",
            f"Economic scenario  : {scen}",
            f"Advice score       : {adv}/100",
        ]
        if sl == "High":
            lines.append("⚠️ USER IS UNDER HIGH FINANCIAL STRESS — be empathetic and focus on quick wins.")
        if dti > 0.45:
            lines.append("⚠️ HIGH DEBT-TO-INCOME — prioritise debt reduction advice.")
        if efm < 1:
            lines.append("⚠️ NO EMERGENCY FUND — flag this as urgent.")

    lines += [
        "",
        "=== SPENDING BY CATEGORY ===",
    ]
    for cat, row in cat_summary.iterrows():
        pct = row["sum"] / total_spent * 100 if total_spent else 0
        lines.append(f"  {cat}: PKR {row['sum']:,.0f} ({pct:.1f}%) — {int(row['count'])} txns")

    lines += ["", "=== TOP 10 MERCHANTS ==="]
    for desc, row in top_merchants.iterrows():
        lines.append(f"  {desc}: PKR {row['sum']:,.0f} ({int(row['count'])} visits)")

    if merchant_hits:
        lines += ["", "=== MERCHANTS WITH CHEAPER ALTERNATIVES ==="]
        for m, count in list(merchant_hits.items())[:8]:
            info = SUBSTITUTIONS[m]
            lines.append(f"  {info['label']}: {count} transactions")

    return "\n".join(lines)
