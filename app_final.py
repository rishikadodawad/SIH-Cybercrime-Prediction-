
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "fraud_type_random_forest_model.pkl"
BANKING_FILE = BASE_DIR / "cybercrime_banking_synthetic_100_records.json"
MULE_FILE = BASE_DIR / "mule_network_fund_flow_100_records.json"
ATM_FILE = BASE_DIR / "atm_location_dataset.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


st.set_page_config(
    page_title="Cybercrime Prediction & Intelligence",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Cybercrime Prediction & Intelligence System")
st.caption(
    "ML-based fraud-type classification with rule-based mule-account "
    "and cash-out intelligence."
)

# -------------------- LOAD FILES --------------------
missing_files = [
    str(p.name)
    for p in [MODEL_FILE, BANKING_FILE, MULE_FILE, ATM_FILE]
    if not p.exists()
]

if missing_files:
    st.error("Missing required project files:")
    for f in missing_files:
        st.write(f"- {f}")
    st.stop()

try:
    model = joblib.load(MODEL_FILE)
    bank_df = load_json(BANKING_FILE)
    mule_df = load_json(MULE_FILE)
    atm_df = load_json(ATM_FILE)
except Exception as e:
    st.error(f"Could not load the project files: {e}")
    st.stop()


# -------------------- HELPERS --------------------
def first_existing(df, names, default=None):
    for name in names:
        if name in df.columns:
            return name
    return default


def minmax_series(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    if len(s) == 0 or s.max() == s.min():
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def risk_label(score):
    if score >= 70:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


def case_risk_score(amount, count, delay, previous, cash_withdrawal):
    score = 0

    if amount >= 50000:
        score += 25
    elif amount >= 20000:
        score += 15
    elif amount >= 10000:
        score += 8

    if count >= 10:
        score += 20
    elif count >= 5:
        score += 12
    elif count >= 3:
        score += 6

    if delay >= 48:
        score += 20
    elif delay >= 24:
        score += 12
    elif delay >= 6:
        score += 6

    if previous >= 3:
        score += 20
    elif previous >= 1:
        score += 10

    if cash_withdrawal:
        score += 15

    return min(score, 100)


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"yes", "true", "1", "y"}


# -------------------- SIDEBAR --------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "🔎 Case Analysis",
        "🏦 Mule Intelligence",
        "📍 Cash-out Intelligence",
    ],
)

# -------------------- DASHBOARD --------------------
if page == "🏠 Dashboard":
    st.header("Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Banking Records", len(bank_df))
    c2.metric("Mule Records", len(mule_df))
    c3.metric("ATM Records", len(atm_df))

    fraud_col = first_existing(bank_df, ["fraud_type", "Fraud Type"])
    unique_fraud = bank_df[fraud_col].nunique() if fraud_col else 0
    c4.metric("Fraud Types", unique_fraud)

    st.subheader("System Architecture")
    st.markdown(
        """
        **Transaction / Case Input**
        ↓  
        **Random Forest ML → Fraud-Type Prediction**
        ↓  
        **Mule Account Risk Analysis**
        ↓  
        **Cash-out Location Intelligence**
        ↓  
        **Combined Risk Score**
        ↓  
        **Actionable Cybercrime Alert**
        """
    )

    st.info(
        "The ML component predicts the likely fraud type. "
        "Mule-account and cash-out intelligence are currently score/rule-based."
    )

    if fraud_col:
        st.subheader("Fraud Type Distribution")
        st.bar_chart(bank_df[fraud_col].value_counts())


# -------------------- CASE ANALYSIS --------------------
elif page == "🔎 Case Analysis":
    st.header("🔎 Case Analysis")
    st.write("Enter transaction details to obtain an ML fraud-type prediction.")

    def options_or_default(df, column, default):
        if column in df.columns:
            values = df[column].dropna().astype(str).unique().tolist()
            if values:
                return sorted(values)
        return [default]

    state_options = options_or_default(bank_df, "State", "Karnataka")
    city_options = options_or_default(bank_df, "City", "Bengaluru")
    area_options = options_or_default(bank_df, "Area", "Electronic City")
    bank_options = options_or_default(bank_df, "bank_type", "Private Bank")
    channel_options = options_or_default(
        bank_df, "payment_channel", "UPI"
    )

    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox("State", state_options)
        city = st.selectbox("City", city_options)
        area = st.selectbox("Area", area_options)
        bank_type = st.selectbox("Bank Type", bank_options)
        payment_channel = st.selectbox(
            "Payment Channel", channel_options
        )

        transaction_amount = st.number_input(
            "Transaction Amount (₹)",
            min_value=0.0,
            value=10000.0,
            step=500.0,
        )

        transaction_count = st.number_input(
            "Transaction Count",
            min_value=0,
            value=3,
            step=1,
        )

    with col2:
        complaint_delay_hours = st.number_input(
            "Complaint Delay (hours)",
            min_value=0.0,
            value=5.5,
            step=0.5,
        )

        previous_complaints = st.number_input(
            "Previous Complaints",
            min_value=0,
            value=0,
            step=1,
        )

        cash_withdrawal = st.selectbox(
            "Cash Withdrawal",
            ["No", "Yes"],
        )

        transaction_hour = st.slider(
            "Transaction Hour",
            min_value=0,
            max_value=23,
            value=14,
        )

        day_of_week = st.selectbox(
            "Day of Week",
            [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"
            ],
        )

    if st.button("🚨 Analyse Case", type="primary"):
        try:
            input_df = pd.DataFrame([{
                "State": state,
                "City": city,
                "Area": area,
                "bank_type": bank_type,
                "transaction_amount": transaction_amount,
                "transaction_count": transaction_count,
                "payment_channel": payment_channel,
                "complaint_delay_hours": complaint_delay_hours,
                # The trained model expects these numeric features.
                "cash_withdrawal": 1 if cash_withdrawal == "Yes" else 0,
                "previous_complaints": previous_complaints,
                "hour": transaction_hour,
                "day_of_week": {
                    "Monday": 0,
                    "Tuesday": 1,
                    "Wednesday": 2,
                    "Thursday": 3,
                    "Friday": 4,
                    "Saturday": 5,
                    "Sunday": 6,
                }[day_of_week],
                "month": 1,
            }])

            prediction = model.predict(input_df)[0]

            probability = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                probability = float(np.max(probs)) * 100

            case_score = case_risk_score(
                transaction_amount,
                transaction_count,
                complaint_delay_hours,
                previous_complaints,
                normalize_bool(cash_withdrawal),
            )

            st.subheader("ML Prediction")
            st.success(f"Predicted Fraud Type: **{prediction}**")

            if probability is not None:
                st.metric(
                    "Model Confidence",
                    f"{probability:.1f}%"
                )

            st.subheader("Case Risk")
            risk = risk_label(case_score)

            if risk == "HIGH":
                st.error(f"🔴 HIGH RISK — {case_score}/100")
            elif risk == "MEDIUM":
                st.warning(f"🟠 MEDIUM RISK — {case_score}/100")
            else:
                st.success(f"🟢 LOW RISK — {case_score}/100")

            if risk == "HIGH":
                st.warning(
                    "Recommended action: Prioritise this case for investigation "
                    "and correlate it with mule and cash-out intelligence."
                )
            elif risk == "MEDIUM":
                st.info(
                    "Recommended action: Perform additional transaction and "
                    "account-level verification."
                )
            else:
                st.info(
                    "Recommended action: Continue monitoring and correlate with "
                    "other available evidence."
                )

        except Exception as e:
            st.error(
                "Prediction failed. Make sure the uploaded model was trained "
                "with the same feature structure used by this app."
            )
            st.exception(e)


# -------------------- MULE INTELLIGENCE --------------------
elif page == "🏦 Mule Intelligence":
    st.header("🏦 Mule Account Intelligence")

    to_col = first_existing(
        mule_df,
        ["to_account", "to account", "receiver_account", "receiver"]
    )
    from_col = first_existing(
        mule_df,
        ["from_account", "from account", "sender_account", "sender"]
    )
    amount_col = first_existing(
        mule_df,
        ["transaction_amount", "amount", "fund_amount"]
    )
    city_col = first_existing(
        mule_df,
        ["city", "City"]
    )

    if not to_col or not amount_col:
        st.error(
            "The mule dataset does not contain the expected account/amount "
            "columns. Check the column names in the JSON file."
        )
        st.stop()

    work = mule_df.copy()
    work[amount_col] = pd.to_numeric(
        work[amount_col], errors="coerce"
    ).fillna(0)

    grouped = work.groupby(to_col).agg(
        incoming_transactions=(amount_col, "count"),
        total_incoming_amount=(amount_col, "sum"),
    )

    if from_col:
        grouped["unique_senders"] = work.groupby(to_col)[from_col].nunique()
    else:
        grouped["unique_senders"] = 0

    if city_col:
        grouped["city_diversity"] = work.groupby(to_col)[city_col].nunique()
    else:
        grouped["city_diversity"] = 0

    grouped["transaction_score"] = minmax_series(
        grouped["incoming_transactions"]
    )
    grouped["amount_score"] = minmax_series(
        grouped["total_incoming_amount"]
    )
    grouped["sender_score"] = minmax_series(
        grouped["unique_senders"]
    )
    grouped["city_score"] = minmax_series(
        grouped["city_diversity"]
    )

    grouped["mule_risk_score"] = (
        0.25 * grouped["transaction_score"]
        + 0.25 * grouped["amount_score"]
        + 0.25 * grouped["sender_score"]
        + 0.25 * grouped["city_score"]
    ) * 100

    grouped["risk"] = grouped["mule_risk_score"].apply(risk_label)
    grouped = grouped.sort_values(
        "mule_risk_score",
        ascending=False
    )

    st.dataframe(
        grouped[
            [
                "incoming_transactions",
                "total_incoming_amount",
                "unique_senders",
                "city_diversity",
                "mule_risk_score",
                "risk",
            ]
        ],
        use_container_width=True,
    )

    st.caption(
        "Mule risk is a rule/score-based intelligence layer, not a separate ML model."
    )


# -------------------- CASH-OUT INTELLIGENCE --------------------
elif page == "📍 Cash-out Intelligence":
    st.header("📍 Cash-out Location Intelligence")

    withdrawal_col = first_existing(
        bank_df,
        ["cash_withdrawal", "Cash Withdrawal"]
    )
    withdrawal_location_col = first_existing(
        bank_df,
        ["withdrawal_location", "Withdrawal Location"]
    )
    withdrawal_area_col = first_existing(
        bank_df,
        ["withdrawal_area", "Withdrawal Area"]
    )

    if not withdrawal_col:
        st.error("Cash-withdrawal column was not found.")
        st.stop()

    history = bank_df[
        bank_df[withdrawal_col].apply(normalize_bool)
    ].copy()

    st.metric("Historical Cash-Withdrawal Records", len(history))

    if withdrawal_location_col and not history.empty:
        location_counts = (
            history[withdrawal_location_col]
            .astype(str)
            .value_counts()
            .rename("withdrawal_count")
            .reset_index()
        )
        location_counts.columns = [
            "withdrawal_location",
            "withdrawal_count",
        ]

        location_counts["historical_score"] = minmax_series(
            location_counts["withdrawal_count"]
        ) * 100

        st.subheader("Withdrawal Locations")
        st.dataframe(
            location_counts,
            use_container_width=True,
        )

    st.info(
        "Cash-out intelligence combines historical withdrawal evidence "
        "with available geographic/mule evidence. It is currently rule-based."
    )
