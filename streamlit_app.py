# streamlit_hdfc_funnel_app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="HDFC Sky – Funnel Lift Analyzer", layout="wide")

st.title("HDFC Sky — App Funnel Lift & Lead Quality Analyzer")
st.markdown("""
This app analyses **app funnel performance** (Installs → KYC → Trade → Esign) and computes
**conversion rates & percentage lifts** between **Pre** and **Campaign** periods.

Key ideas:
- Ignore **revenue** and INR fields.
- Treat **events with `s2s` in the name as trade-related** (e.g., `any_trade_s2s`, `equity_trade_s2s`).
- Focus on **funnel quality**: KYC, trades, e-sign.

You’ll get:
- A **conversion summary** for Pre vs Campaign.
- A **percentage lift table** for:
  - Install → KYC
  - KYC → Trade
  - Install → Esign
""")

# ----------------------------
# Helper functions
# ----------------------------

def parse_dates(df, date_col="Date"):
    """Parse Date column robustly."""
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in data.")
    try:
        df[date_col] = pd.to_datetime(df[date_col], format="%d/%m/%Y", dayfirst=True, errors="coerce")
    except Exception:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df


def safe_to_numeric(series):
    """Coerce a pandas Series to numeric, stripping commas, spaces, % and ₹."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[,\s%₹]", "", regex=True).str.strip(),
        errors="coerce"
    )


def compute_funnel_metrics(df, installs_col, kyc_col, esign_col, trade_user_col, trade_user_fallback_cols):
    """
    Compute aggregate funnel metrics for a given dataframe slice.
    Returns dict with installs, kyc_users, esign_users, trade_users and conversions.
    """
    out = {}

    installs = df[installs_col].sum() if installs_col in df.columns else 0
    kyc_users = df[kyc_col].sum() if kyc_col in df.columns else 0
    esign_users = df[esign_col].sum() if esign_col in df.columns else 0

    # Trade users: prefer any_trade_s2s (Unique users), else sum across *_trade_s2s (Unique users)
    if trade_user_col is not None and trade_user_col in df.columns:
        trade_users = df[trade_user_col].sum()
    else:
        trade_users = 0
        if trade_user_fallback_cols:
            trade_users = df[trade_user_fallback_cols].sum(axis=1).sum()

    out["installs"] = installs
    out["kyc_users"] = kyc_users
    out["esign_users"] = esign_users
    out["trade_users"] = trade_users

    # Conversion rates (as proportions)
    out["cr_install_to_kyc"] = (kyc_users / installs) if installs > 0 else np.nan
    out["cr_kyc_to_trade"] = (trade_users / kyc_users) if kyc_users > 0 else np.nan
    out["cr_install_to_esign"] = (esign_users / installs) if installs > 0 else np.nan

    return out


def compute_lift(pre_val, camp_val):
    """Compute percentage lift (campaign vs pre)."""
    if pre_val is None or np.isnan(pre_val) or pre_val == 0:
        return np.nan
    return (camp_val / pre_val - 1) * 100.0


# ----------------------------
# Sidebar: Data input
# ----------------------------

st.sidebar.header("Data Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload app funnel CSV (e.g. HDFC_SKY_Android_Organic_Paid_13_11_25.csv)",
    type=["csv"]
)

default_path = "/mnt/data/HDFC_SKY_Android_Organic_Paid_13_11_25.csv"

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
else:
    try:
        df_raw = pd.read_csv(default_path)
        st.sidebar.success(f"Loaded default file from: {default_path}")
    except Exception as e:
        st.sidebar.error("Could not load default file. Please upload a CSV.")
        st.stop()

# ----------------------------
# Basic cleaning & column detection
# ----------------------------

# 1. Parse dates
df = parse_dates(df_raw.copy(), date_col="Date")

# 2. Filter out obviously revenue-related columns (we won't use them)
revenue_like = [c for c in df.columns if "sales in inr" in c.lower() or "total revenue" in c.lower() or "roi" in c.lower() or "arpu" in c.lower()]
# We won't drop them (in case user wants to explore later) but we won't touch them.

# 3. Key columns
installs_col = "Installs"
kyc_unique_col = None
esign_unique_col = None
trade_any_unique_col = None
trade_unique_fallback_cols = []

# Detect KYC
kyc_candidates = [c for c in df.columns if "verify_kyc" in c.lower() and "unique users" in c.lower()]
if kyc_candidates:
    kyc_unique_col = kyc_candidates[0]

# Detect Esign
esign_candidates = [c for c in df.columns if "esign" in c.lower() and "unique users" in c.lower()]
if esign_candidates:
    esign_unique_col = esign_candidates[0]

# Detect trade-related s2s events
s2s_unique_cols = [c for c in df.columns if "s2s" in c.lower() and "unique users" in c.lower()]

for c in s2s_unique_cols:
    if "any_trade_s2s" in c.lower():
        trade_any_unique_col = c

if trade_any_unique_col is None:
    # Fallback: any *_trade_s2s (Unique users)
    trade_unique_fallback_cols = [
        c for c in s2s_unique_cols
        if any(x in c.lower() for x in ["equity_trade_s2s", "fno_trade_s2s", "etf_trade_s2s", "mtf_trade_s2s"])
    ]

# Sanity messages
if installs_col not in df.columns:
    st.error(f"Cannot find '{installs_col}' column. Please ensure the file has this column.")
    st.stop()

if kyc_unique_col is None:
    st.warning("Could not find a 'verify_kyc (Unique users)' style column. KYC metrics will be NaN.")
if esign_unique_col is None:
    st.warning("Could not find an 'esign (Unique users)' style column. Esign metrics will be NaN.")
if trade_any_unique_col is None and not trade_unique_fallback_cols:
    st.warning("Could not find any trade-related s2s (Unique users) columns. Trade metrics will be NaN.")

# 4. Coerce relevant numeric columns
numeric_cols_to_clean = [installs_col]
if kyc_unique_col: numeric_cols_to_clean.append(kyc_unique_col)
if esign_unique_col: numeric_cols_to_clean.append(esign_unique_col)
if trade_any_unique_col: numeric_cols_to_clean.append(trade_any_unique_col)
numeric_cols_to_clean.extend(trade_unique_fallback_cols)

numeric_cols_to_clean = list(dict.fromkeys(numeric_cols_to_clean))  # de-duplicate

for col in numeric_cols_to_clean:
    if col in df.columns:
        df[col] = safe_to_numeric(df[col]).fillna(0)

# ----------------------------
# Sidebar: Filters & Campaign window
# ----------------------------

st.sidebar.header("Filters & Campaign Window")

# Media Source filter (e.g. Paid / Organic)
media_col = "Media Source (pid)"
if media_col in df.columns:
    media_options = ["All"] + sorted(df[media_col].dropna().unique().tolist())
    media_choice = st.sidebar.selectbox("Filter by Media Source (pid)", options=media_options, index=0)
    if media_choice != "All":
        df = df[df[media_col] == media_choice]
else:
    st.sidebar.info("No 'Media Source (pid)' column found. Using all rows.")

# Campaign date range
min_date = df["Date"].min()
max_date = df["Date"].max()

st.sidebar.markdown(f"**Data date range:** {min_date.date()} → {max_date.date()}")

default_start = max_date - pd.Timedelta(days=30)
if default_start < min_date:
    default_start = min_date

campaign_start = st.sidebar.date_input("Campaign start date", value=default_start, min_value=min_date.date(), max_value=max_date.date())
campaign_end = st.sidebar.date_input("Campaign end date", value=max_date.date(), min_value=min_date.date(), max_value=max_date.date())

if campaign_start > campaign_end:
    st.sidebar.error("Campaign start date cannot be after end date.")
    st.stop()

campaign_start_dt = pd.to_datetime(campaign_start)
campaign_end_dt = pd.to_datetime(campaign_end)

# Define periods
pre_mask = df["Date"] < campaign_start_dt
camp_mask = (df["Date"] >= campaign_start_dt) & (df["Date"] <= campaign_end_dt)

df_pre = df[pre_mask]
df_camp = df[camp_mask]

if df_pre.empty:
    st.warning("No 'Pre' period data before campaign start date. Consider choosing an earlier start date.")
if df_camp.empty:
    st.warning("No data in campaign window. Adjust the date range.")
    st.stop()

st.markdown(f"""
### Periods used for lift analysis
- **Pre period:** all rows with Date < **{campaign_start_dt.date()}**
- **Campaign period:** rows with **{campaign_start_dt.date()} ≤ Date ≤ {campaign_end_dt.date()}**
""")

# ----------------------------
# Compute funnel metrics for Pre vs Campaign
# ----------------------------

pre_metrics = compute_funnel_metrics(
    df_pre,
    installs_col=installs_col,
    kyc_col=kyc_unique_col,
    esign_col=esign_unique_col,
    trade_user_col=trade_any_unique_col,
    trade_user_fallback_cols=trade_unique_fallback_cols,
)

camp_metrics = compute_funnel_metrics(
    df_camp,
    installs_col=installs_col,
    kyc_col=kyc_unique_col,
    esign_col=esign_unique_col,
    trade_user_col=trade_any_unique_col,
    trade_user_fallback_cols=trade_unique_fallback_cols,
)

# ----------------------------
# Display: raw funnel metrics
# ----------------------------

st.header("Funnel Summary — Pre vs Campaign")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pre period totals")
    st.metric("Installs", f"{pre_metrics['installs']:.0f}")
    st.metric("KYC unique users", f"{pre_metrics['kyc_users']:.0f}")
    st.metric("Trade unique users", f"{pre_metrics['trade_users']:.0f}")
    st.metric("Esign unique users", f"{pre_metrics['esign_users']:.0f}")

with col2:
    st.subheader("Campaign period totals")
    st.metric("Installs", f"{camp_metrics['installs']:.0f}")
    st.metric("KYC unique users", f"{camp_metrics['kyc_users']:.0f}")
    st.metric("Trade unique users", f"{camp_metrics['trade_users']:.0f}")
    st.metric("Esign unique users", f"{camp_metrics['esign_users']:.0f}")

# ----------------------------
# Conversion rates table
# ----------------------------

st.subheader("Conversion Rates by Step")

conv_rows = [
    {
        "Step": "Install → KYC",
        "Pre CR (%)": pre_metrics["cr_install_to_kyc"] * 100 if pre_metrics["cr_install_to_kyc"] == pre_metrics["cr_install_to_kyc"] else np.nan,
        "Campaign CR (%)": camp_metrics["cr_install_to_kyc"] * 100 if camp_metrics["cr_install_to_kyc"] == camp_metrics["cr_install_to_kyc"] else np.nan,
    },
    {
        "Step": "KYC → Trade",
        "Pre CR (%)": pre_metrics["cr_kyc_to_trade"] * 100 if pre_metrics["cr_kyc_to_trade"] == pre_metrics["cr_kyc_to_trade"] else np.nan,
        "Campaign CR (%)": camp_metrics["cr_kyc_to_trade"] * 100 if camp_metrics["cr_kyc_to_trade"] == camp_metrics["cr_kyc_to_trade"] else np.nan,
    },
    {
        "Step": "Install → Esign",
        "Pre CR (%)": pre_metrics["cr_install_to_esign"] * 100 if pre_metrics["cr_install_to_esign"] == pre_metrics["cr_install_to_esign"] else np.nan,
        "Campaign CR (%)": camp_metrics["cr_install_to_esign"] * 100 if camp_metrics["cr_install_to_esign"] == camp_metrics["cr_install_to_esign"] else np.nan,
    },
]

conv_df = pd.DataFrame(conv_rows)

# Compute lift
conv_df["Lift (%)"] = conv_df.apply(
    lambda row: compute_lift(row["Pre CR (%)"], row["Campaign CR (%)"]),
    axis=1
)

# Formatting
conv_df_display = conv_df.copy()
for col in ["Pre CR (%)", "Campaign CR (%)", "Lift (%)"]:
    conv_df_display[col] = conv_df_display[col].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "NA")

st.dataframe(conv_df_display, use_container_width=True)

st.markdown("""
**Interpretation:**
- **Pre CR (%)** – Funnel conversion before the campaign period.
- **Campaign CR (%)** – Funnel conversion during campaign.
- **Lift (%)** – Relative change: \\((Campaign / Pre − 1) × 100\\).  
  Positive lift = funnel is getting **more efficient** during the campaign (better lead quality).
""")

# ----------------------------
# Optional: trade breakdown (s2s events)
# ----------------------------

st.header("Trade-related s2s events breakdown")

if s2s_unique_cols:
    s2s_cols_to_show = s2s_unique_cols.copy()
    st.markdown("Below table shows **unique user counts** for each trade-related `s2s` event in Pre vs Campaign.")
    s2s_pre = df_pre[s2s_cols_to_show].sum().to_frame(name="Pre unique users")
    s2s_camp = df_camp[s2s_cols_to_show].sum().to_frame(name="Campaign unique users")
    s2s_combined = s2s_pre.join(s2s_camp, how="outer").fillna(0)
    st.dataframe(s2s_combined, use_container_width=True)
else:
    st.info("No s2s trade-related unique user columns found to display breakdown.")

st.markdown("---")
st.markdown("App focuses on **funnel CR% and trade-related events (s2s)**. Revenue / INR fields are intentionally ignored.")
