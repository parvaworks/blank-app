import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -------------------------------------------------
# Page config
# -------------------------------------------------
# Add logo (top-left)
logo_path = "/mnt/data/hdfc_sky_logo.png"  # replace with your actual file name

st.markdown(
    """
    <style>
        .logo-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .logo-img {
            height: 48px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="logo-container">
        <img src="{logo_path}" class="logo-img">
        <h1>HDFC Sky — App Funnel Lift & Lead Quality Analyzer (v3)</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Remove the original title line
# st.title("HDFC Sky — App Funnel Lift & Lead Quality Analyzer (v3)")

st.markdown("""
This app analyses **app funnel performance** (Installs → KYC → Trade → Esign) and computes:

- **Conversion rates & % lift** for:
  - Install → KYC  
  - KYC → Trade  
  - Install → Esign  
- **Time-normalised** comparisons (equal-length pre vs campaign, per-day metrics).
- **Daily CR trends**, **Paid vs Organic**, **media source leaderboard**, **trade-related s2s breakdown**, and a
- **Custom Funnel Builder** where you can define your own steps and see lift (including weekly tables).

**Revenue / INR fields are intentionally ignored.**
""")

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def parse_dates(df, date_col="Date"):
    """
    Robust date parser for your case:
    - Some values are stored as actual datetimes, some as text.
    - All are logically month/day/year (m/d/yyyy, mm/dd/yyyy, etc.).

    Strategy:
    1. Convert EVERYTHING to string (ignoring existing dtype).
    2. Strip spaces.
    3. Parse with dayfirst=False (US-style month-first).
    4. Fallback to a second generic parse for any weird edge cases.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in data.")

    s = df[date_col].astype(str).str.strip()

    # Main parse: month/day/year style
    dt = pd.to_datetime(s, dayfirst=False, errors="coerce")

    # Fallback: let pandas try a generic parse where main failed
    dt_fallback = pd.to_datetime(s, errors="coerce")
    dt = dt.where(~dt.isna(), dt_fallback)

    df[date_col] = dt
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df


def safe_to_numeric(series):
    """Coerce a Series to numeric, stripping commas, spaces, % and ₹."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[,\s%₹]", "", regex=True).str.strip(),
        errors="coerce"
    )


def derive_campaign_groups(df, col="Campaign (c)"):
    name_series = df[col].fillna("").astype(str)

    def campaign_channel(name: str) -> str:
        u = name.upper()
        u_stripped = u.strip()

        if u_stripped in ("", "NONE"):
            return "Organic/None"

        if u.startswith("MB_"):
            if "_PUSH_" in u:
                return "MB Push"
            if "_APPBOX_" in u:
                return "MB Appbox"
            if "_ALERTBOX_" in u:
                return "MB Alertbox"
            return "MB Other"

        if u.startswith("NB_"):
            if "_BB_" in u:
                return "NB Banner/BB"
            if "_CP_" in u:
                return "NB Campaign Page"
            return "NB Other"

        if u.startswith("SMS_"):
            return "SMS"

        if u.startswith("EMAIL_"):
            return "Email"

        if u.startswith("RCS_"):
            return "RCS"

        if u.startswith("WEBSITE_") or "LANDING_PAGE" in u or "WEBPUSH" in u:
            return "Website / Webpush"

        if u.startswith("UAC_"):
            return "UAC (App Campaign)"

        if "WHATSAPP" in u:
            return "WhatsApp"

        if "OTT" in u or "JIOCINEMA" in u:
            return "OTT"

        if "TRADINGVIEW" in u:
            return "TradingView"

        if "OLAIPO" in u:
            return "OLAIPO"

        if "QR" in u:
            return "QR / Offline"

        if "GRAFFITI" in u or "PILLER" in u:
            return "OOH / Branch"

        if "HDFC SKY" in u or u_stripped in ("HSLSKY", "HDFCSKY", "ADS_SKY", "ADS SKY"):
            return "Sky Brand / Generic"

        # default catch-all
        return "Other"

    personas = [
        "EARLY_JOBBER", "MIDLEVEL", "SAL", "SEASONED", "SELF",
        "WOMAN", "TRADING_OUTSIDE", "REST", "LOANTAKER",
        "SENIOR_CITIZEN", "SENIOR"
    ]

    def campaign_persona(name: str) -> str:
        u = name.upper()
        for p in personas:
            if p in u:
                return p.title().replace("_", " ")
        return "Generic / None"

    def campaign_product(name: str) -> str:
        u = name.upper()
        if "IPO" in u:
            return "IPO"
        if "ETF" in u:
            return "ETF"
        if "FNO" in u or "FN0" in u:
            return "F&O"
        if "DEMAT" in u:
            return "Demat"
        if "MTF" in u:
            return "MTF"
        if "MF" in u and "EMAIL" in u:
            return "Mutual Funds"
        return "Generic / Not product-specific"

    df = df.copy()
    df["campaign_channel"] = name_series.apply(campaign_channel)
    df["campaign_persona"] = name_series.apply(campaign_persona)
    df["campaign_product"] = name_series.apply(campaign_product)
    return df


def compute_funnel_metrics(df, installs_col, kyc_col, esign_col,
                           trade_user_col, trade_user_fallback_cols,
                           days_in_period):
    """
    Compute aggregate funnel metrics for a given dataframe slice.
    Returns dict with totals, per-day metrics, and conversion rates.
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

    out["days"] = days_in_period
    out["installs"] = installs
    out["kyc_users"] = kyc_users
    out["esign_users"] = esign_users
    out["trade_users"] = trade_users

    # Per-day metrics
    if days_in_period > 0:
        out["installs_per_day"] = installs / days_in_period
        out["kyc_per_day"] = kyc_users / days_in_period
        out["esign_per_day"] = esign_users / days_in_period
        out["trade_per_day"] = trade_users / days_in_period
    else:
        out["installs_per_day"] = np.nan
        out["kyc_per_day"] = np.nan
        out["esign_per_day"] = np.nan
        out["trade_per_day"] = np.nan

    # Conversion rates (proportions)
    out["cr_install_to_kyc"] = (kyc_users / installs) if installs > 0 else np.nan
    out["cr_kyc_to_trade"] = (trade_users / kyc_users) if kyc_users > 0 else np.nan
    out["cr_install_to_esign"] = (esign_users / installs) if installs > 0 else np.nan

    return out


def compute_lift_relative(pre_val, camp_val):
    """Compute percentage lift (campaign vs pre) in percent units."""
    if pre_val is None or np.isnan(pre_val) or pre_val == 0:
        return np.nan
    return (camp_val / pre_val - 1.0) * 100.0


def format_pct(x):
    return f"{x:.2f}%" if pd.notna(x) else "NA"


def weekly_aggregate(df, step_cols):
    """Aggregate to weekly (Monday-start) for selected step columns."""
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["WeekStart"] = df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="d")
    agg = df.groupby("WeekStart")[step_cols].sum().reset_index()
    return agg


def build_pairwise_lift_table(steps, pre_totals, camp_totals):
    """
    Build table: each row = step_i → step_{i+1}, with Pre CR, Campaign CR, Lift.
    pre_totals & camp_totals are dicts {step: total_value}.
    """
    rows = []
    for i in range(len(steps) - 1):
        a = steps[i]
        b = steps[i + 1]
        pre_a = pre_totals.get(a, 0)
        pre_b = pre_totals.get(b, 0)
        camp_a = camp_totals.get(a, 0)
        camp_b = camp_totals.get(b, 0)

        pre_cr = (pre_b / pre_a * 100) if pre_a > 0 else np.nan
        camp_cr = (camp_b / camp_a * 100) if camp_a > 0 else np.nan
        lift = compute_lift_relative(pre_cr, camp_cr)

        rows.append({
            "Step": f"{a} → {b}",
            "Pre CR (%)": pre_cr,
            "Campaign CR (%)": camp_cr,
            "Lift (%)": lift
        })
    return pd.DataFrame(rows)


def build_base_to_step_lift_table(steps, pre_totals, camp_totals):
    """
    Build table: each row = base_step → step_k, with Pre CR, Campaign CR, Lift.
    Base step = steps[0].
    """
    if not steps:
        return pd.DataFrame()
    base = steps[0]
    rows = []
    pre_base = pre_totals.get(base, 0)
    camp_base = camp_totals.get(base, 0)

    for step in steps[1:]:
        pre_step = pre_totals.get(step, 0)
        camp_step = camp_totals.get(step, 0)

        pre_cr = (pre_step / pre_base * 100) if pre_base > 0 else np.nan
        camp_cr = (camp_step / camp_base * 100) if camp_base > 0 else np.nan
        lift = compute_lift_relative(pre_cr, camp_cr)

        rows.append({
            "Base → Step": f"{base} → {step}",
            "Pre CR (%)": pre_cr,
            "Campaign CR (%)": camp_cr,
            "Lift (%)": lift
        })
    return pd.DataFrame(rows)


def add_cr_columns_weekly(df_week, steps):
    """
    Given a weekly aggregated DataFrame with step count columns,
    add CR columns for consecutive steps.
    """
    df_week = df_week.copy()
    for i in range(len(steps) - 1):
        a = steps[i]
        b = steps[i + 1]
        cr_col = f"CR {a} → {b} (%)"
        if a in df_week.columns and b in df_week.columns:
            denom = df_week[a].replace(0, np.nan)
            df_week[cr_col] = (df_week[b] / denom) * 100
        else:
            df_week[cr_col] = np.nan
    return df_week


# -------------------------------------------------
# Sidebar: Data input
# -------------------------------------------------
st.sidebar.header("Data Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload app funnel CSV",
    type=["csv"]
)

default_path = "/mnt/data/HDFC_SKY_Android_Organic_Paid_13_11_25.csv"

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
else:
    try:
        df_raw = pd.read_csv(default_path)
        st.sidebar.success(f"Loaded default file from: {default_path}")
    except Exception:
        st.sidebar.error("Could not load default file. Please upload a CSV.")
        st.stop()

# -------------------------------------------------
# Basic cleaning & column detection
# -------------------------------------------------
df_raw["Date"] = df_raw["Date"].astype(str)  # force text for safety
df = parse_dates(df_raw.copy(), date_col="Date")
df = derive_campaign_groups(df, col="Campaign (c)")

# Detect key columns
installs_col = "Installs"

kyc_unique_col = None
esign_unique_col = None
trade_any_unique_col = None
trade_unique_fallback_cols = []
phone_verify_col = None  # NEW

kyc_candidates = [c for c in df.columns if "verify_kyc" in c.lower() and "unique users" in c.lower()]
if kyc_candidates:
    kyc_unique_col = kyc_candidates[0]

esign_candidates = [c for c in df.columns if "esign" in c.lower() and "unique users" in c.lower()]
if esign_candidates:
    esign_unique_col = esign_candidates[0]

# detect confirm_mobileotp (Install → Phone verify) unique users
phone_candidates = [
    c for c in df.columns
    if "confirm_mobileotp" in c.lower()
]
if phone_candidates:
    phone_verify_col = phone_candidates[0]
else:
    phone_verify_col = None

s2s_unique_cols = [c for c in df.columns if "s2s" in c.lower() and "unique users" in c.lower()]

for c in s2s_unique_cols:
    if "any_trade_s2s" in c.lower():
        trade_any_unique_col = c

if trade_any_unique_col is None:
    trade_unique_fallback_cols = [
        c for c in s2s_unique_cols
        if any(x in c.lower() for x in ["equity_trade_s2s", "fno_trade_s2s", "etf_trade_s2s", "mtf_trade_s2s"])
    ]

# Sanity checks
if installs_col not in df.columns:
    st.error(f"Cannot find '{installs_col}' column. Please ensure the file has this column.")
    st.stop()

if kyc_unique_col is None:
    st.warning("Could not find a 'verify_kyc (Unique users)' style column. KYC metrics will be NaN.")
if esign_unique_col is None:
    st.warning("Could not find an 'esign (Unique users)' style column. Esign metrics will be NaN.")
if phone_verify_col is None:
    st.warning("Could not find a 'verify phone (Unique users)' style column. Weekly Install→Verify Phone CR plot will be disabled.")
if trade_any_unique_col is None and not trade_unique_fallback_cols:
    st.warning("Could not find any trade-related s2s (Unique users) columns. Trade metrics will be NaN.")

# Coerce key numeric columns
numeric_cols_to_clean = [installs_col]
if kyc_unique_col: numeric_cols_to_clean.append(kyc_unique_col)
if esign_unique_col: numeric_cols_to_clean.append(esign_unique_col)
if phone_verify_col: numeric_cols_to_clean.append(phone_verify_col)
if trade_any_unique_col: numeric_cols_to_clean.append(trade_any_unique_col)
numeric_cols_to_clean.extend(trade_unique_fallback_cols)
numeric_cols_to_clean = list(dict.fromkeys(numeric_cols_to_clean))  # de-dup

for col in numeric_cols_to_clean:
    if col in df.columns:
        df[col] = safe_to_numeric(df[col]).fillna(0)

# -------------------------------------------------
# Media source detection & media_group classification
# -------------------------------------------------
media_col = None
for c in df.columns:
    cl = c.lower()
    if "media source" in cl and "pid" in cl:
        media_col = c
        break
if media_col is None:
    for c in df.columns:
        if "media source" in c.lower():
            media_col = c
            break

if media_col is not None:
    def classify_media(ms):
        s = str(ms).lower()
        if "paid" in s:
            return "Paid"
        if "organic" in s:
            return "Organic"
        return "Other"
    df["media_group"] = df[media_col].apply(classify_media)
else:
    st.sidebar.info("No media source column found — Paid vs Organic views will be limited.")

# -------------------------------------------------
# Sidebar: Filters & Campaign window with time normalisation
# -------------------------------------------------
st.sidebar.header("Filters & Campaign Window")

# Optional filter by raw media source value
if media_col is not None:
    media_options = ["All"] + sorted(df[media_col].dropna().unique().tolist())
    media_choice = st.sidebar.selectbox("Filter by Media Source (raw)", options=media_options, index=0)
    if media_choice != "All":
        df = df[df[media_col] == media_choice]
else:
    st.sidebar.info("No media source column found for raw filtering.")

min_date = df["Date"].min()
max_date = df["Date"].max()

st.sidebar.markdown(f"**Data date range:** {min_date.date()} → {max_date.date()}")

default_campaign_end = max_date.date()
default_campaign_start = (max_date - pd.Timedelta(days=49)).date()  # default 49 days to match your example
if default_campaign_start < min_date.date():
    default_campaign_start = min_date.date()

campaign_start = st.sidebar.date_input(
    "Campaign start date",
    value=default_campaign_start,
    min_value=min_date.date(),
    max_value=max_date.date()
)
campaign_end = st.sidebar.date_input(
    "Campaign end date",
    value=default_campaign_end,
    min_value=min_date.date(),
    max_value=max_date.date()
)

if campaign_start > campaign_end:
    st.sidebar.error("Campaign start date cannot be after end date.")
    st.stop()

campaign_start_dt = pd.to_datetime(campaign_start)
campaign_end_dt = pd.to_datetime(campaign_end)

equal_length_pre = st.sidebar.checkbox("Use equal-length pre period", value=True)

# -------------------------------------------------
# Optional: group-by mode for custom funnel later
# -------------------------------------------------
group_by = st.sidebar.selectbox(
    "Group funnel by",
    ["None", "campaign_channel", "campaign_persona", "campaign_product"],
    format_func=lambda x: {
        "None": "No grouping",
        "campaign_channel": "Campaign channel",
        "campaign_persona": "Campaign persona",
        "campaign_product": "Campaign product"
    }[x]
)

# Define campaign mask & slice
camp_mask = (df["Date"] >= campaign_start_dt) & (df["Date"] <= campaign_end_dt)
df_camp = df[camp_mask]

if df_camp.empty:
    st.warning("No data in campaign window. Adjust the date range.")
    st.stop()

campaign_days = (campaign_end_dt - campaign_start_dt).days + 1

# Define pre-period mask & slice
if equal_length_pre:
    pre_end_dt = campaign_start_dt - pd.Timedelta(days=1)
    pre_start_dt = pre_end_dt - pd.Timedelta(days=campaign_days - 1)
    if pre_start_dt < min_date:
        pre_start_dt = min_date
        pre_days = (pre_end_dt - pre_start_dt).days + 1
        st.warning(
            f"Not enough data before campaign for full equal-length pre period. "
            f"Using truncated pre: {pre_start_dt.date()} → {pre_end_dt.date()} ({pre_days} days)."
        )
    else:
        pre_days = campaign_days
    pre_mask = (df["Date"] >= pre_start_dt) & (df["Date"] <= pre_end_dt)
else:
    pre_mask = df["Date"] < campaign_start_dt
    pre_days = (campaign_start_dt - df[pre_mask]["Date"].min()).days if pre_mask.any() else 0

df_pre = df[pre_mask]

if df_pre.empty:
    st.warning("No pre period data based on current settings. Consider adjusting dates or turning off equal-length mode.")

st.markdown(f"""
### Periods used for lift analysis

- **Campaign period:** {campaign_start_dt.date()} → {campaign_end_dt.date()} (**{campaign_days} days**)
- **Pre period:** 
  - {'Equal-length mode' if equal_length_pre else 'All dates before campaign'}
  - Actual pre range: {df_pre['Date'].min().date() if not df_pre.empty else 'NA'} → {df_pre['Date'].max().date() if not df_pre.empty else 'NA'}
  - Days used: **{pre_days}**
""")

# -------------------------------------------------
# Compute funnel metrics for Pre vs Campaign
# -------------------------------------------------
pre_metrics = compute_funnel_metrics(
    df_pre,
    installs_col=installs_col,
    kyc_col=kyc_unique_col,
    esign_col=esign_unique_col,
    trade_user_col=trade_any_unique_col,
    trade_user_fallback_cols=trade_unique_fallback_cols,
    days_in_period=pre_days,
)

camp_metrics = compute_funnel_metrics(
    df_camp,
    installs_col=installs_col,
    kyc_col=kyc_unique_col,
    esign_col=esign_unique_col,
    trade_user_col=trade_any_unique_col,
    trade_user_fallback_cols=trade_unique_fallback_cols,
    days_in_period=campaign_days,
)

# -------------------------------------------------
# Tabs for insights
# -------------------------------------------------
tab_summary, tab_trends, tab_paid_org, tab_media, tab_s2s, tab_custom = st.tabs(
    ["Summary & Lifts", "CR Trends", "Paid vs Organic", "Media Source Leaderboard", "Trade s2s Breakdown", "Custom Funnel & Weekly Lifts"]
)

# -------------------------------------------------
# Tab 1: Summary & Lifts (period-level)
# -------------------------------------------------
with tab_summary:
    st.header("Funnel Summary — Pre vs Campaign (Time-normalised)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Pre period totals")
        st.metric("Days", f"{pre_metrics['days']:.0f}")
        st.metric("Installs", f"{pre_metrics['installs']:.0f}")
        st.metric("KYC unique users", f"{pre_metrics['kyc_users']:.0f}")
        st.metric("Trade unique users", f"{pre_metrics['trade_users']:.0f}")
        st.metric("Esign unique users", f"{pre_metrics['esign_users']:.0f}")

    with col2:
        st.subheader("Campaign period totals")
        st.metric("Days", f"{camp_metrics['days']:.0f}")
        st.metric("Installs", f"{camp_metrics['installs']:.0f}")
        st.metric("KYC unique users", f"{camp_metrics['kyc_users']:.0f}")
        st.metric("Trade unique users", f"{camp_metrics['trade_users']:.0f}")
        st.metric("Esign unique users", f"{camp_metrics['esign_users']:.0f}")

    with col3:
        st.subheader("Per-day metrics (Pre vs Campaign)")
        st.metric("Installs/day (Pre)", f"{pre_metrics['installs_per_day']:.1f}")
        st.metric("Installs/day (Campaign)", f"{camp_metrics['installs_per_day']:.1f}")
        st.metric("KYC/day (Pre)", f"{pre_metrics['kyc_per_day']:.1f}")
        st.metric("KYC/day (Campaign)", f"{camp_metrics['kyc_per_day']:.1f}")
        st.metric("Trade/day (Pre)", f"{pre_metrics['trade_per_day']:.1f}")
        st.metric("Trade/day (Campaign)", f"{camp_metrics['trade_per_day']:.1f}")

    st.subheader("Default Conversion Rates by Step & % Lift")

    conv_rows = [
        {
            "Step": "Install → KYC",
            "Pre CR (%)": pre_metrics["cr_install_to_kyc"] * 100 if pd.notna(pre_metrics["cr_install_to_kyc"]) else np.nan,
            "Campaign CR (%)": camp_metrics["cr_install_to_kyc"] * 100 if pd.notna(camp_metrics["cr_install_to_kyc"]) else np.nan,
        },
        {
            "Step": "KYC → Trade",
            "Pre CR (%)": pre_metrics["cr_kyc_to_trade"] * 100 if pd.notna(pre_metrics["cr_kyc_to_trade"]) else np.nan,
            "Campaign CR (%)": camp_metrics["cr_kyc_to_trade"] * 100 if pd.notna(camp_metrics["cr_kyc_to_trade"]) else np.nan,
        },
        {
            "Step": "Install → Esign",
            "Pre CR (%)": pre_metrics["cr_install_to_esign"] * 100 if pd.notna(pre_metrics["cr_install_to_esign"]) else np.nan,
            "Campaign CR (%)": camp_metrics["cr_install_to_esign"] * 100 if pd.notna(camp_metrics["cr_install_to_esign"]) else np.nan,
        },
    ]

    conv_df = pd.DataFrame(conv_rows)
    conv_df["Lift (%)"] = conv_df.apply(
        lambda row: compute_lift_relative(row["Pre CR (%)"], row["Campaign CR (%)"]),
        axis=1
    )

    conv_df_display = conv_df.copy()
    conv_df_display["Pre CR (%)"] = conv_df_display["Pre CR (%)"].map(format_pct)
    conv_df_display["Campaign CR (%)"] = conv_df_display["Campaign CR (%)"].map(format_pct)
    conv_df_display["Lift (%)"] = conv_df_display["Lift (%)"].map(format_pct)

    st.dataframe(conv_df_display, use_container_width=True)

    st.markdown("""
**Reading this table:**
- **Pre CR (%)** – Funnel conversion in the pre period.
- **Campaign CR (%)** – Funnel conversion in the campaign period.
- **Lift (%)** – Relative change: \\((Campaign / Pre − 1) × 100\\).  
  - Positive lift → **funnel is more efficient** (higher-quality users).  
  - Negative lift → funnel is less efficient.
""")


# -------------------------------------------------
# Tab 2: CR Trends
# -------------------------------------------------
with tab_trends:
    st.header("Daily Conversion Rate Trends")

    daily_cols = [installs_col]
    if kyc_unique_col: daily_cols.append(kyc_unique_col)
    if esign_unique_col: daily_cols.append(esign_unique_col)
    if trade_any_unique_col:
        daily_cols.append(trade_any_unique_col)
    daily_cols = list(dict.fromkeys([c for c in daily_cols if c in df.columns]))

    df_daily = df.groupby("Date")[daily_cols].sum().reset_index()

    df_daily["cr_install_to_kyc"] = df_daily[kyc_unique_col] / df_daily[installs_col] if kyc_unique_col else np.nan
    if trade_any_unique_col and kyc_unique_col:
        df_daily["cr_kyc_to_trade"] = df_daily[trade_any_unique_col] / df_daily[kyc_unique_col].replace(0, np.nan)
    else:
        df_daily["cr_kyc_to_trade"] = np.nan
    df_daily["cr_install_to_esign"] = df_daily[esign_unique_col] / df_daily[installs_col] if esign_unique_col else np.nan

    st.line_chart(
        df_daily.set_index("Date")[["cr_install_to_kyc", "cr_kyc_to_trade", "cr_install_to_esign"]],
        height=400
    )

    st.markdown(f"""
The chart shows **daily CRs**:
- Install → KYC  
- KYC → Trade  
- Install → Esign  

You can visually check if **campaign period** ({campaign_start_dt.date()} → {campaign_end_dt.date()}) lines up with higher CRs.
""")

    # NEW: Weekly Install → Verify Phone CR
    st.subheader("Weekly Install → Verify Phone CR")

    if phone_verify_col is None:
        st.info("No 'verify phone (Unique users)' column detected. Cannot plot weekly Install → Verify Phone CR.")
    else:
        weekly_cols = [installs_col, phone_verify_col]

        pre_week_pv = weekly_aggregate(df_pre, weekly_cols)
        camp_week_pv = weekly_aggregate(df_camp, weekly_cols)

        # compute CR (%) = verify_phone / installs * 100
        for df_w, label in [(pre_week_pv, "Pre"), (camp_week_pv, "Campaign")]:
            if not df_w.empty:
                denom = df_w[installs_col].replace(0, np.nan)
                df_w["CR (%)"] = (df_w[phone_verify_col] / denom) * 100
                df_w["period"] = label

        combined_week = pd.concat(
            [pre_week_pv, camp_week_pv],
            ignore_index=True
        ) if not pre_week_pv.empty or not camp_week_pv.empty else pd.DataFrame()

        if combined_week.empty:
            st.info("No weekly data available for Install → Verify Phone CR in the selected periods.")
        else:
            plot_df = combined_week.pivot(index="WeekStart", columns="period", values="CR (%)")
            st.line_chart(plot_df, height=350)
            st.markdown("""
This chart shows **weekly Install → Verify Phone CR (%)** for Pre vs Campaign.

Use it to see:
- Whether your acquisition during the campaign is bringing users who complete phone verification at a higher rate.
""")


# -------------------------------------------------
# Tab 3: Paid vs Organic
# -------------------------------------------------
with tab_paid_org:
    st.header("Paid vs Organic — Funnel Performance")

    if "media_group" not in df.columns:
        st.info("No media_group information available — cannot split Paid vs Organic.")
    else:
        results = []
        for group in ["Paid", "Organic", "Other"]:
            df_g_pre = df_pre[df_pre["media_group"] == group]
            df_g_camp = df_camp[df_camp["media_group"] == group]

            if df_g_pre.empty and df_g_camp.empty:
                continue

            pre_m = compute_funnel_metrics(
                df_g_pre,
                installs_col=installs_col,
                kyc_col=kyc_unique_col,
                esign_col=esign_unique_col,
                trade_user_col=trade_any_unique_col,
                trade_user_fallback_cols=trade_unique_fallback_cols,
                days_in_period=pre_days,
            )
            camp_m = compute_funnel_metrics(
                df_g_camp,
                installs_col=installs_col,
                kyc_col=kyc_unique_col,
                esign_col=esign_unique_col,
                trade_user_col=trade_any_unique_col,
                trade_user_fallback_cols=trade_unique_fallback_cols,
                days_in_period=campaign_days,
            )

            results.append({
                "Media group": group,
                "Pre Installs/day": pre_m["installs_per_day"],
                "Campaign Installs/day": camp_m["installs_per_day"],
                "Pre Install→KYC CR (%)": pre_m["cr_install_to_kyc"] * 100 if pd.notna(pre_m["cr_install_to_kyc"]) else np.nan,
                "Campaign Install→KYC CR (%)": camp_m["cr_install_to_kyc"] * 100 if pd.notna(camp_m["cr_install_to_kyc"]) else np.nan,
                "Lift (Install→KYC CR %)": compute_lift_relative(
                    pre_m["cr_install_to_kyc"] * 100 if pd.notna(pre_m["cr_install_to_kyc"]) else np.nan,
                    camp_m["cr_install_to_kyc"] * 100 if pd.notna(camp_m["cr_install_to_kyc"]) else np.nan
                ),
                "Pre KYC→Trade CR (%)": pre_m["cr_kyc_to_trade"] * 100 if pd.notna(pre_m["cr_kyc_to_trade"]) else np.nan,
                "Campaign KYC→Trade CR (%)": camp_m["cr_kyc_to_trade"] * 100 if pd.notna(camp_m["cr_kyc_to_trade"]) else np.nan,
                "Lift (KYC→Trade CR %)": compute_lift_relative(
                    pre_m["cr_kyc_to_trade"] * 100 if pd.notna(pre_m["cr_kyc_to_trade"]) else np.nan,
                    camp_m["cr_kyc_to_trade"] * 100 if pd.notna(camp_m["cr_kyc_to_trade"]) else np.nan
                ),
            })

        if not results:
            st.info("No Paid / Organic / Other groups with data in selected periods.")
        else:
            res_df = pd.DataFrame(results)
            for col in res_df.columns:
                if "CR (%)" in col or "Lift" in col:
                    res_df[col] = res_df[col].map(format_pct)
                if "Installs/day" in col:
                    res_df[col] = res_df[col].map(lambda x: f"{x:.1f}" if pd.notna(x) else "NA")
            st.dataframe(res_df, use_container_width=True)

            st.markdown("""
This table compares **Paid vs Organic vs Other** on:

- Installs/day  
- Install → KYC CR (and lift)  
- KYC → Trade CR (and lift)  

Use this to answer:  
> “Did the campaign improve **lead quality** for Paid traffic?”
""")


# -------------------------------------------------
# Tab 4: Media Source Leaderboard
# -------------------------------------------------
with tab_media:
    st.header("Media Source Leaderboard (Campaign vs Pre)")

    if media_col is None:
        st.info("No media source column — cannot compute leaderboard.")
    else:
        media_list = df[media_col].dropna().unique().tolist()
        rows = []
        for ms in media_list:
            df_ms_pre = df_pre[df_pre[media_col] == ms]
            df_ms_camp = df_camp[df_camp[media_col] == ms]

            if df_ms_pre.empty and df_ms_camp.empty:
                continue

            pre_m = compute_funnel_metrics(
                df_ms_pre,
                installs_col=installs_col,
                kyc_col=kyc_unique_col,
                esign_col=esign_unique_col,
                trade_user_col=trade_any_unique_col,
                trade_user_fallback_cols=trade_unique_fallback_cols,
                days_in_period=pre_days,
            )
            camp_m = compute_funnel_metrics(
                df_ms_camp,
                installs_col=installs_col,
                kyc_col=kyc_unique_col,
                esign_col=esign_unique_col,
                trade_user_col=trade_any_unique_col,
                trade_user_fallback_cols=trade_unique_fallback_cols,
                days_in_period=campaign_days,
            )

            rows.append({
                "Media Source": ms,
                "Pre Installs/day": pre_m["installs_per_day"],
                "Campaign Installs/day": camp_m["installs_per_day"],
                "Pre Install→KYC CR (%)": pre_m["cr_install_to_kyc"] * 100 if pd.notna(pre_m["cr_install_to_kyc"]) else np.nan,
                "Campaign Install→KYC CR (%)": camp_m["cr_install_to_kyc"] * 100 if pd.notna(camp_m["cr_install_to_kyc"]) else np.nan,
                "Lift (Install→KYC CR %)": compute_lift_relative(
                    pre_m["cr_install_to_kyc"] * 100 if pd.notna(pre_m["cr_install_to_kyc"]) else np.nan,
                    camp_m["cr_install_to_kyc"] * 100 if pd.notna(camp_m["cr_install_to_kyc"]) else np.nan
                ),
                "Pre KYC→Trade CR (%)": pre_m["cr_kyc_to_trade"] * 100 if pd.notna(pre_m["cr_kyc_to_trade"]) else np.nan,
                "Campaign KYC→Trade CR (%)": camp_m["cr_kyc_to_trade"] * 100 if pd.notna(camp_m["cr_kyc_to_trade"]) else np.nan,
                "Lift (KYC→Trade CR %)": compute_lift_relative(
                    pre_m["cr_kyc_to_trade"] * 100 if pd.notna(pre_m["cr_kyc_to_trade"]) else np.nan,
                    camp_m["cr_kyc_to_trade"] * 100 if pd.notna(camp_m["cr_kyc_to_trade"]) else np.nan
                ),
            })

        if not rows:
            st.info("No media sources with data in selected periods.")
        else:
            ms_df = pd.DataFrame(rows)

            ms_df_sorted = ms_df.sort_values(by="Campaign Install→KYC CR (%)", ascending=False)

            ms_df_display = ms_df_sorted.copy()
            for col in ms_df_display.columns:
                if "CR (%)" in col or "Lift" in col:
                    ms_df_display[col] = ms_df_display[col].map(format_pct)
                if "Installs/day" in col:
                    ms_df_display[col] = ms_df_display[col].map(lambda x: f"{x:.1f}" if pd.notna(x) else "NA")

            st.dataframe(ms_df_display, use_container_width=True)

            st.markdown("""
Use this leaderboard to see **which sources improved most** in:

- Install→KYC conversion  
- KYC→Trade conversion  

You can highlight:
> “Campaign drove better-quality users specifically from Source X and Y.”
""")


# -------------------------------------------------
# Tab 5: Trade s2s Breakdown
# -------------------------------------------------
with tab_s2s:
    st.header("Trade-related s2s Events Breakdown (Pre vs Campaign)")

    if s2s_unique_cols:
        s2s_pre = df_pre[s2s_unique_cols].sum().to_frame(name="Pre unique users")
        s2s_camp = df_camp[s2s_unique_cols].sum().to_frame(name="Campaign unique users")
        s2s_combined = s2s_pre.join(s2s_camp, how="outer").fillna(0)

        st.dataframe(s2s_combined, use_container_width=True)

        st.markdown("""
This table shows **unique users** firing each trade-related **s2s event** in Pre vs Campaign.

You can compute:
- Any_trade_s2s uplift  
- Equity / F&O / ETF / MTF trade user uplift  

And link it back to **lead quality**:
> “Campaign brought in users who were more likely to trade (higher trade_s2s penetration among new KYC).”
""")
    else:
        st.info("No s2s trade-related unique user columns found in dataset.")


# -------------------------------------------------
# Tab 6: Custom Funnel & Weekly Lifts
# -------------------------------------------------
with tab_custom:
    st.header("Custom Funnel Builder & Weekly Lift Tables")

    st.markdown("""
Use this to build **any funnel you want** (e.g. Installs → Signup → KYC → Any trade → Esign) and see:

- Period-level **conversion & lift** for each step.
- **Base → step** conversion (e.g. Install → KYC, Install → Esign).
- **Weekly tables** of counts and CRs for Pre and Campaign.
""")

    # Candidate step columns: Installs + any "unique users" columns
    user_cols = [c for c in df.columns if "unique users" in c.lower()]
    step_options = []

    if installs_col in df.columns:
        step_options.append(installs_col)

    # Try to order: KYC, trade, esign first, then others
    special_order = []
    if kyc_unique_col and kyc_unique_col in user_cols:
        special_order.append(kyc_unique_col)
    if trade_any_unique_col and trade_any_unique_col in user_cols:
        special_order.append(trade_any_unique_col)
    if esign_unique_col and esign_unique_col in user_cols:
        special_order.append(esign_unique_col)

    remaining = [c for c in user_cols if c not in special_order]
    step_options.extend(special_order + remaining)

    step_options = list(dict.fromkeys(step_options))  # de-dup

    default_steps = []
    if installs_col in step_options:
        default_steps.append(installs_col)
    if kyc_unique_col and kyc_unique_col in step_options:
        default_steps.append(kyc_unique_col)
    if trade_any_unique_col and trade_any_unique_col in step_options:
        default_steps.append(trade_any_unique_col)
    if esign_unique_col and esign_unique_col in step_options:
        default_steps.append(esign_unique_col)

    selected_steps = st.multiselect(
        "Select funnel steps (in order). First step is the base (denominator).",
        options=step_options,
        default=default_steps if default_steps else step_options[:3],
        help="These should be user-count style columns (e.g. Installs, verify_kyc (Unique users), any_trade_s2s (Unique users), esign (Unique users), ...)."
    )

    if len(selected_steps) < 2:
        st.info("Select at least two steps to compute funnel conversions and lifts.")
    else:
        # Period-level totals for selected steps
        pre_totals = {step: df_pre[step].sum() if step in df_pre.columns else 0 for step in selected_steps}
        camp_totals = {step: df_camp[step].sum() if step in df_camp.columns else 0 for step in selected_steps}

        st.subheader("Period-level: Pairwise Step Conversion & Lift")
        pair_df = build_pairwise_lift_table(selected_steps, pre_totals, camp_totals)
        pair_display = pair_df.copy()
        for col in ["Pre CR (%)", "Campaign CR (%)", "Lift (%)"]:
            pair_display[col] = pair_display[col].map(format_pct)
        st.dataframe(pair_display, use_container_width=True)

        st.subheader("Period-level: Base → Step Conversion & Lift")
        base_df = build_base_to_step_lift_table(selected_steps, pre_totals, camp_totals)
        base_display = base_df.copy()
        for col in ["Pre CR (%)", "Campaign CR (%)", "Lift (%)"]:
            base_display[col] = base_display[col].map(format_pct)
        st.dataframe(base_display, use_container_width=True)

        st.markdown("""
**Interpretation:**
- **Pairwise table** – each row is *step i → step i+1* (local funnel CR).
- **Base → Step table** – each row is *base step → later step* (cumulative CR from first step).
- Lift shows how the funnel efficiency changed from pre → campaign.
""")

        st.subheader("Weekly tables for selected funnel steps")

        # Weekly aggregation for pre & campaign
        pre_week = weekly_aggregate(df_pre, selected_steps)
        camp_week = weekly_aggregate(df_camp, selected_steps)

        pre_week = add_cr_columns_weekly(pre_week, selected_steps)
        camp_week = add_cr_columns_weekly(camp_week, selected_steps)

        col_w1, col_w2 = st.columns(2)

        with col_w1:
            st.markdown("**Pre period – weekly funnel**")
            if pre_week.empty:
                st.info("No pre-period data to show weekly table.")
            else:
                st.dataframe(pre_week, use_container_width=True)

        with col_w2:
            st.markdown("**Campaign period – weekly funnel**")
            if camp_week.empty:
                st.info("No campaign-period data to show weekly table.")
            else:
                st.dataframe(camp_week, use_container_width=True)

        st.markdown("""
Use these **weekly tables** to:
- See if funnel CRs jump systematically in campaign weeks.
- Spot weeks where lead quality (KYC / Trade / Esign concentration) is unusually high or low.
""")

        # Optional grouped funnel by campaign_channel / persona / product
        if group_by != "None":
            st.subheader(f"Grouped funnel lift by {group_by}")

            groups = df[group_by].dropna().unique().tolist()
            groups = [g for g in groups if g not in ("", "Organic/None")] + (["Organic/None"] if "Organic/None" in groups else [])
            groups = [g for g in groups if g is not None]

            rows = []
            base = selected_steps[0]
            last = selected_steps[-1]

            for g in groups:
                df_pre_g = df_pre[df_pre[group_by] == g]
                df_camp_g = df_camp[df_camp[group_by] == g]

                if df_pre_g.empty and df_camp_g.empty:
                    continue

                pre_totals_g = {step: df_pre_g[step].sum() if step in df_pre_g.columns else 0 for step in selected_steps}
                camp_totals_g = {step: df_camp_g[step].sum() if step in df_camp_g.columns else 0 for step in selected_steps}

                pre_base = pre_totals_g.get(base, 0)
                pre_last = pre_totals_g.get(last, 0)
                camp_base = camp_totals_g.get(base, 0)
                camp_last = camp_totals_g.get(last, 0)

                pre_cr = (pre_last / pre_base * 100) if pre_base > 0 else np.nan
                camp_cr = (camp_last / camp_base * 100) if camp_base > 0 else np.nan
                lift = compute_lift_relative(pre_cr, camp_cr)

                rows.append({
                    group_by: g,
                    f"Pre {base}→{last} CR (%)": pre_cr,
                    f"Camp {base}→{last} CR (%)": camp_cr,
                    "Lift (%)": lift,
                    "Pre base count": pre_base,
                    "Camp base count": camp_base,
                    "Pre last count": pre_last,
                    "Camp last count": camp_last,
                })

            if not rows:
                st.info(f"No data to show by {group_by}.")
            else:
                grp_df = pd.DataFrame(rows)
                for col in grp_df.columns:
                    if "CR (%)" in col or "Lift (%)" in col:
                        grp_df[col] = grp_df[col].map(format_pct)
                st.dataframe(grp_df, use_container_width=True)

                st.markdown(f"""
Each row = one **{group_by}** bucket (e.g. channel / persona / product) with:

- Base → Last-step CR (Pre vs Campaign)
- % Lift
- Base & last-step volumes
""")

st.markdown("---")
st.markdown("**Note:** The app uses time-normalised comparisons (equal-length pre vs campaign, plus per-day metrics) to avoid misleading volume differences.")
