import streamlit as st
import sqlite3
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="AI Startup Success Prediction",
    page_icon="🚀",
    layout="wide"
)

# ---------------- CUSTOM THEME (CSS) ----------------

st.markdown("""
<style>
    /* Overall app background */
    .stApp {
        background-color: #F4F7FF;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1E2761;
    }
    section[data-testid="stSidebar"] * {
        color: #E8EEFD !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 15px;
        padding: 6px 0;
    }

    /* Headings */
    h1 {
        color: #1E2761 !important;
        font-weight: 800 !important;
        padding-bottom: 4px;
    }
    h2, h3 {
        color: #1E2761 !important;
        font-weight: 700 !important;
    }

    /* Cards effect for containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 6px;
        box-shadow: 0 2px 10px rgba(30, 39, 97, 0.08);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        font-weight: 600;
        color: #1E2761;
        border: 1px solid #E2E9F8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E2761 !important;
        color: #FFFFFF !important;
    }

    /* Buttons */
    .stButton button {
        background-color: #1E2761;
        color: #FFFFFF;
        border-radius: 10px;
        border: none;
        font-weight: 700;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background-color: #2C3B8A;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(30, 39, 97, 0.25);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E9F8;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 2px 8px rgba(30, 39, 97, 0.06);
    }

    /* Progress bar */
    .stProgress > div > div {
        background-color: #1E2761;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #FFFFFF;
        border-radius: 10px;
        font-weight: 600;
        color: #1E2761;
    }

    /* Dataframe / table corners */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- PATHS ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "startup.db")
MODEL_PATH = os.path.join(BASE_DIR, "startup_model.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")

# ---------------- LOAD MODEL ----------------

model = joblib.load(MODEL_PATH)

# ---------------- DATABASE ----------------

def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_date TEXT,
        age_first_funding REAL,
        age_last_funding REAL,
        age_first_milestone REAL,
        age_last_milestone REAL,
        relationships INTEGER,
        funding_rounds INTEGER,
        funding_total_usd REAL,
        milestones INTEGER,
        confidence REAL,
        prediction TEXT
    )
    """)
    conn.commit()
    conn.close()

create_database()

# ---------------- MIGRATE DATABASE ----------------

def migrate_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(predictions)")
    existing_columns = [row[1] for row in cursor.fetchall()]
    if "prediction_date" not in existing_columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN prediction_date TEXT")
    if "confidence" not in existing_columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN confidence REAL")
    conn.commit()
    conn.close()

migrate_database()

# ---------------- SAVE PREDICTION ----------------

def save_prediction(
    age_first_funding_year, age_last_funding_year,
    age_first_milestone_year, age_last_milestone_year,
    relationships, funding_rounds, funding_total_usd, milestones,
    confidence, prediction
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO predictions(
        prediction_date,
        age_first_funding, age_last_funding,
        age_first_milestone, age_last_milestone,
        relationships, funding_rounds, funding_total_usd, milestones,
        confidence, prediction
    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
    """,(
        datetime.now().strftime("%d-%m-%Y %H:%M"),
        age_first_funding_year, age_last_funding_year,
        age_first_milestone_year, age_last_milestone_year,
        relationships, funding_rounds, funding_total_usd, milestones,
        confidence, prediction
    ))
    conn.commit()
    conn.close()

# ---------------- DELETE PREDICTION ----------------

def delete_prediction(pred_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    conn.commit()
    conn.close()

# ---------------- LOAD DATASET HELPER ----------------

DROP_COLUMNS = [
    "Unnamed: 0", "state_code", "latitude", "longitude", "zip_code",
    "id", "city", "Unnamed: 6", "name", "labels", "founded_at",
    "closed_at", "first_funding_at", "last_funding_at",
    "state_code.1", "object_id", "category_code"
]

@st.cache_data
def load_model_data():
    df = pd.read_csv(DATASET_PATH)
    for col in DROP_COLUMNS:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    df["status"] = df["status"].map({"closed": 0, "acquired": 1, "operating": 1, "ipo": 1})
    df = df.dropna(subset=["status"]).fillna(0)
    X = df.drop("status", axis=1)
    y = df["status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X.columns.tolist()

# ---------------- SIDEBAR ----------------

st.sidebar.title("🚀 Navigation")

menu = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "🚀 Prediction",
        "📊 Prediction History",
        "🤖 AI Insights",
        "🧪 What-If Lab",
        "📈 Dashboard",
        "ℹ About"
    ]
)

# ---------------- HOME PAGE ----------------

if menu == "🏠 Home":

    st.title("🚀 AI Powered Startup Success Prediction System")
    st.markdown("---")
    st.write("""
Welcome to the **AI Powered Startup Success Prediction System**.

This application predicts whether a startup is likely to succeed
using a trained Machine Learning model.

### Features

✅ Startup Success Prediction

✅ Random Forest Machine Learning Model

✅ SQLite Database Connectivity

✅ AI Insights with Confusion Matrix & Feature Importance

✅ Prediction History with Delete & Filter

✅ Data Validation

✅ Dashboard

✅ Confidence Score

Use the navigation menu on the left to explore the application.
""")

# ---------------- PREDICTION PAGE ----------------

elif menu == "🚀 Prediction":

    st.title("🚀 Startup Success Prediction")
    st.caption("Fill in the details across each tab below, then check the live preview or run the full prediction.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📋 Startup Metrics", "📍 Location & Category", "💰 Investment Details"])

    with tab1:
        st.subheader("📋 Startup Metrics")
        col1, col2 = st.columns(2)
        with col1:
            age_first_funding_year = st.number_input("Age at First Funding (years)", 0.0, 100.0)
            age_last_funding_year = st.number_input("Age at Last Funding (years)", 0.0, 100.0)
            age_first_milestone_year = st.number_input("Age at First Milestone (years)", 0.0, 100.0)
            age_last_milestone_year = st.number_input("Age at Last Milestone (years)", 0.0, 100.0)
        with col2:
            relationships = st.number_input("Relationships", 0)
            funding_rounds = st.number_input("Funding Rounds", 0)
            funding_total_usd = st.number_input("Funding Total (₹)", 0)
            milestones = st.number_input("Milestones", 0)

    with tab2:
        st.subheader("📍 Location & Category")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**State**")
            is_CA = st.checkbox("Maharashtra")
            is_NY = st.checkbox("Delhi")
            is_MA = st.checkbox("Karnataka")
            is_TX = st.checkbox("Tamil Nadu")
            is_otherstate = st.checkbox("Other State")
        with col4:
            st.markdown("**Category**")
            is_software = st.checkbox("Software")
            is_web = st.checkbox("Web")
            is_mobile = st.checkbox("Mobile")
            is_enterprise = st.checkbox("Enterprise")
            is_advertising = st.checkbox("Advertising")
            is_gamesvideo = st.checkbox("Games / Video")
            is_ecommerce = st.checkbox("E-Commerce")
            is_biotech = st.checkbox("Biotech")
            is_consulting = st.checkbox("Consulting")
            is_othercategory = st.checkbox("Other Category")

    with tab3:
        st.subheader("💰 Investment Details")
        col5, col6 = st.columns(2)
        with col5:
            has_VC = st.checkbox("Has VC")
            has_angel = st.checkbox("Has Angel")
            has_roundA = st.checkbox("Has Round A")
            has_roundB = st.checkbox("Has Round B")
            has_roundC = st.checkbox("Has Round C")
            has_roundD = st.checkbox("Has Round D")
        with col6:
            avg_participants = st.number_input("Average Participants", 0.0)
            is_top500 = st.checkbox("Top 500 Startup")

    st.markdown("---")

    # ---- REAL-TIME LIVE PREVIEW ----
    # Streamlit reruns this whole script automatically on every widget
    # interaction, so this section recalculates instantly as the user
    # types or checks a box — no button click required.
    st.subheader("⚡ Live Prediction Preview (Real-Time)")
    st.caption("Updates instantly as you adjust the inputs above — no button needed.")

    live_data = np.array([[
        age_first_funding_year, age_last_funding_year,
        age_first_milestone_year, age_last_milestone_year,
        relationships, funding_rounds, funding_total_usd, milestones,
        int(is_CA), int(is_NY), int(is_MA), int(is_TX), int(is_otherstate),
        int(is_software), int(is_web), int(is_mobile), int(is_enterprise),
        int(is_advertising), int(is_gamesvideo), int(is_ecommerce),
        int(is_biotech), int(is_consulting), int(is_othercategory),
        int(has_VC), int(has_angel), int(has_roundA), int(has_roundB),
        int(has_roundC), int(has_roundD),
        avg_participants, int(is_top500)
    ]])

    live_prediction = model.predict(live_data)
    live_probability = model.predict_proba(live_data)
    live_confidence = round(np.max(live_probability) * 100, 2)

    live_col1, live_col2 = st.columns(2)
    with live_col1:
        if live_prediction[0] == 1:
            st.success("📈 Trending toward: **Success**")
        else:
            st.warning("📉 Trending toward: **Failure**")
    with live_col2:
        st.metric("Live Confidence", f"{live_confidence}%")

    st.progress(int(live_confidence))

    st.markdown("---")

    # ---- PREDICT BUTTON (centered) ----
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

    with col_btn2:
        predict_clicked = st.button("🚀 Predict Startup", use_container_width=True)

    if predict_clicked:

        # ---- DATA VALIDATION ----
        errors = []

        if funding_rounds == 0:
            errors.append("⚠️ Funding Rounds cannot be 0.")
        if funding_total_usd == 0:
            errors.append("⚠️ Funding Total (₹) cannot be 0.")
        if age_last_funding_year < age_first_funding_year:
            errors.append("⚠️ Age at Last Funding cannot be less than Age at First Funding.")
        if age_last_milestone_year < age_first_milestone_year:
            errors.append("⚠️ Age at Last Milestone cannot be less than Age at First Milestone.")
        state_selected = any([is_CA, is_NY, is_MA, is_TX, is_otherstate])
        if not state_selected:
            errors.append("⚠️ Please select at least one State.")
        category_selected = any([is_software, is_web, is_mobile, is_enterprise, is_advertising,
                                  is_gamesvideo, is_ecommerce, is_biotech, is_consulting, is_othercategory])
        if not category_selected:
            errors.append("⚠️ Please select at least one Category.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            data = np.array([[
                age_first_funding_year, age_last_funding_year,
                age_first_milestone_year, age_last_milestone_year,
                relationships, funding_rounds, funding_total_usd, milestones,
                int(is_CA), int(is_NY), int(is_MA), int(is_TX), int(is_otherstate),
                int(is_software), int(is_web), int(is_mobile), int(is_enterprise),
                int(is_advertising), int(is_gamesvideo), int(is_ecommerce),
                int(is_biotech), int(is_consulting), int(is_othercategory),
                int(has_VC), int(has_angel), int(has_roundA), int(has_roundB),
                int(has_roundC), int(has_roundD),
                avg_participants, int(is_top500)
            ]])

            prediction = model.predict(data)
            probability = model.predict_proba(data)
            confidence = round(np.max(probability) * 100, 2)

            if prediction[0] == 1:
                result = "Success"
                st.success("✅ Startup is likely to Succeed!")
            else:
                result = "Failure"
                st.error("❌ Startup is likely to Fail!")

            st.metric("Prediction Confidence", f"{confidence}%")

            save_prediction(
                age_first_funding_year, age_last_funding_year,
                age_first_milestone_year, age_last_milestone_year,
                relationships, funding_rounds, funding_total_usd, milestones,
                confidence, result
            )

            st.success("✅ Prediction saved to database successfully!")

            st.markdown("---")
            st.subheader("🤖 AI-Generated Insights")
            st.caption("Generated dynamically by comparing your inputs against dataset averages and the model's learned feature importances — not static text.")

            # ---- Pull dataset averages for comparison ----
            X_train_ref, X_test_ref, y_train_ref, y_test_ref, feat_names_ref = load_model_data()
            full_X_ref = pd.concat([X_train_ref, X_test_ref])
            avg_funding_rounds = full_X_ref["funding_rounds"].mean()
            avg_milestones = full_X_ref["milestones"].mean()
            avg_relationships = full_X_ref["relationships"].mean()
            avg_funding_total = full_X_ref["funding_total_usd"].mean()

            # ---- Get model's top feature importances ----
            importances = model.feature_importances_
            feat_importance_map = dict(zip(feat_names_ref, importances))

            insights = []

            # Funding rounds comparison
            if funding_rounds > avg_funding_rounds:
                diff_pct = round(((funding_rounds - avg_funding_rounds) / avg_funding_rounds) * 100, 1)
                insights.append(f"✅ Your **{funding_rounds} funding rounds** is **{diff_pct}% above** the dataset average of {avg_funding_rounds:.1f} — a strong positive signal.")
            else:
                diff_pct = round(((avg_funding_rounds - funding_rounds) / avg_funding_rounds) * 100, 1)
                insights.append(f"⚠️ Your **{funding_rounds} funding rounds** is **{diff_pct}% below** the dataset average of {avg_funding_rounds:.1f} — consider pursuing additional rounds.")

            # Milestones comparison
            if milestones > avg_milestones:
                diff_pct = round(((milestones - avg_milestones) / max(avg_milestones, 0.1)) * 100, 1)
                insights.append(f"✅ Your **{milestones} milestones** is **{diff_pct}% above** the dataset average of {avg_milestones:.1f} — strong execution track record.")
            else:
                diff_pct = round(((avg_milestones - milestones) / max(avg_milestones, 0.1)) * 100, 1)
                insights.append(f"⚠️ Your **{milestones} milestones** is **{diff_pct}% below** the dataset average of {avg_milestones:.1f} — more milestones would strengthen the profile.")

            # Relationships / investor network comparison
            if relationships > avg_relationships:
                diff_pct = round(((relationships - avg_relationships) / max(avg_relationships, 0.1)) * 100, 1)
                insights.append(f"✅ Your **{relationships} investor relationships** is **{diff_pct}% above** average ({avg_relationships:.1f}) — a well-connected network.")
            else:
                diff_pct = round(((avg_relationships - relationships) / max(avg_relationships, 0.1)) * 100, 1)
                insights.append(f"⚠️ Your **{relationships} investor relationships** is **{diff_pct}% below** average ({avg_relationships:.1f}) — expanding investor connections could help.")

            # Funding amount comparison
            if funding_total_usd > avg_funding_total:
                insights.append(f"✅ Total funding of **₹{funding_total_usd:,.0f}** exceeds the dataset average of ₹{avg_funding_total:,.0f}.")
            else:
                insights.append(f"⚠️ Total funding of **₹{funding_total_usd:,.0f}** is below the dataset average of ₹{avg_funding_total:,.0f}.")

            # Top model-driven factor
            top_feature = max(feat_importance_map, key=feat_importance_map.get)
            top_feature_pct = round(feat_importance_map[top_feature] * 100, 1)
            insights.append(f"🧠 The single most influential factor in this model is **{top_feature.replace('_', ' ').title()}**, contributing **{top_feature_pct}%** to every prediction it makes.")

            if result == "Success":
                st.success("**Overall: This startup shows patterns consistent with successful startups in the dataset.**")
            else:
                st.warning("**Overall: This startup shows patterns more consistent with startups that did not succeed in the dataset.**")

            for point in insights:
                st.markdown(f"- {point}")

# ---------------- PREDICTION HISTORY ----------------

elif menu == "📊 Prediction History":

    st.title("📊 Prediction History")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()

    if len(df) > 0:

        # ---- FILTER ----
        st.subheader("🔍 Filter")
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            filter_result = st.selectbox("Filter by Result", ["All", "Success", "Failure"])

        with col_f2:
            filter_confidence = st.slider("Minimum Confidence %", 0, 100, 0)

        if filter_result != "All":
            df = df[df["prediction"] == filter_result]
        df = df[df["confidence"] >= filter_confidence]

        st.success(f"Showing {len(df)} prediction(s)")
        st.dataframe(df, use_container_width=True)

        st.markdown("---")

        # ---- DELETE ----
        st.subheader("🗑️ Delete a Prediction")
        delete_id = st.number_input("Enter Prediction ID to delete", min_value=1, step=1)
        if st.button("🗑️ Delete"):
            delete_prediction(delete_id)
            st.success(f"Prediction ID {delete_id} deleted successfully!")
            st.rerun()

        st.markdown("---")

        # ---- DOWNLOAD ----
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download History as CSV",
            csv,
            "prediction_history.csv",
            "text/csv"
        )

    else:
        st.warning("No predictions found. Make a prediction first!")

# ---------------- AI INSIGHTS ----------------

elif menu == "🤖 AI Insights":

    st.title("🤖 AI Insights")

    try:
        X_train, X_test, y_train, y_test, feature_names = load_model_data()
        preds = model.predict(X_test)
        accuracy = round(accuracy_score(y_test, preds) * 100, 2)
        accuracy_display = f"{accuracy}%"
        X_test = X_test
        y_test = y_test
        preds = preds
        feature_names = feature_names
    except Exception as e:
        accuracy_display = "N/A"
        X_test = y_test = preds = feature_names = None
        st.error(f"Could not calculate accuracy: {e}")

    # ---- METRICS ----
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "Random Forest")
    col2.metric("Accuracy", accuracy_display)
    col3.metric("Database", "SQLite")

    st.markdown("---")

    # ---- CONFUSION MATRIX ----
    if preds is not None:
        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_test, preds)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Failure", "Actual: Success"],
            columns=["Predicted: Failure", "Predicted: Success"]
        )
        st.dataframe(cm_df, use_container_width=True)

        st.markdown("---")

        # ---- CLASSIFICATION REPORT ----
        st.subheader("📋 Classification Report")
        report = classification_report(y_test, preds, target_names=["Failure", "Success"], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

        st.markdown("---")

        # ---- FEATURE IMPORTANCE ----
        st.subheader("🔑 Top 10 Feature Importance")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(10)
        st.bar_chart(feat_df.set_index("Feature")["Importance"])

        st.markdown("---")

    # ---- AI FEATURES ----
    st.subheader("✅ AI Features")
    st.success("✔ Machine Learning Prediction")
    st.success("✔ Confidence Score")
    st.success("✔ Confusion Matrix")
    st.success("✔ Classification Report")
    st.success("✔ Feature Importance")
    st.success("✔ Database Connectivity")
    st.success("✔ Historical Prediction Storage")

# ---------------- WHAT-IF LAB ----------------

elif menu == "🧪 What-If Lab":

    st.title("🧪 What-If Lab")
    st.caption("Simulate changes live, see exactly why the model decided what it decided, get a graded health score, and download a full report.")
    st.markdown("---")

    # Pull the most recent saved prediction as a starting point, if one exists
    conn = sqlite3.connect(DB_PATH)
    last_row_df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT 1", conn)
    conn.close()

    if last_row_df.empty:
        st.info("💡 Make a prediction first on the **Prediction** page — this lab will load your latest inputs as a starting point. For now, default values are used below.")
        seed = {
            "age_first_funding": 1.5, "age_last_funding": 3.0,
            "age_first_milestone": 1.0, "age_last_milestone": 2.5,
            "relationships": 5, "funding_rounds": 2,
            "funding_total_usd": 5000000.0, "milestones": 2,
        }
    else:
        r = last_row_df.iloc[0]
        seed = {
            "age_first_funding": float(r.get("age_first_funding", 1.5) or 1.5),
            "age_last_funding": float(r.get("age_last_funding", 3.0) or 3.0),
            "age_first_milestone": float(r.get("age_first_milestone", 1.0) or 1.0),
            "age_last_milestone": float(r.get("age_last_milestone", 2.5) or 2.5),
            "relationships": int(r.get("relationships", 5) or 5),
            "funding_rounds": int(r.get("funding_rounds", 2) or 2),
            "funding_total_usd": float(r.get("funding_total_usd", 5000000.0) or 5000000.0),
            "milestones": int(r.get("milestones", 2) or 2),
        }
        st.success(f"📎 Loaded your latest saved prediction as the starting point (originally predicted: **{r.get('prediction', 'N/A')}**)")

    st.markdown("### 🎛️ Adjust the Sliders — Watch the Prediction Update Live")

    sl1, sl2 = st.columns(2)
    with sl1:
        sim_age_first_funding = st.slider("Age at First Funding (yrs)", 0.0, 15.0, float(seed["age_first_funding"]), 0.1)
        sim_age_last_funding = st.slider("Age at Last Funding (yrs)", 0.0, 15.0, float(seed["age_last_funding"]), 0.1)
        sim_age_first_milestone = st.slider("Age at First Milestone (yrs)", 0.0, 15.0, float(seed["age_first_milestone"]), 0.1)
        sim_age_last_milestone = st.slider("Age at Last Milestone (yrs)", 0.0, 15.0, float(seed["age_last_milestone"]), 0.1)
    with sl2:
        sim_relationships = st.slider("Investor Relationships", 0, 30, int(seed["relationships"]))
        sim_funding_rounds = st.slider("Funding Rounds", 0, 12, int(seed["funding_rounds"]))
        sim_funding_total = st.slider("Total Funding (₹, in lakhs)", 0, 5000, int(seed["funding_total_usd"] / 100000), 10) * 100000
        sim_milestones = st.slider("Milestones Achieved", 0, 10, int(seed["milestones"]))

    # Build the full 31-feature vector — categorical/location flags default to
    # the most common values in the dataset so the simulation stays realistic
    sim_data = np.array([[
        sim_age_first_funding, sim_age_last_funding,
        sim_age_first_milestone, sim_age_last_milestone,
        sim_relationships, sim_funding_rounds, sim_funding_total, sim_milestones,
        1, 0, 0, 0, 0,        # state: Maharashtra
        1, 0, 0, 0,           # category: Software
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0,     # has_VC, has_angel, has_roundA
        3.0, 0                # avg_participants, is_top500
    ]])

    sim_prediction = model.predict(sim_data)
    sim_probability = model.predict_proba(sim_data)
    sim_confidence = round(np.max(sim_probability) * 100, 2)
    success_prob = round(sim_probability[0][1] * 100, 2)

    st.markdown("---")
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        if sim_prediction[0] == 1:
            st.success("📈 **Success**")
        else:
            st.error("📉 **Failure**")
    with res_col2:
        st.metric("Success Probability", f"{success_prob}%")
    with res_col3:
        st.metric("Model Confidence", f"{sim_confidence}%")
    st.progress(int(success_prob))

    st.markdown("---")

    # ---- TABS: Explainability | Health Score | Report ----
    explain_tab, health_tab, report_tab = st.tabs(["🔍 Why This Prediction?", "🏥 Health Score & Action Plan", "📄 Download Report"])

    # ===== TAB 1: SHAP EXPLAINABILITY =====
    with explain_tab:
        st.subheader("🔍 Why did the model decide this?")
        st.caption("Each bar shows how much a feature pushed THIS specific prediction toward Success (green) or Failure (red) — not just general importance.")

        if not SHAP_AVAILABLE:
            st.warning("Install `shap` for this feature: `pip install shap`")
        else:
            X_train_shap, X_test_shap, y_train_shap, y_test_shap, feat_names_shap = load_model_data()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sim_data)

            # shap_values shape handling for binary RandomForest
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0, :, 1] if shap_values.ndim == 3 else shap_values[0]

            shap_df = pd.DataFrame({
                "Feature": feat_names_shap,
                "Impact": sv
            }).sort_values("Impact", key=abs, ascending=False).head(8)

            shap_df["Direction"] = shap_df["Impact"].apply(lambda x: "Pushes toward Success" if x > 0 else "Pushes toward Failure")
            shap_df["Feature"] = shap_df["Feature"].str.replace("_", " ").str.title()

            chart_data = shap_df.set_index("Feature")["Impact"]
            st.bar_chart(chart_data, color="#1E2761")

            st.markdown("**Plain-language breakdown:**")
            for _, row in shap_df.iterrows():
                arrow = "🟢 ↑" if row["Impact"] > 0 else "🔴 ↓"
                st.markdown(f"- {arrow} **{row['Feature']}** — {row['Direction']} (impact score: {row['Impact']:.3f})")

    # ===== TAB 2: HEALTH SCORE + ACTION PLAN =====
    with health_tab:
        st.subheader("🏥 Startup Health Score")

        # Weighted scoring against dataset benchmarks (0-100 scale)
        X_train_h, X_test_h, y_train_h, y_test_h, feat_names_h = load_model_data()
        full_X_h = pd.concat([X_train_h, X_test_h])

        def score_metric(value, series, weight):
            pct_rank = (series < value).mean() * 100
            return min(pct_rank, 100) * weight

        score = 0
        score += score_metric(sim_funding_rounds, full_X_h["funding_rounds"], 0.25)
        score += score_metric(sim_milestones, full_X_h["milestones"], 0.25)
        score += score_metric(sim_relationships, full_X_h["relationships"], 0.25)
        score += score_metric(sim_funding_total, full_X_h["funding_total_usd"], 0.25)
        score = round(score, 1)

        if score >= 80:
            grade, grade_color = "A", "🟢"
        elif score >= 65:
            grade, grade_color = "B", "🟢"
        elif score >= 50:
            grade, grade_color = "C", "🟡"
        elif score >= 35:
            grade, grade_color = "D", "🟠"
        else:
            grade, grade_color = "F", "🔴"

        hcol1, hcol2 = st.columns([1, 2])
        with hcol1:
            st.markdown(f"<h1 style='text-align:center; font-size:80px; color:#1E2761;'>{grade_color} {grade}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; font-size:20px;'>Score: {score}/100</p>", unsafe_allow_html=True)
        with hcol2:
            st.markdown("**Grading scale:**")
            st.markdown("- **A (80-100):** Excellent — top-tier metrics vs. dataset benchmarks")
            st.markdown("- **B (65-79):** Strong — above average across most factors")
            st.markdown("- **C (50-64):** Average — comparable to typical startups in the data")
            st.markdown("- **D (35-49):** Below average — several weak areas")
            st.markdown("- **F (0-34):** High risk — significantly underperforming benchmarks")

        st.markdown("---")
        st.markdown("### 📋 Personalized Action Plan")

        action_items = []
        fr_percentile = (full_X_h["funding_rounds"] < sim_funding_rounds).mean() * 100
        if fr_percentile < 50:
            action_items.append(f"🎯 **Raise more funding rounds** — you're in the bottom {fr_percentile:.0f}% for funding rounds. Aim for at least {full_X_h['funding_rounds'].median():.0f} rounds.")
        ms_percentile = (full_X_h["milestones"] < sim_milestones).mean() * 100
        if ms_percentile < 50:
            action_items.append(f"🎯 **Hit more business milestones** — currently in the bottom {ms_percentile:.0f}%. Target {full_X_h['milestones'].median():.0f}+ milestones.")
        rel_percentile = (full_X_h["relationships"] < sim_relationships).mean() * 100
        if rel_percentile < 50:
            action_items.append(f"🎯 **Expand your investor network** — bottom {rel_percentile:.0f}% for relationships. Build toward {full_X_h['relationships'].median():.0f}+ connections.")
        fund_percentile = (full_X_h["funding_total_usd"] < sim_funding_total).mean() * 100
        if fund_percentile < 50:
            action_items.append(f"🎯 **Increase total funding raised** — bottom {fund_percentile:.0f}% by amount. Dataset median is ₹{full_X_h['funding_total_usd'].median():,.0f}.")

        if not action_items:
            st.success("✅ This startup profile is strong across all measured factors — no urgent gaps detected!")
        else:
            for item in action_items:
                st.markdown(f"- {item}")

    # ===== TAB 3: PDF REPORT =====
    with report_tab:
        st.subheader("📄 Downloadable Investor-Style Report")
        st.caption("Generates a one-page PDF summarizing this simulation — prediction, score, and key drivers.")

        if not REPORTLAB_AVAILABLE:
            st.warning("Install `reportlab` for this feature: `pip install reportlab`")
        else:
            if st.button("📄 Generate PDF Report"):
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm)
                styles = getSampleStyleSheet()
                title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], textColor=colors.HexColor("#1E2761"))
                heading_style = ParagraphStyle("HeadingStyle", parent=styles["Heading2"], textColor=colors.HexColor("#1E2761"))
                normal_style = styles["Normal"]

                elements = []
                elements.append(Paragraph("Startup Success Prediction Report", title_style))
                elements.append(Spacer(1, 6))
                elements.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", normal_style))
                elements.append(Spacer(1, 16))

                elements.append(Paragraph("Prediction Summary", heading_style))
                result_text = "Success" if sim_prediction[0] == 1 else "Failure"
                summary_data = [
                    ["Predicted Outcome", result_text],
                    ["Success Probability", f"{success_prob}%"],
                    ["Model Confidence", f"{sim_confidence}%"],
                    ["Health Score", f"{score}/100 (Grade {grade})"],
                ]
                t1 = Table(summary_data, colWidths=[80*mm, 80*mm])
                t1.setStyle(TableStyle([
                    ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#1E2761")),
                    ("TEXTCOLOR", (0,0), (0,-1), colors.white),
                    ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
                    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#D8E4F5")),
                    ("FONTSIZE", (0,0), (-1,-1), 10),
                    ("TOPPADDING", (0,0), (-1,-1), 6),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                ]))
                elements.append(t1)
                elements.append(Spacer(1, 16))

                elements.append(Paragraph("Input Parameters", heading_style))
                input_data = [
                    ["Funding Rounds", str(sim_funding_rounds)],
                    ["Milestones", str(sim_milestones)],
                    ["Investor Relationships", str(sim_relationships)],
                    ["Total Funding", f"₹{sim_funding_total:,.0f}"],
                    ["Age at First Funding", f"{sim_age_first_funding} yrs"],
                    ["Age at Last Funding", f"{sim_age_last_funding} yrs"],
                ]
                t2 = Table(input_data, colWidths=[80*mm, 80*mm])
                t2.setStyle(TableStyle([
                    ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#EEF3FC")),
                    ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
                    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#D8E4F5")),
                    ("FONTSIZE", (0,0), (-1,-1), 10),
                    ("TOPPADDING", (0,0), (-1,-1), 6),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                ]))
                elements.append(t2)
                elements.append(Spacer(1, 16))

                if action_items:
                    elements.append(Paragraph("Recommended Action Items", heading_style))
                    for item in action_items:
                        clean_item = item.replace("🎯", "").replace("**", "")
                        elements.append(Paragraph(f"• {clean_item}", normal_style))
                        elements.append(Spacer(1, 4))

                elements.append(Spacer(1, 20))
                elements.append(Paragraph(
                    "Generated by AI Powered Startup Success Prediction System | "
                    "Model: Random Forest Classifier | For educational purposes only.",
                    ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
                ))

                doc.build(elements)
                buffer.seek(0)

                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=buffer,
                    file_name=f"startup_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )
                st.success("✅ Report generated successfully!")

# ---------------- DASHBOARD ----------------

elif menu == "📈 Dashboard":

    st.title("📈 Dashboard")

    if not os.path.exists(DATASET_PATH):
        st.error("dataset.csv not found. Please make sure it is in the same folder as app.py.")
    else:
        df = pd.read_csv(DATASET_PATH)

        # ---- METRICS ----
        total = len(df)
        success_count = df["status"].isin(["acquired", "operating", "ipo"]).sum()
        fail_count = df["status"].isin(["closed"]).sum()
        success_rate = round(success_count / total * 100, 1)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Startups", total)
        col2.metric("Successful", success_count)
        col3.metric("Failed", fail_count)
        col4.metric("Success Rate", f"{success_rate}%")

        st.markdown("---")

        # ---- CHART PREP ----
        df["outcome"] = df["status"].apply(
            lambda x: "Success" if x in ["acquired", "operating", "ipo"] else "Failure"
        )

        # ---- ROW 1: Status Distribution | Success by Category ----
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("📊 Success vs Failure")
            status_counts = df["outcome"].value_counts().reset_index()
            status_counts.columns = ["Outcome", "Count"]
            st.bar_chart(status_counts.set_index("Outcome"), color="#4CAF50")

        with col_b:
            st.subheader("🏷️ Success Rate by Category")
            cat_group = df.groupby("category_code")["outcome"].value_counts().unstack(fill_value=0)
            if "Success" not in cat_group.columns:
                cat_group["Success"] = 0
            if "Failure" not in cat_group.columns:
                cat_group["Failure"] = 0
            cat_group = cat_group[["Success", "Failure"]].sort_values("Success", ascending=False).head(8)
            st.bar_chart(cat_group)

        st.markdown("---")

        # ---- ROW 2: Funding Rounds | Milestones ----
        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("🔄 Startups by Funding Rounds")
            rounds_df = df.groupby("funding_rounds")["outcome"].value_counts().unstack(fill_value=0)
            if "Success" not in rounds_df.columns:
                rounds_df["Success"] = 0
            if "Failure" not in rounds_df.columns:
                rounds_df["Failure"] = 0
            rounds_df.index.name = "Funding Rounds"
            st.bar_chart(rounds_df[["Success", "Failure"]])

        with col_d:
            st.subheader("🏆 Startups by Milestones")
            mile_df = df.groupby("milestones")["outcome"].value_counts().unstack(fill_value=0)
            if "Success" not in mile_df.columns:
                mile_df["Success"] = 0
            if "Failure" not in mile_df.columns:
                mile_df["Failure"] = 0
            mile_df.index.name = "Milestones"
            st.bar_chart(mile_df[["Success", "Failure"]])

        st.markdown("---")

        # ---- ROW 3: Avg Funding by State ----
        st.subheader("💰 Average Funding by State (Top 6) in ₹")
        state_funding = (
            df.groupby("state_code")["funding_total_usd"]
            .mean()
            .sort_values(ascending=False)
            .head(6)
            .reset_index()
        )
        state_funding.columns = ["State", "Avg Funding (₹)"]
        st.bar_chart(state_funding.set_index("State"))

        st.markdown("---")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

# ---------------- ABOUT ----------------

elif menu == "ℹ About":

    st.title("ℹ About Project")

    st.markdown("""
## AI Powered Startup Success Prediction System

This application predicts whether a startup is likely
to succeed using Machine Learning.

### Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Random Forest
- SQLite
- Pandas
- NumPy

### Features

- Startup Success Prediction with Data Validation
- AI Analysis with Confidence Score
- Confusion Matrix & Classification Report
- Feature Importance Chart
- SQLite Database with Filter & Delete
- Dashboard with Live Metrics
- Prediction History with CSV Export

Developed as a BCA Machine Learning Project.
""")
