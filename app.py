import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
import lightgbm as lgb
import joblib
import numpy as np

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="MedGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.stApp{
    background:#050A18;
    color:white;
}

[data-testid="stSidebar"]{
    background:#0B1120;
}

h1,h2,h3{
    color:#00D4FF;
}

div[data-testid="metric-container"]{
    background:#101B32;
    padding:15px;
    border-radius:15px;
    border:1px solid #00D4FF;
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================

if "df" not in st.session_state:
    st.session_state.df = None
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
@st.cache_resource
def load_models():

    xgb_model = xgb.Booster()
    xgb_model.load_model(
        str(BASE_DIR / "saved_models" / "model_xgb.json")
    )

    lgb_model = lgb.Booster(
        model_file=str(BASE_DIR / "saved_models" / "model_lgbm.txt")
    )

    rf_model = joblib.load(
        BASE_DIR / "saved_models" / "rf_model.joblib"
    )

    mlp_model = joblib.load(
        BASE_DIR / "saved_models" / "mlp_model.joblib"
    )

    stacker = joblib.load(
        BASE_DIR / "saved_models" / "stacker_model.joblib"
    )

    return xgb_model, lgb_model, rf_model, mlp_model, stacker
xgb_model, lgb_model, rf_model, mlp_model, stacker = load_models()
# =========================
# SIDEBAR
# =========================

st.sidebar.title("🛡️ MedGuard AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Upload Dataset",
        "Fraud Detection",
        "Visualizations",
        "Model Insights",
        "Download Report",
        "About"
    ]
)

# =========================
# HOME PAGE
# =========================

if page == "Home":

    st.markdown(
        """
        <h1 style='text-align:center;'>
        🛡️ MedGuard AI
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h3 style='text-align:center;color:white;'>
        AI Health Insurance Fraud Detection System
        </h3>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📄 Claims", "10,000")

    with col2:
        st.metric("🚨 Fraud Cases", "1,250")

    with col3:
        st.metric(
            "🎯 AUCPR",
            "0.683"
)

    with col4:
        st.metric("🏥 Hospitals", "350")

    st.divider()

    st.subheader("System Features")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("🔍 Fraud Detection")

    with c2:
        st.info("📊 Real-Time Analytics")

    with c3:
        st.warning("🛡️ Risk Assessment")

    st.divider()

    st.write("""
    This system uses Artificial Intelligence and Machine Learning
    to identify suspicious insurance claims and help insurance
    companies reduce fraud.
    """)

# =========================
# UPLOAD DATASET PAGE
# =========================

elif page == "Upload Dataset":

    st.header("📂 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.session_state.df = df

        st.success("Dataset Uploaded Successfully")

        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])
        st.dataframe(df.head())

# =========================
# FRAUD DETECTION PAGE
# =========================
# =========================
# FRAUD DETECTION PAGE
# =========================

elif page == "Fraud Detection":

    st.header("🔍 Fraud Detection")

    if st.session_state.df is None:

        st.warning("Please upload a dataset first.")

    else:

        df = st.session_state.df

        if st.button("Run Fraud Detection"):

            feature_cols = [
                "InscClaimAmtReimbursed_in",
                "DeductibleAmtPaid_in",
                "InscClaimAmtReimbursed_out",
                "DeductibleAmtPaid_out"
            ]

            X = df[feature_cols]

            dtest = xgb.DMatrix(X)

            xgb_pred = xgb_model.predict(dtest)
            lgb_pred = lgb_model.predict(X)
            rf_pred = rf_model.predict_proba(X)[:, 1]
            mlp_pred = mlp_model.predict_proba(X)[:, 1]

            stack_input = np.column_stack([
                xgb_pred,
                lgb_pred,
                rf_pred,
                mlp_pred
            ])

            final_prob = stacker.predict_proba(
                stack_input
            )[:, 1]

            df["Fraud Probability"] = final_prob

            df["Fraud Prediction"] = (
                final_prob > 0.5
            ).astype(int)

            fraud_count = int(
                df["Fraud Prediction"].sum()
            )

            st.success(
                f"{fraud_count} suspicious claims detected."
            )

            st.dataframe(
                df[
                    [
                        "Provider",
                        "Fraud Probability",
                        "Fraud Prediction"
                    ]
                ].head()
            )
# =========================        
# VISUALIZATIONS PAGE
# =========================
elif page == "Visualizations":

    st.header("📊 Advanced Fraud Analytics")

    if st.session_state.df is None:

        st.warning("Upload dataset first.")

    else:

        df = st.session_state.df

        if "Fraud Prediction" in df.columns:

            fraud_count = int(
                df["Fraud Prediction"].sum()
            )

            genuine_count = (
                len(df) - fraud_count
            )

            total_claims = len(df)

            col1,col2,col3 = st.columns(3)

            with col1:
                st.metric(
                    "🚨 Fraud Claims",
                    fraud_count
                )

            with col2:
                st.metric(
                    "✅ Genuine Claims",
                    genuine_count
                )

            with col3:
                st.metric(
                    "📄 Total Claims",
                    total_claims
                )

            st.divider()

            fraud_df = pd.DataFrame({
                "Type":[
                    "Fraud",
                    "Genuine"
                ],
                "Count":[
                    fraud_count,
                    genuine_count
                ]
            })

            col1,col2 = st.columns(2)

            with col1:

                fig1 = px.pie(
                    fraud_df,
                    names="Type",
                    values="Count",
                    hole=0.6,
                    title="Fraud Distribution"
                )

                st.plotly_chart(
                    fig1,
                    use_container_width=True
                )

            with col2:

                fraud_percent = (
                    fraud_count /
                    total_claims
                ) * 100

                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=fraud_percent,
                        title={
                            "text":
                            "Fraud Risk %"
                        },
                        gauge={
                            "axis":{
                                "range":[0,100]
                            }
                        }
                    )
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

            fig2 = px.bar(
                fraud_df,
                x="Type",
                y="Count",
                color="Type",
                title="Fraud vs Genuine Claims"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            fig3 = px.sunburst(
                fraud_df,
                path=["Type"],
                values="Count",
                title="Claim Analysis"
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

        else:

            st.info(
                "Run Fraud Detection first."
            )
# =========================
# MODEL INSIGHTS PAGE
# =========================

elif page == "Model Insights":

    st.header("📈 Feature Importance Analysis")

    fi = pd.read_csv(
        "saved_models/feature_importance.csv"
    )

    fi.columns = [
    "Feature",
    "Importance_XGB",
    "Importance_LGBM",
    "Importance"

    ]

    st.subheader("Feature Importance Table")
    st.dataframe(fi)

    fig = px.bar(
        fi,
        x="Feature",
        y="Importance_XGB",
        title="XGBoost Feature Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
elif page == "Download Report":

    st.header("📄 Download Report")

    if st.session_state.df is None:

        st.warning(
            "No results available."
        )

    else:

        df = st.session_state.df

        csv = df.to_csv(
            index=False
        )

        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="fraud_results.csv",
            mime="text/csv"
        )

# =========================
# ABOUT PAGE
# =========================

elif page == "About":

    st.header("ℹ️ About")

    st.write("""
    MedGuard AI is an AI-powered Health Insurance
    Fraud Detection System designed to identify
    suspicious claims and improve claim verification.

    Technologies Used:
    - Python
    - Streamlit
    - Pandas
    - Machine Learning
    - XGBoost
    - LightGBM
    - Random Forest
    - Neural Network (MLP)
    - Logistic Regression Stacking
    - Data Visualization
    """)

# =========================
# FOOTER
# =========================

st.divider()

st.markdown(
    """
    <center>
    MedGuard AI © 2026
    </center>
    """,
    unsafe_allow_html=True
)