"""
King Kohli Boundary Predictor - Interactive Streamlit Web Application.
Simulates live ball-by-ball boundary likelihood using the trained Hybrid Probability Ensemble.
"""

import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier

# Page Configuration
st.set_page_config(
    page_title="King Kohli Boundary Predictor | IPL AI",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .prediction-box-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(249, 115, 22, 0.15) 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        animation: pulse 2s infinite;
    }
    .prediction-box-low {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.2) 0%, rgba(14, 116, 144, 0.2) 100%);
        border: 2px solid #0284c7;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }
    .verdict-title {
        font-size: 1.7rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """Loads pre-trained models, preprocessor, and ensemble configuration from .pkl files."""
    model_dir = "models"
    
    # 1. Load preprocessor (.pkl)
    prep_path = os.path.join(model_dir, "preprocessor.pkl")
    if not os.path.exists(prep_path):
        prep_path = os.path.join(model_dir, "preprocessor.joblib")
    preprocessor = joblib.load(prep_path)

    # 2. Load Logistic Regression model (.pkl)
    log_path = os.path.join(model_dir, "logistic_model.pkl")
    if not os.path.exists(log_path):
        log_path = os.path.join(model_dir, "logistic_model.joblib")
    log_model = joblib.load(log_path)

    # 3. Load CatBoost model (.pkl fallback to .cbm)
    cat_path = os.path.join(model_dir, "catboost_model.pkl")
    if os.path.exists(cat_path):
        with open(cat_path, "rb") as f:
            cat_model = pickle.load(f)
    else:
        cat_model = CatBoostClassifier()
        cat_model.load_model(os.path.join(model_dir, "catboost_model.cbm"))

    # 4. Load configuration
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)

    return preprocessor, log_model, cat_model, config


try:
    preprocessor, log_model, cat_model, config = load_model_artifacts()
    models_ready = True
except Exception as e:
    models_ready = False
    st.error(f"Models are currently preparing or not found: {e}. Please run model exporter.")


# Top Header
st.markdown('<div class="main-title">👑 King Kohli IPL Boundary Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Live Ball-by-Ball Predictive Intelligence powered by a 76.5% Accurate Hybrid Probability Ensemble (BorderlineSMOTE + CatBoost)</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🎮 Live Delivery Simulator", "📊 Model Performance & Charts", "📖 Research Journey & Insights"])

with tab1:
    col_input, col_pred = st.columns([1.1, 1], gap="large")

    with col_input:
        st.subheader("⚙️ Match & Batter Situation")

        # Preset Selector
        preset = st.selectbox(
            "⚡ Quick-Load Match Scenarios:",
            [
                "Custom Scenario",
                "🔥 2016 Peak Form: Powerplay Assault",
                "⏳ Middle Overs Pressure: Spin Strangulation",
                "💥 Death Over Crunch: 19th Over Need Boundary",
                "🛡️ Early Anchor Mode: Opening Overs Rebuild"
            ]
        )

        # Default values based on preset
        if preset == "🔥 2016 Peak Form: Powerplay Assault":
            default_over, default_ball, default_sr = 4, 3, 175.0
            default_r5, default_r10, default_prev = 14, 26, 4
            default_drought, default_dots, default_bowler = 2, 1, "pace"
            default_pos = 3
        elif preset == "⏳ Middle Overs Pressure: Spin Strangulation":
            default_over, default_ball, default_sr = 12, 4, 118.0
            default_r5, default_r10, default_prev = 3, 7, 0
            default_drought, default_dots, default_bowler = 14, 4, "spin"
            default_pos = 3
        elif preset == "💥 Death Over Crunch: 19th Over Need Boundary":
            default_over, default_ball, default_sr = 19, 5, 160.0
            default_r5, default_r10, default_prev = 11, 22, 6
            default_drought, default_dots, default_bowler = 1, 1, "pace"
            default_pos = 3
        elif preset == "🛡️ Early Anchor Mode: Opening Overs Rebuild":
            default_over, default_ball, default_sr = 2, 2, 75.0
            default_r5, default_r10, default_prev = 2, 2, 1
            default_drought, default_dots, default_bowler = 6, 3, "pace"
            default_pos = 3
        else:
            default_over, default_ball, default_sr = 14, 3, 135.0
            default_r5, default_r10, default_prev = 8, 16, 1
            default_drought, default_dots, default_bowler = 5, 2, "pace"
            default_pos = 3

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            over = st.slider("Over Number", min_value=1, max_value=20, value=default_over)
            ball = st.slider("Ball of the Over", min_value=1, max_value=6, value=default_ball)
            bat_pos = st.selectbox("Batting Position", options=[1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5].index(default_pos))

            # Determine over phase
            if over <= 6:
                over_phase = "Powerplay"
            elif over <= 15:
                over_phase = "Middle"
            else:
                over_phase = "Death"
            st.info(f"🏟️ Game Phase: **{over_phase}**")

        with col_m2:
            bowler_type = st.radio("Bowler Type", options=["pace", "spin"], index=0 if default_bowler == "pace" else 1, horizontal=True)
            current_sr = st.number_input("Progressive Strike Rate (%)", min_value=0.0, max_value=400.0, value=default_sr, step=5.0)
            prev_ball_runs = st.selectbox("Runs Scored on Previous Ball", options=[0, 1, 2, 3, 4, 6], index=[0, 1, 2, 3, 4, 6].index(default_prev))

        st.markdown("---")
        st.subheader("🔥 Momentum & Psychological Pressure")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            runs_last_5 = st.slider("Runs in Last 5 Balls", min_value=0, max_value=30, value=default_r5)
            runs_last_10 = st.slider("Runs in Last 10 Balls", min_value=0, max_value=60, value=default_r10)
        with col_p2:
            balls_since_boundary = st.slider("Deliveries Since Last Boundary", min_value=-1, max_value=30, value=default_drought, help="-1 indicates no boundary hit yet in the innings")
            dotballs_last_5 = st.slider("Dot Balls Faced in Last 5", min_value=0, max_value=5, value=default_dots)
            boundaries_last_5 = st.slider("Boundaries in Last 5 Balls", min_value=0, max_value=5, value=1 if runs_last_5 >= 8 else 0)

    # Prediction Engine
    with col_pred:
        st.subheader("⚡ Live Model Inference")

        if models_ready:
            # Construct DataFrame row
            input_dict = {
                "over": over,
                "ball": ball,
                "bat_pos": bat_pos,
                "bowler": bowler_type,
                "current_sr": current_sr,
                "runs_last_5": runs_last_5,
                "runs_last_10": runs_last_10,
                "prev_ball_runs": prev_ball_runs,
                "boundaries_last_5": boundaries_last_5,
                "dotballs_last_5": dotballs_last_5,
                "balls_since_boundary": balls_since_boundary,
                "over_phase": over_phase
            }
            input_df = pd.DataFrame([input_dict])

            # Preprocess for Logistic component
            input_proc = preprocessor.transform(input_df)

            # Predict Probabilities
            prob_log = float(log_model.predict_proba(input_proc)[:, 1][0])
            prob_cat = float(cat_model.predict_proba(input_df)[:, 1][0])

            # Blend
            w_log = config.get("logistic_weight", 0.30)
            w_cat = config.get("catboost_weight", 0.70)
            threshold = config.get("decision_threshold", 0.550)

            prob_ensemble = (w_log * prob_log) + (w_cat * prob_cat)
            is_boundary = prob_ensemble >= threshold

            # Visual Result Box
            if is_boundary:
                st.markdown(f"""
                <div class="prediction-box-high">
                    <div style="font-size: 2.8rem; margin-bottom: 5px;">🚨 💥 🏏</div>
                    <div class="verdict-title" style="color: #f97316;">HIGH BOUNDARY RISK!</div>
                    <p style="color: #fed7aa; font-size: 1.1rem; margin-bottom: 15px;">
                        The model predicts a <strong>FOUR or SIX</strong> on this delivery!
                    </p>
                    <div style="font-size: 3rem; font-weight: 800; color: #fff;">
                        {prob_ensemble * 100:.1f}%
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 1px;">
                        Boundary Probability (Threshold: {threshold * 100:.1f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box-low">
                    <div style="font-size: 2.8rem; margin-bottom: 5px;">🛡️ 🟢 ⚾</div>
                    <div class="verdict-title" style="color: #38bdf8;">NON-BOUNDARY LIKELY</div>
                    <p style="color: #bae6fd; font-size: 1.1rem; margin-bottom: 15px;">
                        Expected outcome: <strong>Dot Ball, Single, or Double</strong>.
                    </p>
                    <div style="font-size: 3rem; font-weight: 800; color: #fff;">
                        {prob_ensemble * 100:.1f}%
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 1px;">
                        Boundary Probability (Threshold: {threshold * 100:.1f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Component Breakdown Cards
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Ensemble Blended</div>
                    <div class="metric-value" style="color: #f59e0b;">{prob_ensemble*100:.1f}%</div>
                    <div style="font-size: 0.75rem; color: #64748b;">30% Log + 70% Cat</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">CatBoost Tree</div>
                    <div class="metric-value" style="color: #06b6d4;">{prob_cat*100:.1f}%</div>
                    <div style="font-size: 0.75rem; color: #64748b;">Weight: 70%</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Borderline Logistic</div>
                    <div class="metric-value" style="color: #8b5cf6;">{prob_log*100:.1f}%</div>
                    <div style="font-size: 0.75rem; color: #64748b;">Weight: 30%</div>
                </div>
                """, unsafe_allow_html=True)

            # Contextual Insight Box
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🔍 Strategic Model Reasoning", expanded=True):
                insights = []
                if balls_since_boundary >= 10:
                    insights.append(f"⏱️ **High Boundary Drought ({balls_since_boundary} balls)**: Boundary pressure is heavily compounded; historic data shows Kohli actively manufactures boundary attempts after 8+ balls without a four/six.")
                if current_sr >= 150:
                    insights.append(f"🚀 **Accelerated Strike Rate ({current_sr:.1f}%)**: Indicates Kohli is in high gear and looking to clear the infield.")
                if dotballs_last_5 >= 3:
                    insights.append(f"⚠️ **Dot Ball Accumulation ({dotballs_last_5}/5 dots)**: bowler is applying pressure, triggering aggressive counter-attacking intent.")
                if over_phase == "Death":
                    insights.append("💥 **Death Overs (16-20)**: Baseline boundary intent is 2.4x higher than middle overs.")
                if not insights:
                    insights.append("📊 Steady state: Moderate risk with balanced accumulation. Typical anchor phase.")
                
                for ins in insights:
                    st.write(ins)

with tab2:
    st.subheader("📈 Model Progression & Benchmark Tournament")
    st.write("Visual evidence documenting how the hybrid ensemble achieved **76.51% accuracy** and **52.17% recall** despite the 6:1 class imbalance.")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if os.path.exists("charts/01_class_imbalance.png"):
            st.image("charts/01_class_imbalance.png", caption="The Harsh 6:1 Class Imbalance in Kohli's First Innings Balls", width="stretch")
        if os.path.exists("charts/02_model_evolution.png"):
            st.image("charts/02_model_evolution.png", caption="F1-Score Progression Across 8 Models (+77.2% Gain)", width="stretch")

    with col_c2:
        if os.path.exists("charts/04_ensemble_confusion_matrix.png"):
            st.image("charts/04_ensemble_confusion_matrix.png", caption="Final Ensemble Confusion Matrix (52.2% Recall on Unseen Test Balls)", width="stretch")
        if os.path.exists("charts/05_catboost_feature_importance.png"):
            st.image("charts/05_catboost_feature_importance.png", caption="What Predicts a Kohli Boundary? Top CatBoost Feature Importances", width="stretch")

with tab3:
    st.subheader("📚 The Machine Learning Engineering Story")
    st.markdown("""
    ### Why this is a True Machine Learning Project:
    1. **Domain Feature Engineering**: Rather than relying purely on static inputs (`over`, `ball`), we designed `create_dataset()`, tracking dynamic variables like **strike rate acceleration**, **dot ball pressure**, and **boundary drought intervals**.
    2. **Tackling Imbalance Beyond Naive Accuracy**:
       * A dummy model predicting 0 gets 85.3% accuracy, but has **0% recall**.
       * Our ensemble catches **52.17% of all boundaries** with **76.51% overall accuracy** and an F1 of **0.3934**.
    3. **Ensemble Inductive Diversity**:
       * Linear models (Logistic with BorderlineSMOTE) find clean linear separation planes.
       * Gradient Boosted Trees (CatBoost) master complex non-linear phase interactions.
       * Linear probability blending ($0.30 P_{log} + 0.70 P_{cat}$ at threshold $0.550$) yielded our peak performance.
    """)
