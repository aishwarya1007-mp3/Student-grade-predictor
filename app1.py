"""
app.py
------
Main Streamlit application for Student Grade Prediction.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ── Page configuration (must be the very first Streamlit call) ────────────────
st.set_page_config(
    page_title="Student Grade Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports ─────────────────────────────────────────────────────────────
from data_loader import (
    load_data, preprocess,
    grade_to_letter, grade_to_percentage,
    FEATURE_DESCRIPTIONS, MODEL_FEATURES,
)
from model import (
    train_models, evaluate_models,
    get_best_model, get_feature_importance,
    predict_grade, save_models, load_saved_models,
)
import utils


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1d2e 0%, #0e1117 100%);
}
/* Metric cards */
div[data-testid="metric-container"] {
    background: #1a1d2e;
    border: 1px solid #2e3250;
    border-radius: 10px;
    padding: 12px 18px;
}
/* Header banner */
.banner {
    background: linear-gradient(135deg, #1f2b6b 0%, #152866 50%, #0e1117 100%);
    padding: 28px 32px;
    border-radius: 14px;
    margin-bottom: 20px;
    border: 1px solid #2e3250;
}
.banner h1 { color: #ffffff; font-size: 2.2rem; margin: 0; }
.banner p  { color: #a0aec0; font-size: 1rem; margin: 6px 0 0; }

/* Grade result card */
.grade-card {
    background: linear-gradient(135deg, #1f2b6b, #0e3d2e);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    border: 1px solid #2e5050;
}
.grade-card h2 { color: #fff; font-size: 1.4rem; }
.grade-card .big { color: #2ECC71; font-size: 3rem; font-weight: bold; }
.grade-card .sub { color: #a0d9c0; font-size: 1.1rem; }

/* Info boxes */
.info-box {
    background: #1a1d2e;
    border-left: 4px solid #4C72B0;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #d0d8f0;
}
</style>
""", unsafe_allow_html=True)


# ── Session state — cache heavy objects ───────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    return load_data()


@st.cache_resource(show_spinner=False)
def get_trained_models():
    """Train models once and cache them for the session."""
    df = get_data()
    X_train, X_test, y_train, y_test, scaler, features = preprocess(df)
    trained = train_models(X_train, y_train)
    results = evaluate_models(trained, X_test, y_test)
    importances = get_feature_importance(trained, features)
    return trained, scaler, features, X_test, y_test, results, importances


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Grade Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home Dashboard",
         "📊 Data Exploration",
         "🤖 Model Evaluation",
         "🎯 Predict Grade",
         "📈 Model Insights"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#6b7280;'>Dataset: UCI Student Performance<br>"
        "Target: Final Grade (G3, 0–20)</small>",
        unsafe_allow_html=True,
    )


# ── Load data & models (runs once) ───────────────────────────────────────────
df = get_data()
trained_models, scaler, feature_names, X_test, y_test, results, importances = (
    get_trained_models()
)
best_model_name = get_best_model(results)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home Dashboard":
    st.markdown("""
    <div class="banner">
      <h1>🎓 Student Grade Prediction System</h1>
      <p>Machine Learning · Python · Scikit-learn · Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Total Students", f"{len(df):,}")
    with col2:
        st.metric("🔢 Features Used", len(feature_names))
    with col3:
        st.metric("📈 Avg Final Grade", f"{df['G3'].mean():.2f} / 20")
    with col4:
        best_r2 = results[best_model_name]["R2 Score"]
        st.metric("🏆 Best R² Score", f"{best_r2:.4f}",
                  help="R² of the best-performing model on the test set")

    st.markdown("---")

    # ── About the project ─────────────────────────────────────────────────────
    col_a, col_b = st.columns([1.4, 1])
    with col_a:
        st.markdown("### 📖 About This Project")
        st.markdown("""
        This application predicts a student's **final mathematics grade (G3)**
        using features such as study habits, parental education, absences,
        and previous period grades.

        **Problem Type:** Regression (predict a numeric grade 0–20)

        **Four models are trained and compared:**
        - Linear Regression
        - Decision Tree Regressor
        - Random Forest Regressor *(typically best)*
        - Gradient Boosting Regressor

        All metrics are computed from real predictions on the held-out **test set**.
        """)

    with col_b:
        st.markdown("### 🗺️ Navigation Guide")
        st.markdown("""
        | Page | What you'll find |
        |---|---|
        | 📊 Data Exploration | Charts, distributions, correlations |
        | 🤖 Model Evaluation | Metrics table & comparison charts |
        | 🎯 Predict Grade | Enter student info → get prediction |
        | 📈 Model Insights | Feature importance & residuals |
        """)

    st.markdown("---")
    st.markdown("### 📌 Dataset Features")
    feat_cols = st.columns(3)
    items = list(FEATURE_DESCRIPTIONS.items())
    chunk = (len(items) + 2) // 3
    for i, col in enumerate(feat_cols):
        for feat, desc in items[i*chunk:(i+1)*chunk]:
            col.markdown(
                f'<div class="info-box"><b>{feat}</b><br><small>{desc}</small></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### 📋 Sample Data Preview")
    st.dataframe(df.head(10), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Exploration":
    st.markdown("## 📊 Data Exploration")
    st.markdown("Visual analysis of the student performance dataset.")

    # Row 1
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Grade Distribution")
        st.pyplot(utils.plot_grade_distribution(df))
    with c2:
        st.markdown("#### Letter Grade Breakdown")
        st.pyplot(utils.plot_letter_grade_pie(df))

    # Row 2
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Study Time vs Final Grade")
        st.pyplot(utils.plot_studytime_vs_grade(df))
    with c4:
        st.markdown("#### Past Failures vs Average Grade")
        st.pyplot(utils.plot_failures_vs_grade(df))

    # Row 3
    c5, c6 = st.columns(2)
    with c5:
        st.markdown("#### Absences vs Final Grade")
        st.pyplot(utils.plot_absences_vs_grade(df))
    with c6:
        st.markdown("#### Grade Distribution by Gender")
        st.pyplot(utils.plot_grade_by_sex(df))

    # Full-width
    st.markdown("#### Grade Progression Across Periods")
    st.pyplot(utils.plot_grade_progression(df))

    st.markdown("#### Correlation Heatmap")
    st.pyplot(utils.plot_correlation_heatmap(df))

    st.markdown("---")
    st.markdown("#### 📋 Dataset Statistics")
    st.dataframe(df.describe().T.round(2), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Evaluation":
    st.markdown("## 🤖 Model Training & Evaluation")
    st.markdown(
        "All models are trained on **80 %** of the data and evaluated on the "
        "held-out **20 % test set**. Metrics are real — not pre-filled."
    )

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.markdown("### 📊 Performance Metrics")
    metrics_df = pd.DataFrame(
        {name: {k: v for k, v in res.items() if k != "y_pred"}
         for name, res in results.items()}
    ).T.rename_axis("Model").reset_index()

    # Highlight the best model row
    def highlight_best(row):
        if row["Model"] == best_model_name:
            return ["background-color: #1a3a2e; color: #2ECC71"] * len(row)
        return [""] * len(row)

    st.dataframe(
        metrics_df.style.apply(highlight_best, axis=1).format(
            {"MAE": "{:.4f}", "MSE": "{:.4f}", "RMSE": "{:.4f}", "R2 Score": "{:.4f}"}
        ),
        use_container_width=True,
    )
    st.success(f"🏆 Best model: **{best_model_name}** "
               f"(R² = {results[best_model_name]['R2 Score']:.4f})")

    st.markdown("---")
    st.markdown("### 📈 Visual Comparison")

    metric_choice = st.selectbox(
        "Select metric to compare",
        ["R2 Score", "MAE", "RMSE", "MSE"],
        index=0,
    )
    st.pyplot(utils.plot_model_comparison(results, metric=metric_choice))

    st.markdown("---")
    st.markdown("### 🔍 Metric Explanations")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        '<div class="info-box"><b>MAE</b><br>Mean Absolute Error — '
        'average absolute difference between predicted and actual grades. '
        'Lower is better.</div>', unsafe_allow_html=True)
    col2.markdown(
        '<div class="info-box"><b>MSE</b><br>Mean Squared Error — '
        'penalises large errors more heavily. Lower is better.</div>',
        unsafe_allow_html=True)
    col3.markdown(
        '<div class="info-box"><b>RMSE</b><br>Root Mean Squared Error — '
        'same units as the target grade. Lower is better.</div>',
        unsafe_allow_html=True)
    col4.markdown(
        '<div class="info-box"><b>R² Score</b><br>Coefficient of determination — '
        'proportion of variance explained. Closer to 1 is better.</div>',
        unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT GRADE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict Grade":
    st.markdown("## 🎯 Predict a Student's Final Grade")
    st.markdown(
        "Fill in the student's details below and click **Predict** to get "
        "the estimated final grade."
    )

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.markdown("### 📝 Student Information")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Academic**")
            studytime = st.selectbox(
                "Weekly Study Time",
                options=[1, 2, 3, 4],
                format_func=lambda x: {1:"<2 hrs", 2:"2–5 hrs",
                                        3:"5–10 hrs", 4:">10 hrs"}[x],
                index=1,
            )
            failures = st.selectbox("Past Class Failures", [0, 1, 2, 3], index=0)
            absences = st.slider("School Absences", 0, 75, 5)
            G1 = st.slider("Period 1 Grade (G1)", 0, 20, 10)
            G2 = st.slider("Period 2 Grade (G2)", 0, 20, 11)

        with c2:
            st.markdown("**Family**")
            Medu = st.selectbox(
                "Mother's Education",
                [0, 1, 2, 3, 4],
                format_func=lambda x: {0:"None",1:"Primary",2:"5th–9th grade",
                                        3:"Secondary",4:"Higher education"}[x],
                index=2,
            )
            Fedu = st.selectbox(
                "Father's Education",
                [0, 1, 2, 3, 4],
                format_func=lambda x: {0:"None",1:"Primary",2:"5th–9th grade",
                                        3:"Secondary",4:"Higher education"}[x],
                index=2,
            )
            famrel = st.slider("Family Relationship Quality (1–5)", 1, 5, 4)
            higher = st.selectbox("Wants Higher Education?", ["yes","no"], index=0)
            schoolsup = st.selectbox("Extra School Support?", ["yes","no"], index=1)

        with c3:
            st.markdown("**Lifestyle**")
            goout = st.slider("Going Out with Friends (1–5)", 1, 5, 3)
            health = st.slider("Health Status (1–5)", 1, 5, 3)
            freetime = st.slider("Free Time After School (1–5)", 1, 5, 3)
            Dalc = st.slider("Workday Alcohol (1–5)", 1, 5, 1)
            Walc = st.slider("Weekend Alcohol (1–5)", 1, 5, 1)
            internet = st.selectbox("Internet Access at Home?", ["yes","no"], index=0)
            romantic = st.selectbox("In a Romantic Relationship?", ["yes","no"], index=1)
            age = st.slider("Student Age", 15, 22, 17)

        st.markdown("---")
        model_choice = st.selectbox(
            "Choose Model for Prediction",
            list(trained_models.keys()),
            index=list(trained_models.keys()).index(best_model_name),
        )
        submitted = st.form_submit_button("🔮 Predict Grade", use_container_width=True)

    # ── Prediction result ─────────────────────────────────────────────────────
    if submitted:
        # Build input vector (same order as MODEL_FEATURES)
        input_map = {
            "studytime": studytime, "failures": failures, "absences": absences,
            "G1": G1, "G2": G2, "Medu": Medu, "Fedu": Fedu,
            "goout": goout, "health": health, "famrel": famrel,
            "freetime": freetime, "Dalc": Dalc, "Walc": Walc,
            "internet": 1 if internet == "yes" else 0,
            "schoolsup": 1 if schoolsup == "yes" else 0,
            "higher": 1 if higher == "yes" else 0,
            "romantic": 1 if romantic == "yes" else 0,
            "age": age,
        }
        input_values = [input_map[f] for f in feature_names]

        model = trained_models[model_choice]
        predicted_grade = predict_grade(model, scaler, input_values)
        letter = grade_to_letter(predicted_grade)
        pct    = grade_to_percentage(predicted_grade)

        # Colour by grade
        letter_colors = {"A":"#2ECC71","B":"#3498DB","C":"#F39C12",
                         "D":"#E67E22","F":"#E74C3C"}
        color = letter_colors.get(letter, "#FFFFFF")

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                f'<div class="grade-card">'
                f'<h2>Predicted Grade</h2>'
                f'<div class="big" style="color:{color}">{predicted_grade:.1f}</div>'
                f'<div class="sub">out of 20</div>'
                f'</div>', unsafe_allow_html=True
            )
        with r2:
            st.markdown(
                f'<div class="grade-card">'
                f'<h2>Letter Grade</h2>'
                f'<div class="big" style="color:{color}">{letter}</div>'
                f'<div class="sub">Grade Band</div>'
                f'</div>', unsafe_allow_html=True
            )
        with r3:
            st.markdown(
                f'<div class="grade-card">'
                f'<h2>Percentage</h2>'
                f'<div class="big" style="color:{color}">{pct}%</div>'
                f'<div class="sub">Equivalent</div>'
                f'</div>', unsafe_allow_html=True
            )

        # Progress bar (Streamlit expects a float 0.0–1.0)
        st.markdown("#### Grade Progress")
        st.progress(min(1.0, max(0.0, predicted_grade / 20.0)))

        # Grade band explanation
        band_info = {
            "A": "Excellent! The student is performing at the highest level.",
            "B": "Good performance. The student is above average.",
            "C": "Average performance. There is room for improvement.",
            "D": "Below average. The student may need additional support.",
            "F": "Failing. Immediate intervention is recommended.",
        }
        st.info(f"**{letter} Grade** — {band_info[letter]}")

        # Model used
        model_r2 = results[model_choice]["R2 Score"]
        st.caption(
            f"Model used: **{model_choice}** | R² Score: {model_r2:.4f} | "
            f"MAE: {results[model_choice]['MAE']:.4f}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Insights":
    st.markdown("## 📈 Model Insights")

    model_choice = st.selectbox(
        "Select a model to inspect",
        list(trained_models.keys()),
        index=list(trained_models.keys()).index(best_model_name),
    )

    y_pred = results[model_choice]["y_pred"]

    # Actual vs predicted
    st.markdown("### 🔵 Actual vs Predicted Grades")
    st.pyplot(utils.plot_actual_vs_predicted(y_test, y_pred, model_choice))

    # Residuals
    st.markdown("### 📉 Residual Plot")
    st.pyplot(utils.plot_residuals(y_test, y_pred, model_choice))

    # Feature importance
    if model_choice in importances:
        st.markdown("### 🌟 Feature Importance")
        st.pyplot(utils.plot_feature_importance(importances, model_choice))
    else:
        st.info("Feature importance is not available for this model.")

    # Error distribution
    st.markdown("### 📊 Prediction Error Distribution")
    errors = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(errors, bins=25, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="#C44E52", linewidth=2, linestyle="--", label="Zero error")
    ax.set_xlabel("Prediction Error (Actual − Predicted)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Prediction Errors", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1A1D23", labelcolor="#FAFAFA")
    ax.set_facecolor("#1A1D23")
    fig.patch.set_facecolor("#0E1117")
    ax.tick_params(colors="#FAFAFA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3A3D47")
    ax.xaxis.label.set_color("#FAFAFA")
    ax.yaxis.label.set_color("#FAFAFA")
    ax.title.set_color("#FAFAFA")
    plt.tight_layout()
    st.pyplot(fig)

    # Summary stats
    st.markdown("### 📋 Quick Stats")
    ec1, ec2, ec3, ec4 = st.columns(4)
    ec1.metric("Mean Error", f"{errors.mean():.4f}")
    ec2.metric("Std of Error", f"{errors.std():.4f}")
    ec3.metric("Max Overestimate", f"{errors.min():.2f}")
    ec4.metric("Max Underestimate", f"{errors.max():.2f}")
