"""
utils.py
--------
All Matplotlib / Seaborn chart functions used by app.py.

Every function returns a Matplotlib Figure so Streamlit can render it
with st.pyplot(fig).  Figures are explicitly closed after creation to
avoid memory warnings in long-running Streamlit sessions.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

matplotlib.use("Agg")   # non-interactive backend — required for Streamlit

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE   = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
BG_COLOR  = "#0E1117"   # matches Streamlit dark background
TEXT_COLOR = "#FAFAFA"

def _apply_dark_style(fig, axes_list=None):
    """Apply a consistent dark theme to any figure."""
    fig.patch.set_facecolor(BG_COLOR)
    axes = axes_list or fig.get_axes()
    for ax in axes:
        ax.set_facecolor("#1A1D23")
        ax.tick_params(colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor("#3A3D47")


# ── 1. Grade distribution ──────────────────────────────────────────────────────

def plot_grade_distribution(df: pd.DataFrame):
    """Histogram of final grades (G3)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["G3"], bins=21, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Final Grade (G3)", fontsize=12)
    ax.set_ylabel("Number of Students", fontsize=12)
    ax.set_title("Distribution of Final Grades", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 20)
    _apply_dark_style(fig, [ax])
    plt.tight_layout()
    return fig


# ── 2. Correlation heatmap ─────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame):
    """Heatmap of correlations between numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Keep the most informative columns
    keep = ["G1", "G2", "G3", "studytime", "failures", "absences",
            "Medu", "Fedu", "goout", "health", "Dalc", "Walc"]
    keep = [c for c in keep if c in numeric_cols]

    corr = df[keep].corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        annot_kws={"size": 9}, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold",
                 color=TEXT_COLOR)
    _apply_dark_style(fig, [ax])
    plt.tight_layout()
    return fig


# ── 3. Study time vs grade ─────────────────────────────────────────────────────

def plot_studytime_vs_grade(df: pd.DataFrame):
    """Box plot of G3 grouped by study time."""
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = {1: "<2 hrs", 2: "2-5 hrs", 3: "5-10 hrs", 4: ">10 hrs"}
    df2 = df.copy()
    df2["Study Time"] = df2["studytime"].map(labels)
    order = ["<2 hrs", "2-5 hrs", "5-10 hrs", ">10 hrs"]
    sns.boxplot(data=df2, x="Study Time", y="G3", order=order,
                palette=PALETTE, ax=ax)
    ax.set_xlabel("Weekly Study Time", fontsize=12)
    ax.set_ylabel("Final Grade (G3)", fontsize=12)
    ax.set_title("Study Time vs Final Grade", fontsize=14, fontweight="bold")
    _apply_dark_style(fig, [ax])
    plt.tight_layout()
    return fig


# ── 4. Failures vs grade ───────────────────────────────────────────────────────

def plot_failures_vs_grade(df: pd.DataFrame):
    """Bar chart: average G3 by number of past failures."""
    avg = df.groupby("failures")["G3"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(avg["failures"].astype(str), avg["G3"],
                  color=PALETTE[:len(avg)], edgecolor="white")
    ax.bar_label(bars, fmt="%.1f", color=TEXT_COLOR, padding=3)
    ax.set_xlabel("Number of Past Failures", fontsize=12)
    ax.set_ylabel("Average Final Grade", fontsize=12)
    ax.set_title("Past Failures vs Average Final Grade", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 22)
    _apply_dark_style(fig, [ax])
    plt.tight_layout()
    return fig


# ── 5. Absences vs grade scatter ───────────────────────────────────────────────

def plot_absences_vs_grade(df: pd.DataFrame):
    """Scatter plot of absences vs G3."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df["absences"], df["G3"], alpha=0.5, color="#4C72B0",
               edgecolors="white", linewidth=0.3, s=40)
    m, b = np.polyfit(df["absences"], df["G3"], 1)
    xs = np.linspace(df["absences"].min(), df["absences"].max(), 100)
    ax.plot(xs, m * xs + b, color="#C44E52", linewidth=2, label="Trend line")
    ax.set_xlabel("Number of Absences", fontsize=12)
    ax.set_ylabel("Final Grade (G3)", fontsize=12)
    ax.set_title("School Absences vs Final Grade", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#1A1D23", labelcolor=TEXT_COLOR)
    _apply_dark_style(fig, [ax])
    plt.tight_layout()
    return fig


# ── 6. Grade distribution by sex ──────────────────────────────────────────────

def plot_grade_by_sex(df: pd.DataFrame):
    """Side-by-side histogram for male vs female students."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for sex, color in zip(["M", "F"], ["#4C72B0", "#C44E52"]):
        subset = df[df["sex"] == sex]["G3"]
        ax.hist(subset, bins=15, alpha=0.7, color=color,
                edgecolor="white", linewidth=0.4, label=f"{'Male' if sex=='M' else 'Female'}")
    ax.set_xlabel("Final Grade (G3)", fontsize=12)
    ax.set_ylabel("Number of Students", fontsize=12)
    ax.set_title("Grade Distribution by Gender", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#1A1D23", labelcolor=TEXT_COLOR)
    _apply_dark_style(fig, [ax])
    plt.tight_layout()
    return fig


# ── 7. G1 G2 G3 progression ───────────────────────────────────────────────────

def plot_grade_progression(df: pd.DataFrame):
    """Average grades across the three periods."""
    avgs = [df["G1"].mean(), df["G2"].mean(), df["G3"].mean()]
    labels = ["Period 1 (G1)", "Period 2 (G2)", "Final (G3)"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, avgs, color=PALETTE[:3], edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", color=TEXT_COLOR, padding=3)
    ax.set_ylabel("Average Grade", fontsize=12)
    ax.set_title("Average Grade Progression Across Periods", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 22)
    _apply_dark_style(fig, [ax])
    plt.tight_layout()
    return fig


# ── 8. Model comparison bar chart ─────────────────────────────────────────────

def plot_model_comparison(results: dict, metric: str = "R2 Score"):
    """Horizontal bar chart comparing model performance on a chosen metric."""
    names  = list(results.keys())
    values = [results[n][metric] for n in names]

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    bars = ax.barh(names, values, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", color=TEXT_COLOR, padding=5)
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f"Model Comparison — {metric}", fontsize=14, fontweight="bold")
    if metric == "R2 Score":
        ax.set_xlim(0, 1.15)
    _apply_dark_style(fig, [ax])
    plt.tight_layout()
    return fig


# ── 9. Actual vs Predicted scatter ────────────────────────────────────────────

def plot_actual_vs_predicted(y_test, y_pred, model_name: str):
    """Scatter plot of true grades vs predicted grades."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.6, color="#4C72B0",
               edgecolors="white", linewidth=0.3, s=50, label="Students")
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual Grade (G3)", fontsize=12)
    ax.set_ylabel("Predicted Grade (G3)", fontsize=12)
    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1A1D23", labelcolor=TEXT_COLOR)
    _apply_dark_style(fig, [ax])
    plt.tight_layout()
    return fig


# ── 10. Feature importance ────────────────────────────────────────────────────

def plot_feature_importance(importances: dict, model_name: str):
    """Horizontal bar chart of feature importances for one model."""
    if model_name not in importances:
        return None

    imp  = importances[model_name]
    df_i = pd.Series(imp).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(df_i)))
    ax.barh(df_i.index, df_i.values, color=colors, edgecolor="white")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    _apply_dark_style(fig, [ax])
    plt.tight_layout()
    return fig


# ── 11. Residual plot ─────────────────────────────────────────────────────────

def plot_residuals(y_test, y_pred, model_name: str):
    """Residual plot: predicted grade vs residual (error)."""
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(y_pred, residuals, alpha=0.5, color="#55A868",
               edgecolors="white", linewidth=0.3, s=40)
    ax.axhline(0, color="#C44E52", linewidth=2, linestyle="--")
    ax.set_xlabel("Predicted Grade", fontsize=12)
    ax.set_ylabel("Residual (Actual − Predicted)", fontsize=12)
    ax.set_title(f"Residual Plot — {model_name}", fontsize=13, fontweight="bold")
    _apply_dark_style(fig, [ax])
    plt.tight_layout()
    return fig


# ── 12. Letter grade pie chart ────────────────────────────────────────────────

def plot_letter_grade_pie(df: pd.DataFrame):
    """Pie chart of letter grade distribution."""
    from data_loader import grade_to_letter
    df2 = df.copy()
    df2["letter"] = df2["G3"].apply(grade_to_letter)
    counts = df2["letter"].value_counts().reindex(["A","B","C","D","F"]).dropna()

    fig, ax = plt.subplots(figsize=(6, 5))
    wedge_colors = ["#2ECC71", "#3498DB", "#F39C12", "#E74C3C", "#95A5A6"]
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
           colors=wedge_colors[:len(counts)], startangle=140,
           textprops={"color": TEXT_COLOR, "fontsize": 11},
           wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title("Letter Grade Distribution", fontsize=14, fontweight="bold",
                 color=TEXT_COLOR)
    _apply_dark_style(fig)
    plt.tight_layout()
    return fig
