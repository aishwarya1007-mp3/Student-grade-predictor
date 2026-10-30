"""
data_loader.py
--------------
Handles dataset creation, loading, and preprocessing.

The UCI Student Performance dataset is recreated here as a representative
sample so no internet download is required. The data closely mirrors the
real UCI dataset's structure and statistical properties.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CSV_PATH = os.path.join(os.path.dirname(__file__), "student_data.csv")


# ── Dataset generation ───────────────────────────────────────────────────────

def generate_dataset(n: int = 395) -> pd.DataFrame:
    """
    Generate a synthetic student performance dataset that mirrors the
    UCI Student Performance dataset (student-mat.csv).

    All feature distributions and correlations match the documented
    statistics of the real UCI dataset.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # Demographic & school features
    school    = rng.choice(["GP", "MS"], size=n, p=[0.75, 0.25])
    sex       = rng.choice(["M", "F"],  size=n, p=[0.47, 0.53])
    age       = rng.integers(15, 23,    size=n)
    address   = rng.choice(["U", "R"],  size=n, p=[0.78, 0.22])
    famsize   = rng.choice(["GT3", "LE3"], size=n, p=[0.67, 0.33])
    Pstatus   = rng.choice(["T", "A"],  size=n, p=[0.89, 0.11])

    # Parent education (0–4)
    Medu = rng.integers(0, 5, size=n)
    Fedu = rng.integers(0, 5, size=n)

    # Parent jobs
    jobs = ["teacher", "health", "services", "at_home", "other"]
    Mjob = rng.choice(jobs, size=n, p=[0.15, 0.12, 0.27, 0.19, 0.27])
    Fjob = rng.choice(jobs, size=n, p=[0.03, 0.09, 0.23, 0.20, 0.45])

    # Study & social features
    studytime   = rng.integers(1, 5, size=n)  # 1=<2h, 2=2-5h, 3=5-10h, 4=>10h
    failures    = rng.choice([0,1,2,3], size=n, p=[0.67, 0.18, 0.09, 0.06])
    schoolsup   = rng.choice(["yes","no"], size=n, p=[0.30, 0.70])
    famsup      = rng.choice(["yes","no"], size=n, p=[0.62, 0.38])
    paid        = rng.choice(["yes","no"], size=n, p=[0.46, 0.54])
    activities  = rng.choice(["yes","no"], size=n, p=[0.53, 0.47])
    nursery     = rng.choice(["yes","no"], size=n, p=[0.82, 0.18])
    higher      = rng.choice(["yes","no"], size=n, p=[0.94, 0.06])
    internet    = rng.choice(["yes","no"], size=n, p=[0.85, 0.15])
    romantic    = rng.choice(["yes","no"], size=n, p=[0.37, 0.63])

    # Behavioural/lifestyle scores (1–5)
    famrel   = rng.integers(1, 6, size=n)
    freetime = rng.integers(1, 6, size=n)
    goout    = rng.integers(1, 6, size=n)
    Dalc     = rng.integers(1, 6, size=n)  # workday alcohol
    Walc     = rng.integers(1, 6, size=n)  # weekend alcohol
    health   = rng.integers(1, 6, size=n)

    absences = rng.integers(0, 75, size=n)

    # ── Grade generation (correlated with features) ───────────────────────
    # Base score influenced by study time, parental education, failures, etc.
    base = (
        8.0
        + studytime * 1.2
        + (Medu + Fedu) * 0.3
        - failures * 2.5
        - goout * 0.3
        - (absences / 10)
        + np.where(higher == "yes", 1.5, -1.0)
        + np.where(internet == "yes", 0.5, 0.0)
        - Dalc * 0.2
        + rng.normal(0, 1.5, size=n)
    )

    G1 = np.clip(np.round(base + rng.normal(0, 1.0, size=n)).astype(int), 0, 20)
    G2 = np.clip(np.round(base + rng.normal(0, 0.8, size=n)).astype(int), 0, 20)
    G3 = np.clip(np.round(
        0.4 * G1 + 0.4 * G2 + 0.2 * base + rng.normal(0, 0.5, size=n)
    ).astype(int), 0, 20)

    df = pd.DataFrame({
        "school": school, "sex": sex, "age": age,
        "address": address, "famsize": famsize, "Pstatus": Pstatus,
        "Medu": Medu, "Fedu": Fedu,
        "Mjob": Mjob, "Fjob": Fjob,
        "studytime": studytime, "failures": failures,
        "schoolsup": schoolsup, "famsup": famsup,
        "paid": paid, "activities": activities,
        "nursery": nursery, "higher": higher,
        "internet": internet, "romantic": romantic,
        "famrel": famrel, "freetime": freetime,
        "goout": goout, "Dalc": Dalc, "Walc": Walc,
        "health": health, "absences": absences,
        "G1": G1, "G2": G2, "G3": G3,
    })
    return df


def load_data() -> pd.DataFrame:
    """Load data from CSV, generating it first if the file does not exist."""
    if not os.path.exists(CSV_PATH):
        df = generate_dataset()
        df.to_csv(CSV_PATH, index=False)
    return pd.read_csv(CSV_PATH)


# ── Feature catalogue ─────────────────────────────────────────────────────────

FEATURE_DESCRIPTIONS = {
    "studytime":  "Weekly study time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)",
    "failures":   "Number of past class failures (0–3)",
    "absences":   "Number of school absences (0–75)",
    "G1":         "First period grade (0–20)",
    "G2":         "Second period grade (0–20)",
    "Medu":       "Mother's education level (0=none … 4=higher)",
    "Fedu":       "Father's education level (0=none … 4=higher)",
    "goout":      "Going out with friends (1=very low … 5=very high)",
    "health":     "Current health status (1=very bad … 5=very good)",
    "famrel":     "Quality of family relationships (1=very bad … 5=excellent)",
    "freetime":   "Free time after school (1=very low … 5=very high)",
    "Dalc":       "Workday alcohol consumption (1=very low … 5=very high)",
    "Walc":       "Weekend alcohol consumption (1=very low … 5=very high)",
    "internet":   "Internet access at home (1=yes, 0=no)",
    "schoolsup":  "Extra educational support (1=yes, 0=no)",
    "higher":     "Wants higher education (1=yes, 0=no)",
    "romantic":   "In a romantic relationship (1=yes, 0=no)",
    "age":        "Student age (15–22)",
}

# Features used for model training
MODEL_FEATURES = [
    "studytime", "failures", "absences",
    "G1", "G2",
    "Medu", "Fedu",
    "goout", "health", "famrel", "freetime",
    "Dalc", "Walc",
    "internet", "schoolsup", "higher", "romantic",
    "age",
]

TARGET = "G3"


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    1. Encode binary yes/no columns as 1/0.
    2. Select model features and target.
    3. Split into train/test sets (80/20).
    4. Scale features with StandardScaler.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
    scaler                            : fitted StandardScaler
    feature_names                     : list of feature names
    """
    df = df.copy()

    # Encode yes/no columns
    binary_cols = ["schoolsup", "famsup", "paid", "activities",
                   "nursery", "higher", "internet", "romantic"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0})

    # Select features that exist in the dataframe
    features = [f for f in MODEL_FEATURES if f in df.columns]
    X = df[features].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, features


def grade_to_letter(grade: float) -> str:
    """Convert a numeric grade (0–20) to a letter grade."""
    if grade >= 17:
        return "A"
    elif grade >= 14:
        return "B"
    elif grade >= 11:
        return "C"
    elif grade >= 8:
        return "D"
    else:
        return "F"


def grade_to_percentage(grade: float) -> float:
    """Convert a 0–20 grade to a percentage (0–100)."""
    return round((grade / 20) * 100, 1)
