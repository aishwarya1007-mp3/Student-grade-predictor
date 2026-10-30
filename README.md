# 🎓 Student Grade Prediction — ML Project

A complete college-level Machine Learning project that predicts a student's
final mathematics grade using Scikit-learn models and a professional
Streamlit web interface.

---

## 📁 Project Structure

```
AIML_project/
├── app.py              ← Main Streamlit web application (run this)
├── model.py            ← ML model training, evaluation & prediction
├── data_loader.py      ← Dataset generation, loading & preprocessing
├── utils.py            ← All chart / visualisation functions
├── student_data.csv    ← Auto-generated on first run (UCI-style dataset)
├── requirements.txt    ← Python package dependencies
└── README.md           ← This file
```

---

## 🗃️ Dataset

The project uses a synthetic version of the **UCI Student Performance Dataset**
(Mathematics grades). The CSV is generated automatically the first time you run
the app — no manual download is required.

| Feature | Description |
|---|---|
| `studytime` | Weekly study time (1=<2 h … 4=>10 h) |
| `failures` | Number of past class failures |
| `absences` | Number of school absences |
| `G1` | First period grade (0–20) |
| `G2` | Second period grade (0–20) |
| `Medu / Fedu` | Mother's / Father's education level (0–4) |
| `goout` | Going out with friends (1–5) |
| `health` | Health status (1–5) |
| `internet` | Internet access at home |
| `higher` | Wants higher education |
| **`G3`** | **Final grade — prediction target (0–20)** |

---

## 🤖 Machine Learning Models

All models are regression models (predicting a numeric grade):

| Model | Description |
|---|---|
| Linear Regression | Interpretable baseline model |
| Decision Tree | Simple rule-based tree |
| Random Forest | Ensemble of decision trees (usually best) |
| Gradient Boosting | Boosted ensemble (high accuracy) |

**Metrics used:** MAE, MSE, RMSE, R² Score  
All metrics are computed from real predictions on the held-out test set.

---

## ⚙️ Setup & Installation

### Step 1 — Make sure Python is installed

Open a terminal / PowerShell and check:

```powershell
python --version
```

You need **Python 3.9 or higher**. Download from https://www.python.org if needed.

---

### Step 2 — (Recommended) Create a virtual environment

```powershell
cd "C:\Users\Aishwarya Gadhave\AIML_project"
python -m venv venv
venv\Scripts\activate
```

After activation you will see `(venv)` at the start of your prompt.

---

### Step 3 — Install dependencies

```powershell
pip install -r requirements.txt
```

This installs Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn.

---

### Step 4 — Run the application

```powershell
streamlit run app.py
```

Streamlit will open the app in your default browser automatically at:
```
http://localhost:8501
```

---

## 🖥️ Running in VS Code

1. Open the `DSML_project` folder in VS Code (`File → Open Folder`).
2. Open the integrated terminal: **Terminal → New Terminal** (`Ctrl+\``)
3. Activate the virtual environment:
   ```powershell
   venv\Scripts\activate
   ```
4. Run the app:
   ```powershell
   streamlit run app.py
   ```

---

## 🌐 App Pages

| Page | What it shows |
|---|---|
| 🏠 Home Dashboard | Project overview, KPIs, feature list, data preview |
| 📊 Data Exploration | Grade distributions, heatmap, study time charts |
| 🤖 Model Evaluation | Metrics table, bar chart comparison, metric explanations |
| 🎯 Predict Grade | Input form → predicted grade + letter grade + percentage |
| 📈 Model Insights | Actual vs predicted, residuals, feature importance |

---

## 📊 Evaluation Metrics Explained

| Metric | Meaning | Better when |
|---|---|---|
| **MAE** | Average absolute error in grade units | Lower |
| **MSE** | Mean of squared errors (penalises big errors) | Lower |
| **RMSE** | Square root of MSE (same unit as grade) | Lower |
| **R²** | % of grade variance explained by the model | Closer to 1 |

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| App doesn't open in browser | Navigate to `http://localhost:8501` manually |
| `streamlit: command not found` | Make sure your virtual environment is activated |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |

---

## 📚 Technologies Used

- **Python 3.9+** — Core language
- **Pandas & NumPy** — Data manipulation
- **Scikit-learn** — Machine learning models and preprocessing
- **Streamlit** — Interactive web interface
- **Matplotlib & Seaborn** — Data visualisation

---

*Built as a college Machine Learning project demonstration.*
