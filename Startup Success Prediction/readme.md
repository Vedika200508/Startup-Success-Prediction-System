# 🚀 AI Powered Startup Success Prediction System

A Machine Learning web application that predicts whether a startup is likely to **succeed or fail**, based on funding history, milestones, investor participation, and startup category.

Built with Python, Streamlit, Scikit-Learn, and SQLite.

---

## 📌 Project Overview

This project uses a **Random Forest classifier** trained on real-world startup data to predict startup outcomes. Users can input startup details through an interactive web interface and instantly receive a prediction along with a confidence score.

All predictions are stored in a local **SQLite database** for history tracking and analysis.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application framework |
| Scikit-Learn | Machine Learning (Random Forest) |
| Pandas & NumPy | Data preprocessing and analysis |
| Joblib | Model serialization (.pkl) |
| SQLite | Prediction history database |

---

## 📁 Project Structure

```
Startup Success Prediction/
│
├── app.py               → Main Streamlit web application
├── train_model.py       → Model training script
├── database.py          → Database setup helper
├── dataset.csv          → Startup dataset (923 records)
├── startup_model.pkl    → Pre-trained Random Forest model
├── startup.db           → SQLite database (auto-created)
└── requirements.txt     → Python dependencies
```

---

## ⚙️ How to Run

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — (Optional) Train the model yourself

```bash
python train_model.py
```

This will preprocess the dataset, train the Random Forest model, print accuracy results, and save the model as `startup_model.pkl`.

> Skip this step if you want to use the pre-trained model directly.

### Step 3 — Launch the web app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📊 Dataset

- **Source:** Real-world startup data
- **Records:** 923 startups
- **Features:** 31 (after preprocessing)
- **Target:** `status` — Success (acquired/operating/ipo) or Failure (closed)

**Key features used for prediction:**

- Age at first and last funding
- Age at first and last milestone
- Number of funding rounds
- Total funding amount (USD)
- Number of business milestones
- Investor relationships
- Startup category (Software, Web, Mobile, Biotech, etc.)
- Location (California, New York, Texas, etc.)
- Investment type (VC, Angel, Round A/B/C/D)

---

## 🤖 Machine Learning Model

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Number of Trees | 100 |
| Max Depth | 10 |
| Train/Test Split | 80% / 20% |
| Accuracy | ~80% |

---

## 🖥️ Application Pages

| Page | Description |
|---|---|
| 🏠 Home | Project introduction and feature overview |
| 🚀 Prediction | Enter startup details and get prediction |
| 📊 Prediction History | View and download all past predictions |
| 🤖 AI Insights | Model accuracy, features, and analysis |
| 📈 Dashboard | Dataset statistics and visual charts |
| ℹ️ About | Project and technology summary |

---

## 👩‍💻 Developed By

**Vedika Chauhan**
BCA — Machine Learning Project