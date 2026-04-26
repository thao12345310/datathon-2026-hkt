# 🏁 Datathon 2026 — The Gridbreakers (Round 1)

**Team**: HKT  
**Competition**: [Datathon 2026 Round 1](https://www.kaggle.com/competitions/datathon-2026-round-1)  
**Topic**: E-commerce Fashion Business Data Analysis (Vietnam, 2012–2024)

## 📁 Project Structure

```
datathon-2026/
├── README.md
├── data/                       # CSV data files (14 files, not committed)
│   └── *.csv
├── notebooks/
│   ├── 01_eda_overview.ipynb   # Phase 1: Setup & initial EDA
│   ├── 02_mcq_answers.ipynb    # Phase 2: 10 MCQ answers with code
│   ├── 03_eda_deep.ipynb       # Phase 3: Deep EDA & visualizations
│   └── 04_forecasting.ipynb    # Phase 4: Revenue/COGS forecasting
├── src/
│   ├── utils.py                # Data loading & merging utilities
│   └── feature_engineering.py  # Feature engineering for forecasting
├── reports/
│   └── report.pdf              # LaTeX report (NeurIPS 2025 template)
├── submission/
│   └── submission.csv          # Kaggle submission file
├── Styles/                     # NeurIPS LaTeX template
└── docs/
    └── Đề thi Vòng 1.pdf      # Competition brief
```

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn plotly xgboost lightgbm shap statsmodels jupyter

# 3. Download data from Kaggle
kaggle competitions download -c datathon-2026-round-1 -p data/

# 4. Run notebooks in order
jupyter notebook notebooks/
```

## 📝 Competition Sections

| Section | Points | Description |
|---------|--------|-------------|
| Part 1 — MCQ | 20 | 10 multiple-choice questions (2 pts each) |
| Part 2 — EDA & Viz | 60 | Visualizations + 4-level analysis (D→D→P→P) |
| Part 3 — Forecasting | 20 | Revenue & COGS prediction (2023-01-01 → 2024-07-01) |

## 🔧 Random Seed

All models use `random_seed = 42` for reproducibility.
