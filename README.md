# 💳 Loan Default Risk Predictor

A risk-based loan approval framework for Malaysian digital banks, using machine learning to score default probability and route applications through configurable lending strategies.

**🔗 Live demo:** [loan-default-risk-app-uzi2u6tsu244akfc94thny.streamlit.app](https://loan-default-risk-app-uzi2u6tsu244akfc94thny.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red)

---

## Problem

Digital banks need to decide, per application, whether to **approve, reject, or manually review** a loan — balancing two competing costs:
- Approving an applicant who later defaults (expensive: RM 20,000 assumed loss per case)
- Rejecting an applicant who would have repaid (opportunity cost: RM 1,000 assumed loss per case)

A single fixed cutoff doesn't fit every bank's risk appetite. This project builds a probability-based scoring model and layers three lending strategies on top, so the *business* — not the model — decides how conservative to be.

## Dataset

Kaggle Loan Default Prediction Dataset — **255,347 records**, 11.61% default rate. Features include applicant demographics, income, credit history, loan terms, and employment status.

## Approach

- Stratified train/test split (80/20) performed **before** any resampling, to avoid data leakage
- **SMOTE** applied only to the training fold to address class imbalance
- Two models compared: Logistic Regression vs. Random Forest, inside a full scikit-learn `Pipeline` (imputation → scaling/encoding → resampling → classifier)
- **Logistic Regression selected** — despite lower raw accuracy, it achieved far better recall on the minority (default) class, which matters more for a risk use case than overall accuracy

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (selected) | 0.222 | **0.692** | 0.336 | 0.753 |
| Random Forest | 0.429 | 0.140 | 0.211 | 0.744 |

> Random Forest looks "better" on accuracy (87.9% vs 68.3%) purely because it defaults to predicting the majority class. In a lending context, missing actual defaulters is the costlier mistake — so recall on the default class drove model selection, not accuracy.

## Business impact — three lending strategies

| Strategy | Approval Rate | Rejection Rate | Estimated Cost / Application (RM) | Estimated Total Cost (RM) |
|---|---|---|---|---|
| Conservative | 18.5% | 36.2% | 439.07 | 22,423,450 |
| **Balanced (recommended)** | 34.0% | 13.2% | **407.59** | **20,815,750** |
| Growth-Oriented | 49.5% | 5.1% | 538.59 | 27,505,750 |

The **Balanced strategy** minimizes total estimated cost across the test portfolio — it isn't the most conservative or the most aggressive option, but the one that best balances missed defaults against unnecessarily rejected good applicants.

## Live app

The Streamlit app loads the trained pipeline and lets you enter applicant details to get:
- Predicted probability of default
- The Approve / Manual Review / Reject decision under all three strategies side by side

## Tech stack

`Python` · `scikit-learn` · `imbalanced-learn (SMOTE)` · `pandas` / `numpy` · `Streamlit` · deployed on Streamlit Community Cloud

## Run locally

```bash
git clone https://github.com/vicheolis/loan-default-risk-app.git
cd loan-default-risk-app
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer

Built for academic coursework (MCSD2163 — Machine Learning in Finance, Master's degree). Cost assumptions (RM 20,000 per missed default, RM 1,000 per false rejection, RM 150 per manual review) are illustrative, not sourced from a real institution's risk model. Not intended for production lending decisions.

## Author

Nur Aina Farraain 
