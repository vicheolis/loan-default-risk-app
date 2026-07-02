# 💳 Loan Default Risk Predictor

A risk-based loan approval framework for Malaysian digital banks, using machine learning to score default probability and route applications through configurable lending strategies — built and shipped with a full MLOps pipeline (experiment tracking, automated CI/CD, containerized deployment).

**🔗 Live demo:** [loan-default-risk-app-uzi2u6tsu244akfc94thny.streamlit.app](https://loan-default-risk-app-uzi2u6tsu244akfc94thny.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red)
![MLflow](https://img.shields.io/badge/MLflow-experiment_tracking-0194E2)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF)
![Docker](https://img.shields.io/badge/Docker-GHCR-2496ED)

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
- Two models compared during development: Logistic Regression vs. Random Forest, inside a full scikit-learn `Pipeline` (imputation → scaling/encoding → resampling → classifier)
- **Logistic Regression selected for production** — despite lower raw accuracy, it achieved far better recall on the minority (default) class, which matters more for a risk use case than overall accuracy

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (selected) | 0.222 | **0.692** | 0.336 | 0.753 |
| Random Forest | 0.429 | 0.140 | 0.211 | 0.744 |

> Random Forest looks "better" on accuracy (87.9% vs 68.3%) purely because it defaults to predicting the majority class. In a lending context, missing actual defaulters is the costlier mistake — so recall on the default class drove model selection, not accuracy.

## Business impact — three lending strategies

| Strategy | Approval Rate | Rejection Rate | Estimated Cost / Application (RM) | Estimated Total Cost (RM) |
|---|---|---|---|---|
| **Conservative (production)** | 18.5% | 36.2% | 439.07 | 22,423,450 |
| Balanced (lowest modeled cost) | 34.0% | 13.2% | 407.59 | 20,815,750 |
| Growth-Oriented | 49.5% | 5.1% | 538.59 | 27,505,750 |

While Balanced minimizes total *modeled* cost, **Conservative was selected for production** to further limit exposure to approved defaults — prioritizing downside protection over marginal cost efficiency, a reasonable stance for a digital bank still building its risk appetite. All three strategies remain available in the scoring function for comparison.

## MLOps pipeline

This project goes beyond a single trained model — it's wired into an automated pipeline so the model can be retrained, validated, and shipped safely:

- **Experiment tracking (MLflow):** every training run logs its hyperparameters, metrics, and model artifact, so runs are comparable and reproducible rather than overwritten each time.
- **CI/CD with a quality gate (GitHub Actions):** every push to `main` automatically retrains the model and re-evaluates it. If F1 or recall drops below a set threshold, the build fails on purpose — catching a regression before it reaches production, rather than after.
- **Containerized build (Docker + GHCR):** the app and model are packaged into a Docker image and automatically built and published to GitHub Container Registry on every push, so the environment is reproducible on any machine:
  ```bash
  docker pull ghcr.io/vicheolis/loan-default-risk-app:latest
  docker run -p 8501:8501 ghcr.io/vicheolis/loan-default-risk-app:latest
  ```

## Live app

The Streamlit app loads the trained pipeline and lets you enter applicant details to get:
- Predicted probability of default
- The Approve / Manual Review / Reject decision under all three strategies side by side

## Tech stack

`Python` · `scikit-learn` · `imbalanced-learn (SMOTE)` · `pandas` / `numpy` · `MLflow` · `GitHub Actions` · `Docker` · `Streamlit` — deployed on Streamlit Community Cloud, containerized via GitHub Container Registry

## Run locally

```bash
git clone https://github.com/vicheolis/loan-default-risk-app.git
cd loan-default-risk-app
pip install -r requirements.txt
streamlit run app.py
```

Or via Docker (no local Python setup needed):
```bash
docker pull ghcr.io/vicheolis/loan-default-risk-app:latest
docker run -p 8501:8501 ghcr.io/vicheolis/loan-default-risk-app:latest
```

## Disclaimer

Built for academic coursework (MCSD2163 — Machine Learning in Finance, Master's degree). Cost assumptions (RM 20,000 per missed default, RM 1,000 per false rejection, RM 150 per manual review) are illustrative, not sourced from a real institution's risk model. Not intended for production lending decisions.

## Author

Nur Aina Farraain
