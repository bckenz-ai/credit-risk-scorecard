# Credit Risk Scorecard
> [View notebook on Google Colab](https://colab.research.google.com/github/bckenz-ai/credit-risk-scorecard/blob/main/Credit_Risk_Scorecard.ipynb)

An end-to-end credit risk scorecard built on the
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
dataset from Kaggle.

## Project Overview

This notebook walks through the full scorecard development pipeline:

- **EDA:** Income distribution, credit density, and income-annuity relationships
- **Imbalance Diagnostics:** Severity (IR = 11.4:1), EPV check, KS-based class overlap analysis, and PCA/UMAP projections before and after Borderline-SMOTE
- **Feature Engineering:** Five domain-driven credit ratios derived from raw fields
- **WOE/IV Filtering:** Industry-standard Weight of Evidence encoding and Information Value filtering to retain only features with predictive power
- **Final Pipeline:** Imputation, scaling, Borderline-SMOTE, and Logistic Regression evaluated by ROC-AUC and the Gini Coefficient

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/credit-risk-scorecard.git
cd credit-risk-scorecard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download `application_train.csv` from the
[Kaggle competition page](https://www.kaggle.com/competitions/home-credit-default-risk/data?select=application_train.csv)
and place it in the `data/` folder:
credit-risk-scorecard/
└── data/
└── application_train.csv

### 4. Run the notebook

Open `Credit_Risk_Scorecard.ipynb` in Jupyter and run all cells.

## Results

| Metric | Score |
|---|---|
| ROC-AUC (3 WOE features) | 0.584 |
| Gini Coefficient | 0.169 |
| ROC-AUC (D4 baseline, 80 features) | 0.708 |

## References

Elreedy, D., & Atiya, A. F. (2019). A Comprehensive Analysis of Synthetic Minority
Oversampling Technique (SMOTE) for handling class imbalance. *Information Sciences*,
505, 32–64.

Thomas, L. C., Crook, J. N., & Edelman, D. B. (2017). *Credit scoring and its
applications*. Society for Industrial and Applied Mathematics.
