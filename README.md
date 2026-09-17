# Credit Card Fraud Detection (XGBoost + SMOTE)

A student assignment applying **XGBoost** to detect fraudulent credit card transactions on the **IEEE-CIS Fraud Detection** dataset from Kaggle. The dataset is heavily imbalanced, so the notebook uses **SMOTE** for oversampling and tunes the decision threshold instead of relying on the default 0.5 cutoff.

## Dataset

This project uses two files from the [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/c/ieee-fraud-detection) on Kaggle:

- `train_transaction.csv`
- `train_identity.csv`

The target variable is `isFraud`.

> **Note:** The CSV files are **not included** in this repository (they're too large and excluded via `.gitignore`). Download them from Kaggle and place them in the `data/` folder before running the notebook.

`test_transaction.csv`, `test_identity.csv`, and `sample_submission.csv` are **not used** in this project, since they don't include ground-truth labels.

## Project Structure

```
credit-card-fraud-detection/
├── .gitignore
├── README.md
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
├── data/
│   ├── train_transaction.csv      # not pushed (ignored by .gitignore)
│   └── train_identity.csv         # not pushed (ignored by .gitignore)
└── outputs/
    └── predictions.csv            # generated after running the notebook
```

## What the Notebook Does

1. Loads and merges `train_transaction.csv` and `train_identity.csv` on `TransactionID`
2. Inspects the data and checks the class imbalance in `isFraud`
3. Handles missing values (median for numerical columns, `"Unknown"` for categorical columns)
4. Encodes categorical variables with one-hot encoding
5. Splits the data into train/test sets using a **stratified split**
6. Applies **SMOTE** to the training data only (`sampling_strategy=0.2`)
7. Trains an **XGBoost** classifier
8. Evaluates the model at the default 0.5 threshold using accuracy, precision, recall, F1, ROC-AUC, and PR-AUC
9. Tunes the decision threshold and picks the one with the best F1 score
10. Compares default vs. tuned threshold results
11. Saves final predictions to `predictions.csv`
12. Plots the top 20 most important features from the trained model
13. Includes markdown explanations throughout, plus an interpretation and conclusion section

## How to Run

1. Download `train_transaction.csv` and `train_identity.csv` from Kaggle and place them in the `data/` folder (or upload them to `/content/` if running in Google Colab).
2. Open `notebooks/credit_card_fraud_detection.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. Update the file paths in the "Load Dataset" cell if needed (defaults to `/content/...` for Colab).
4. Run all cells top to bottom.
5. Check `predictions.csv` for the saved model output.

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
imbalanced-learn
xgboost
```

Install with:

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn xgboost
```

## Notes

- SMOTE is applied **only** to the training data, never to the test data, to keep evaluation realistic.
- Feature importance scores show what the model relied on most for its predictions — they do **not** imply that a feature *causes* fraud.
