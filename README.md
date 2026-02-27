# Crohn's Disease Genetic Classifier

Binary classification of Crohn's disease using genetic SNP (Single Nucleotide Polymorphism) data.

## Problem Statement

Crohn's disease is a chronic inflammatory bowel disease with a significant genetic component. This project builds a machine-learning pipeline that classifies individuals as Crohn's disease patients or healthy controls based on genotype data from 103 genetic loci (206 allele features).

## Dataset

| Property | Value |
|---|---|
| Source | `data/crohn.csv` |
| Rows | 387 |
| Features | 207 (sex + 206 SNP allele columns) |
| Label | `crohn` — binarised to 1 (Crohn's) / 0 (control) |

Columns dropped before modeling: `Unnamed: 0`, `pid`, `id`, `fid`, `mid`.

## Modeling Approach

1. **Preprocessing** — Features are standardised via `StandardScaler` (embedded inside each sklearn `Pipeline`).
2. **Models trained** — Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machine.
3. **Hyperparameter tuning** — `GridSearchCV` with 5-fold stratified cross-validation, optimising ROC-AUC.
4. **Train/test split** — 70/30 stratified split (fixed seed for reproducibility).

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.8718 | 0.8085 | 0.8636 | 0.8352 | 0.9402 |
| Logistic Regression | 0.8803 | 0.7885 | 0.9318 | 0.8542 | 0.9402 |
| Gradient Boosting | 0.8632 | 0.7917 | 0.8636 | 0.8261 | 0.9324 |
| SVM | 0.8462 | 0.7708 | 0.8409 | 0.8043 | 0.8733 |

> Sorted by ROC-AUC (primary metric). Best model saved to `models/best_model.joblib`.

## Project Structure

```
├── data/
│   └── crohn.csv
├── models/
│   └── best_model.joblib      # generated at runtime
├── src/
│   ├── __init__.py
│   ├── data.py                 # data loading & preprocessing
│   ├── train.py                # pipeline definitions & GridSearchCV
│   └── evaluate.py             # metrics & comparison table
├── run.py                      # orchestrator
├── requirements.txt
└── README.md
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

## Key Findings

- All four models achieve ROC-AUC above 0.87, indicating strong separability between Crohn's and control genotypes in this dataset.
- Random Forest and Logistic Regression tie at 0.9402 ROC-AUC. Logistic Regression achieves the highest recall (0.93), while Random Forest has better precision (0.81).
- The strong performance of a simple regularised Logistic Regression suggests the decision boundary is largely linear in the scaled feature space.
- Gradient Boosting and SVM perform competitively but do not outperform the simpler models on this small dataset.

## Limitations

- Small sample size (387 rows) limits generalisability and makes overfitting a risk.
- SNP values are treated as ordinal integers; no domain-specific allele encoding (e.g., dominant/recessive) is applied.
- Some locus columns contain apparent missing-data markers (`0,0` pairs) that are kept as-is rather than imputed.
- Hyperparameter grids are intentionally small to keep training time reasonable.
- No external validation set — results are from a single stratified split.

## License

MIT
