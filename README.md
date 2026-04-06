# ml-diagnostics-overfitting
Nine diagnostic tools for detecting and understanding overfitting in scikit-learn models — polynomial overfitting, learning curves, validation curves, bias-variance decomposition, regularisation sweeps, data leakage detection, and more. Companion code for the ML Diagnostics Mastery series.


# ml-diagnostics-overfitting
 
> **ML Diagnostics Mastery — Part 1:** Why Your Model Overfits (And How to Catch It)
 
Companion code for the article published at [emitechlogic.com](https://emitechlogic.com/why-your-model-overfits-and-how-to-diagnose-it/).
 
Nine self-contained diagnostic functions that cover every major overfitting failure mode — from polynomial curve-fitting to data leakage detection. All experiments use synthetic data and fixed random seeds, so every number is fully reproducible.
 
---
 
## Diagnostics Covered
 
| # | Diagnostic | What It Reveals |
|---|---|---|
| 1 | Polynomial overfitting | Visual anchor: underfitting → good fit → overfit |
| 2 | Learning curves | High variance vs. high bias; whether more data helps |
| 3 | Validation curve | Hyperparameter sweet spot before model commitment |
| 4 | Train/val loss curves | Neural-network overfitting signature; early-stopping point |
| 5 | CV score distribution | Model instability across data splits |
| 6 | Bias–variance decomposition | Root cause of error; which lever to pull |
| 7 | Ridge & Lasso sweep | Regularisation effect on generalisation gap |
| 8 | Feature importance + leakage | Data leakage detection via importance imbalance |
| 9 | Full diagnosis report | One-call verdict: severe overfit / moderate / good fit / underfit |
 
---
 
## Quickstart
 
```bash
git clone https://github.com/YEmmimal/ml-diagnostics-overfitting.git
cd ml-diagnostics-overfitting
pip install -r requirements.txt
python app.py
```
 
Nine numbered PNGs will be saved to the working directory. Each corresponds to one diagnostic section in the article.
 
---
 
## Requirements
 
```
numpy>=1.24
matplotlib>=3.7
scikit-learn>=1.3
```
 
Or install from the provided file:
 
```bash
pip install -r requirements.txt
```
 
Python 3.9+ recommended. No GPU required.
 
---
 
## Output Files
 
Running `app.py` generates:
 
```
01_polynomial_overfitting.png
02_learning_curve_overfitting_deep_tree.png
02_learning_curve_good_fit_depth=4.png
02_learning_curve_underfitting_stump.png
03_validation_curve_decision_tree_max_depth.png
04_loss_curves.png
05_cv_distribution.png
06_bias_variance.png
07_regularisation.png
08_feature_importance.png
```
 
---
 
## Using `full_diagnosis()` in Your Own Project
 
The `full_diagnosis()` function is designed to drop into any training script as a one-call sanity check:
 
```python
from app import full_diagnosis
from sklearn.ensemble import RandomForestClassifier
 
model = RandomForestClassifier(n_estimators=100, random_state=42)
full_diagnosis(model, X, y, model_name="My Model")
```
 
Sample output:
 
```
============================================================
  OVERFITTING DIAGNOSIS — My Model
============================================================
  Train accuracy      : 1.0000
  Test accuracy       : 0.9333
  Train–Test gap      : 0.0667  ✓ OK
  CV mean (10-fold)   : 0.8958
  CV std  (10-fold)   : 0.0280  ✓ STABLE
============================================================
  Verdict: MILD OVERFITTING — monitor, may be acceptable in some cases
============================================================
```
 
**Verdict thresholds:**
 
| Condition | Verdict |
|---|---|
| `train_acc < 0.78` and `test_acc < 0.78` | UNDERFITTING |
| Gap > 0.18 | SEVERE OVERFITTING |
| Gap > 0.12 | MODERATE OVERFITTING |
| Gap > 0.06 | MILD OVERFITTING |
| Gap ≤ 0.06 | GOOD FIT |
 
---
 
## Key Results
 
**Best `max_depth` (validation curve sweep):** 8 — Val accuracy 0.808
 
**Early stopping point (simulated loss curves):** Epoch 68
 
**Bias–variance sweet spot:** `max_depth=4` — Total error 66,964 (minimum across all depths tested)
 
**CV stability comparison:**
 
| Model | CV Std | Verdict |
|---|---|---|
| Deep Tree | 0.064 | ⚠ Unstable |
| Shallow Tree | 0.054 | Borderline |
| Random Forest | 0.049 | ✓ Stable |
| Gradient Boosting | 0.046 | ✓ Stable |
 
**Data leakage simulation:** Train 1.000 → Test 0.547. LEAKY_feat claimed 0.575 feature importance — 8× the next highest feature.
 
---
 
## Series
 
This is Part 1 of the **ML Diagnostics Mastery** series.
 
- **Part 1 (this repo):** Overfitting — detection, decomposition, and remedies
- Part 2 (coming): Class imbalance — when accuracy lies
- Part 3 (coming): Feature engineering audits
- Part 4 (coming): Production drift detection
 
---
 
## Article
 
Full write-up with figures, analysis, and interpretation:
[emitechlogic.com — Why Your Model Overfits (And How to Diagnose It)](https://emitechlogic.com/why-your-model-overfits-and-how-to-diagnose-it/)
 
---
 
## Citation
 
If you use this code in your own work or teaching:
 
```bibtex
@misc{ml_diagnostics_2026,
  title   = {ML Diagnostics Mastery Part 1: Why Your Model Overfits},
  author  = {Emmimal P Alexander},
  year    = {2026},
  url     = {https://github.com/Emmimal/ml-diagnostics-overfitting},
  note    = {Companion code for the ML Diagnostics Mastery series}
}
```
 
---
 
## License
 
MIT — use it, adapt it, teach with it.
