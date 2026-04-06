"""
Why Your Model Overfits (And How to Diagnose It)
Complete diagnostic toolkit with examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import (
    train_test_split, cross_val_score, learning_curve, validation_curve
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, mean_squared_error, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# 1. SIMULATE OVERFITTING WITH POLYNOMIAL REGRESSION
# ==============================================================================

def demo_polynomial_overfitting():
    """Show underfitting vs overfitting vs good fit on a regression task."""
    np.random.seed(42)
    X = np.sort(np.random.uniform(0, 1, 30))
    y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.3, size=X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X.reshape(-1, 1), y, test_size=0.3, random_state=42
    )

    degrees = [1, 4, 15]
    plt.figure(figsize=(15, 4))

    for i, degree in enumerate(degrees):
        pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=degree)),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])
        pipeline.fit(X_train, y_train)

        train_score = pipeline.score(X_train, y_train)
        test_score  = pipeline.score(X_test,  y_test)

        X_plot = np.linspace(0, 1, 200).reshape(-1, 1)
        y_plot = pipeline.predict(X_plot)

        label = {1: "Underfitting", 4: "Good Fit", 15: "Overfitting"}[degree]

        plt.subplot(1, 3, i + 1)
        plt.scatter(X_train, y_train, label="Train", alpha=0.7, color="steelblue")
        plt.scatter(X_test,  y_test,  label="Test",  alpha=0.7, color="tomato")
        plt.plot(X_plot, y_plot, color="green", linewidth=2)
        plt.title(f"{label}\nDegree={degree} | Train R²={train_score:.2f} | Test R²={test_score:.2f}")
        plt.legend()
        plt.ylim(-2.5, 2.5)

    plt.tight_layout()
    plt.savefig("01_polynomial_overfitting.png", dpi=120)
    plt.show()
    print("Saved: 01_polynomial_overfitting.png")


# ==============================================================================
# 2. LEARNING CURVES — THE PRIMARY DIAGNOSTIC TOOL
# ==============================================================================

def plot_learning_curves(estimator, X, y, title="Learning Curve"):
    """
    Plot train vs cross-validation score as a function of training set size.

    Interpretation:
    - Large gap between train & val  → overfitting (high variance)
    - Both curves low & converging   → underfitting (high bias)
    - Converging with high scores    → good fit
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, "o-", color="steelblue", label="Training score")
    plt.fill_between(train_sizes,
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.2, color="steelblue")
    plt.plot(train_sizes, val_mean, "o-", color="tomato", label="Validation score")
    plt.fill_between(train_sizes,
                     val_mean - val_std,
                     val_mean + val_std,
                     alpha=0.2, color="tomato")

    gap = (train_mean - val_mean)[-1]
    plt.title(f"{title}\nFinal gap (train - val): {gap:.3f}")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    safe_title = title.replace(" ", "_").replace("(", "").replace(")", "").lower()
    fname = f"02_learning_curve_{safe_title}.png"
    plt.savefig(fname, dpi=120)
    plt.show()
    print(f"Saved: {fname}")
    return train_mean, val_mean


def demo_learning_curves():
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=10,
        n_redundant=5, random_state=42
    )

    overfit_model = DecisionTreeClassifier(max_depth=None, random_state=42)
    good_model    = DecisionTreeClassifier(max_depth=4,    random_state=42)
    underfit_model= DecisionTreeClassifier(max_depth=1,    random_state=42)

    for model, name in [
        (overfit_model,  "Overfitting (deep tree)"),
        (good_model,     "Good Fit (depth=4)"),
        (underfit_model, "Underfitting (stump)"),
    ]:
        plot_learning_curves(model, X, y, title=name)


# ==============================================================================
# 3. VALIDATION CURVE — HYPERPARAMETER SWEEP
# ==============================================================================

def plot_validation_curve(estimator, X, y, param_name, param_range, title=""):
    """
    Show how a single hyperparameter affects train vs val performance.
    The sweet spot is where validation score peaks.
    """
    train_scores, val_scores = validation_curve(
        estimator, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)

    best_idx   = np.argmax(val_mean)
    best_param = param_range[best_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(param_range, train_mean, "o-", color="steelblue", label="Train")
    plt.plot(param_range, val_mean,   "o-", color="tomato",    label="Validation")
    plt.axvline(best_param, color="green", linestyle="--",
                label=f"Best {param_name}={best_param}")
    plt.title(f"Validation Curve — {title}")
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    safe_title = title.replace(" ", "_").replace("(", "").replace(")", "").lower()
    fname = f"03_validation_curve_{safe_title}.png"
    plt.savefig(fname, dpi=120)
    plt.show()
    print(f"Best {param_name}: {best_param}  |  Val accuracy: {val_mean[best_idx]:.3f}")
    print(f"Saved: {fname}")


def demo_validation_curve():
    X, y = make_classification(n_samples=500, n_features=20,
                               n_informative=10, random_state=42)
    param_range = np.arange(1, 20)
    plot_validation_curve(
        DecisionTreeClassifier(random_state=42),
        X, y,
        param_name="max_depth",
        param_range=param_range,
        title="Decision Tree max_depth"
    )


# ==============================================================================
# 4. TRAIN / VALIDATION / TEST SPLIT — TRACKING LOSS OVER EPOCHS
# ==============================================================================

def simulate_training_curves():
    """
    Simulate a neural-network style loss curve showing the classic
    overfitting signature: val loss rising while train loss keeps falling.
    """
    np.random.seed(0)
    epochs = np.arange(1, 101)

    # Simulated curves
    train_loss = 1.0 / (1 + 0.07 * epochs) + np.random.normal(0, 0.005, 100)
    val_loss   = (1.0 / (1 + 0.05 * epochs)
                  + 0.003 * np.maximum(0, epochs - 40)
                  + np.random.normal(0, 0.01, 100))

    best_epoch = np.argmin(val_loss) + 1

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_loss, color="steelblue", label="Train loss")
    plt.plot(epochs, val_loss,   color="tomato",    label="Validation loss")
    plt.axvline(best_epoch, color="green", linestyle="--",
                label=f"Early stopping → epoch {best_epoch}")
    plt.fill_between(epochs, train_loss, val_loss,
                     where=(val_loss > train_loss), alpha=0.15, color="tomato",
                     label="Generalisation gap")
    plt.title("Train vs Validation Loss — Classic Overfitting Signature")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("04_loss_curves.png", dpi=120)
    plt.show()
    print(f"Best epoch (early stopping): {best_epoch}")
    print("Saved: 04_loss_curves.png")


# ==============================================================================
# 5. CROSS-VALIDATION SCORE DISTRIBUTION
# ==============================================================================

def demo_cross_validation():
    """
    High variance across CV folds → model is unstable → likely overfitting.
    Stable, high scores across folds → generalises well.
    """
    X, y = make_classification(n_samples=500, n_features=20,
                               n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Deep Tree (overfit)": DecisionTreeClassifier(max_depth=None),
        "Shallow Tree":        DecisionTreeClassifier(max_depth=4),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    print(f"\n{'Model':<30} {'CV Mean':>10} {'CV Std':>10} {'Test Acc':>10}")
    print("-" * 64)

    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy")
        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        cv_results[name] = scores
        print(f"{name:<30} {scores.mean():>10.3f} {scores.std():>10.3f} {test_acc:>10.3f}")

    # Box-plot of CV distributions
    plt.figure(figsize=(10, 5))
    plt.boxplot(cv_results.values(), labels=cv_results.keys(), patch_artist=True)
    plt.title("Cross-Validation Score Distribution\n(High variance = instability / overfitting)")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("05_cv_distribution.png", dpi=120)
    plt.show()
    print("Saved: 05_cv_distribution.png")


# ==============================================================================
# 6. BIAS–VARIANCE DECOMPOSITION (MANUAL)
# ==============================================================================

def bias_variance_decomposition(estimator, X, y, n_bootstrap=200, test_size=0.3):
    """
    Estimate bias² and variance via bootstrap resampling.

    Algorithm (manual approximation):
      1. Hold out a fixed test set.
      2. Train `n_bootstrap` models, each on a bootstrap sample of the train set.
      3. For each test point, collect all n_bootstrap predictions.
         - Bias²   = (mean prediction − true value)²      averaged over test points
         - Variance = variance of predictions across boots  averaged over test points
      4. Noise is estimated as the residual variance of the target itself.
         NOTE: This is a rough approximation suitable for illustration.  For a
         statistically rigorous decomposition use mlxtend.evaluate.bias_variance_decomp.

    Returns: (bias_sq, variance, noise, total_error)
    """
    np.random.seed(42)
    n = len(X)
    test_idx  = np.random.choice(n, size=int(n * test_size), replace=False)
    train_idx = np.setdiff1d(np.arange(n), test_idx)

    X_test,       y_test       = X[test_idx],  y[test_idx]
    X_train_full, y_train_full = X[train_idx], y[train_idx]

    all_preds = []
    for i in range(n_bootstrap):
        # Fix the per-iteration seed so results are fully reproducible across runs.
        rng = np.random.RandomState(i)
        idx = rng.choice(len(X_train_full), size=len(X_train_full), replace=True)
        estimator.fit(X_train_full[idx], y_train_full[idx])
        all_preds.append(estimator.predict(X_test))

    all_preds = np.array(all_preds)   # shape: (n_bootstrap, n_test)
    mean_pred = all_preds.mean(axis=0)

    bias_sq  = np.mean((mean_pred - y_test) ** 2)
    variance = np.mean(all_preds.var(axis=0))
    # Noise: irreducible error estimated from target variance on the test set.
    # For synthetic data with known noise=20 this is a lower-bound approximation.
    noise    = np.var(y_test - y_test.mean())
    total    = bias_sq + variance + noise

    return bias_sq, variance, noise, total


def demo_bias_variance():
    np.random.seed(42)
    X, y = make_regression(n_samples=300, n_features=10, noise=20, random_state=42)

    depths  = [1, 2, 4, 6, 10, None]
    biases, variances, totals = [], [], []

    for d in depths:
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(max_depth=d)
        b, v, n, t = bias_variance_decomposition(model, X, y)
        biases.append(b)
        variances.append(v)
        totals.append(t)
        label = str(d) if d else "None"
        print(f"max_depth={label:>4}  |  Bias²={b:8.1f}  Variance={v:8.1f}  Total={t:8.1f}")

    depth_labels = [str(d) if d else "None" for d in depths]
    x = np.arange(len(depths))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, biases,    width, label="Bias²",    color="steelblue")
    plt.bar(x + width/2, variances, width, label="Variance", color="tomato")
    plt.plot(x, totals, "D-", color="green", label="Total error", linewidth=2)
    plt.xticks(x, depth_labels)
    plt.xlabel("max_depth")
    plt.ylabel("Error")
    plt.title("Bias–Variance Tradeoff as Tree Depth Increases")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("06_bias_variance.png", dpi=120)
    plt.show()
    print("Saved: 06_bias_variance.png")


# ==============================================================================
# 7. REGULARISATION — RIDGE & LASSO
# ==============================================================================

def demo_regularisation():
    """
    Show how increasing regularisation strength shrinks coefficients and
    controls overfitting on a high-dimensional regression problem.
    """
    np.random.seed(42)
    n_samples, n_features = 100, 80          # more features than useful → easy to overfit
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           n_informative=10, noise=30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    alphas = np.logspace(-3, 4, 30)

    ridge_train, ridge_test = [], []
    lasso_train, lasso_test = [], []

    for alpha in alphas:
        for ModelClass, tr, te in [
            (Ridge, ridge_train, ridge_test),
            (Lasso, lasso_train, lasso_test),
        ]:
            m = ModelClass(alpha=alpha)
            m.fit(X_train, y_train)
            tr.append(mean_squared_error(y_train, m.predict(X_train)))
            te.append(mean_squared_error(y_test,  m.predict(X_test)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (tr, te, name) in zip(axes, [
        (ridge_train, ridge_test, "Ridge"),
        (lasso_train, lasso_test, "Lasso"),
    ]):
        ax.semilogx(alphas, tr, "o-", color="steelblue", label="Train MSE")
        ax.semilogx(alphas, te, "o-", color="tomato",    label="Test MSE")
        ax.set_xlabel("Alpha (regularisation strength)")
        ax.set_ylabel("MSE")
        ax.set_title(f"{name} Regularisation")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("07_regularisation.png", dpi=120)
    plt.show()
    print("Saved: 07_regularisation.png")


# ==============================================================================
# 8. FEATURE IMPORTANCE — DETECTING DATA LEAKAGE
# ==============================================================================

def demo_feature_importance():
    """
    Demonstrate data leakage as a hidden form of overfitting.

    The key mistake being replicated here: the leaky feature is derived from
    the label BEFORE the train/test split, so it contaminates the test set too
    — giving an unrealistically perfect score.  We then show the CORRECT way:
    inject the leaky feature only into the training split, which exposes the
    real train/test gap that leakage causes in production.
    """
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20,
                               n_informative=5, random_state=42)

    # --- CORRECT approach: split FIRST, then inject leak only into train ---
    # This mimics real-world leakage where a feature computed on the full
    # dataset (e.g. target-encoding without CV, future data) leaks into train
    # but is unavailable / different at serving time.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    feature_names = [f"feat_{i}" for i in range(20)]
    feature_names[0] = "LEAKY_feat"

    # Add a near-perfect copy of the label to the training set only.
    # At test/serving time this signal is absent → model collapses.
    X_train_leaky = X_train.copy()
    X_test_leaky  = X_test.copy()
    X_train_leaky[:, 0] = y_train + np.random.normal(0, 0.05, len(y_train))
    # Test set gets noise only — the label signal is gone (simulates production)
    X_test_leaky[:, 0]  = np.random.normal(0, 1.0, len(y_test))

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_leaky, y_train)

    train_acc = accuracy_score(y_train, rf.predict(X_train_leaky))
    test_acc  = accuracy_score(y_test,  rf.predict(X_test_leaky))
    print(f"\nWith leaky feature  →  Train: {train_acc:.3f}  |  Test: {test_acc:.3f}")
    print("⚠  Large train/test gap despite high train accuracy = leakage signature!")

    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:10]

    plt.figure(figsize=(10, 5))
    colors = ["tomato" if "LEAKY" in feature_names[i] else "steelblue" for i in idx]
    plt.bar(range(10), importances[idx], color=colors)
    plt.xticks(range(10), [feature_names[i] for i in idx], rotation=30, ha="right")
    plt.title(f"Feature Importances (Train={train_acc:.2f}, Test={test_acc:.2f})\n"
              "Red bar = leaky feature dominating train importance but absent at test time")
    plt.tight_layout()
    plt.savefig("08_feature_importance.png", dpi=120)
    plt.show()
    print("Saved: 08_feature_importance.png")


# ==============================================================================
# 9. COMPREHENSIVE OVERFITTING DIAGNOSIS REPORT
# ==============================================================================

def full_diagnosis(model, X, y, model_name="Model"):
    """
    Run a complete overfitting diagnostic and print a structured report.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    cv_scores = cross_val_score(model, X_train, y_train, cv=10)

    gap = train_acc - test_acc

    print(f"\n{'='*60}")
    print(f"  OVERFITTING DIAGNOSIS — {model_name}")
    print(f"{'='*60}")
    cv_unstable = cv_scores.std() > 0.07 or (cv_scores.std() / max(cv_scores.mean(), 1e-6)) > 0.10
    print(f"  Train accuracy      : {train_acc:.4f}")
    print(f"  Test accuracy       : {test_acc:.4f}")
    print(f"  Train–Test gap      : {gap:.4f}  {'⚠ HIGH' if gap > 0.10 else '✓ OK'}")
    print(f"  CV mean (10-fold)   : {cv_scores.mean():.4f}")
    print(f"  CV std  (10-fold)   : {cv_scores.std():.4f}  {'⚠ UNSTABLE' if cv_unstable else '✓ STABLE'}")
    print(f"{'='*60}")

    # Verdict logic — order matters:
    # 1. Check underfitting first (both scores low, gap is small but absolute performance is poor)
    # 2. Then check overfitting severity via train–test gap
    # 3. Fall through to good fit
    # Check underfitting FIRST: both scores are low regardless of the gap,
    # meaning the model lacks capacity, not that it memorised the training set.
    # Threshold 0.78 is intentionally slightly above the stump's 0.77 train score
    # to catch high-bias models before the gap-based overfitting checks run.
    if train_acc < 0.78 and test_acc < 0.78:
        verdict = "UNDERFITTING — increase model complexity, add better features, or reduce regularization"
    elif gap > 0.18:
        verdict = "SEVERE OVERFITTING — reduce complexity, add regularization, get more data"
    elif gap > 0.12:
        verdict = "MODERATE OVERFITTING — consider pruning or regularization"
    elif gap > 0.06:
        verdict = "MILD OVERFITTING — monitor, may be acceptable in some cases"
    else:
        verdict = "GOOD FIT — train and test performance are well aligned"

    print(f"  Verdict: {verdict}")
    print(f"{'='*60}\n")
    return train_acc, test_acc, cv_scores


def demo_full_diagnosis():
    X, y = make_classification(n_samples=600, n_features=20,
                               n_informative=10, random_state=42)

    models = {
        "Deep Decision Tree (overfit)": DecisionTreeClassifier(max_depth=None),
        "Pruned Decision Tree":         DecisionTreeClassifier(max_depth=4),
        "Random Forest":                RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":            GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Decision Stump (underfit)":    DecisionTreeClassifier(max_depth=1),
    }

    for name, model in models.items():
        full_diagnosis(model, X, y, model_name=name)


# ==============================================================================
# MAIN — Run all demos
# ==============================================================================

if __name__ == "__main__":
    # All demos use random_state=42 or np.random.seed(42) for reproducibility.
    # Each section is self-contained and saves a numbered PNG to the working directory.

    print("\n[1] Polynomial Overfitting Demo")
    print("    WHY: Shows visually how model complexity drives underfitting → good fit → overfit.")
    print("    LOOK FOR: degree-1 misses the curve; degree-15 wiggles wildly through noise.")
    demo_polynomial_overfitting()

    print("\n[2] Learning Curves")
    print("    WHY: The #1 diagnostic tool. Reveals whether adding more data will help.")
    print("    LOOK FOR: large train–val gap = high variance (overfit); both low = high bias (underfit).")
    demo_learning_curves()

    print("\n[3] Validation Curve — Hyperparameter Sweep")
    print("    WHY: Find the hyperparameter sweet spot before committing to a model.")
    print("    LOOK FOR: val score peaks then drops as complexity grows past the optimum.")
    demo_validation_curve()

    print("\n[4] Simulated Training / Validation Loss Curves")
    print("    WHY: Classic neural-network diagnostic — val loss rising while train loss falls.")
    print("    LOOK FOR: the epoch where val loss is minimum → that is your early-stopping point.")
    simulate_training_curves()

    print("\n[5] Cross-Validation Score Distribution")
    print("    WHY: High CV std means the model is unstable — it overfits to the particular split.")
    print("    LOOK FOR: deep trees have higher std; ensembles stabilise variance across folds.")
    demo_cross_validation()

    print("\n[6] Bias–Variance Decomposition")
    print("    WHY: Decomposes error into its root causes so you know which lever to pull.")
    print("    LOOK FOR: shallow trees → high bias; deep trees → high variance; sweet spot in between.")
    demo_bias_variance()

    print("\n[7] Regularisation (Ridge & Lasso)")
    print("    WHY: Shows how penalising coefficient magnitude shrinks the generalisation gap.")
    print("    LOOK FOR: test MSE bottoms out at an intermediate alpha — too low = overfit, too high = underfit.")
    demo_regularisation()

    print("\n[8] Feature Importance & Data Leakage Detection")
    print("    WHY: A dominant feature with near-perfect train accuracy often indicates leakage.")
    print("    LOOK FOR: one feature dwarfing all others in importance → investigate its provenance.")
    demo_feature_importance()

    print("\n[9] Full Overfitting Diagnosis Report")
    print("    WHY: Single-function summary covering gap, CV stability, and a plain-English verdict.")
    print("    LOOK FOR: gap > 0.15 = severe; both scores low = underfitting; converged & high = good fit.")
    demo_full_diagnosis()

    print("\nAll diagnostics complete. Check the generated PNG files for visuals.")
    print("Tip: Integrate full_diagnosis() into your training scripts as a quick sanity check!")
    print("""
╔══════════════════════════════════════════════════════════════╗
║                  KEY TAKEAWAYS                               ║
╠══════════════════════════════════════════════════════════════╣
║ 1. Learning curves are your first stop — always plot them.   ║
║    Large gap → get more data or regularize.                  ║
║    Both scores low → rethink features or model complexity.   ║
║                                                              ║
║ 2. Validation curves find the hyperparameter sweet spot      ║
║    before you commit to a final model.                       ║
║                                                              ║
║ 3. High CV std (> 0.06–0.07) means the model is             ║
║    split-sensitive — an ensemble or more data usually fixes. ║
║                                                              ║
║ 4. Near-perfect train accuracy + sharp test drop →           ║
║    suspect data leakage before celebrating.                  ║
║                                                              ║
║ 5. Bias–Variance: shallow = high bias, deep = high variance. ║
║    Total error is minimized somewhere in between.            ║
║                                                              ║
║ 6. Regularization (Ridge/Lasso) is the cheapest fix for      ║
║    overfit on high-dimensional regression problems.          ║
║                                                              ║
║ 7. The full_diagnosis() function gives a one-call verdict —  ║
║    wire it into your training pipeline as a sanity check.    ║
╚══════════════════════════════════════════════════════════════╝
""")
