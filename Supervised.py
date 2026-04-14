#Comparing both methods
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

# load data
data = pd.read_csv(
    "Data/raw/HIGGS.csv.gz",
    header=None,
    compression="gzip",
    nrows=1000000
)

y = data.iloc[:,0].astype(int)
X = data.iloc[:,1:]


# Train 80, test 20.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train the tree
tree = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)

start_train = time.time()
tree.fit(X_train, y_train)
train_time = time.time() - start_train


# Have tree make pridictions.
start_pred = time.time()
preds = tree.predict(X_test)
probs = tree.predict_proba(X_test)[:,1]
infer_time = time.time() - start_pred


#Evaluate results
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
roc = roc_auc_score(y_test, probs)
pr = average_precision_score(y_test, probs)

print("Decision Tree Results")
print("Accuracy:", acc)
print("F1 Score:", f1)
print("ROC-AUC:", roc)
print("PR-AUC:", pr)
print("Training Time:", train_time)
print("Inference Time:", infer_time)

# Train the Random Forest
rf = RandomForestClassifier(
    n_estimators=100,     # number of trees
    max_depth=10,         # keep similar to your tree for fair comparison
    n_jobs=-1,            # use all cores
    random_state=42
)

start_train = time.time()
rf.fit(X_train, y_train)
rf_train_time = time.time() - start_train

# Predictions
start_pred = time.time()
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:,1]
rf_infer_time = time.time() - start_pred

# Evaluation
rf_acc = accuracy_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds)
rf_roc = roc_auc_score(y_test, rf_probs)
rf_pr = average_precision_score(y_test, rf_probs)

print("\nRandom Forest Results")
print("Accuracy:", rf_acc)
print("F1 Score:", rf_f1)
print("ROC-AUC:", rf_roc)
print("PR-AUC:", rf_pr)
print("Training Time:", rf_train_time)
print("Inference Time:", rf_infer_time)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train k-NN
knn = KNeighborsClassifier(n_neighbors=5)

start_train = time.time()
knn.fit(X_train_scaled, y_train)
knn_train_time = time.time() - start_train

# Predictions
start_pred = time.time()
knn_preds = knn.predict(X_test_scaled)
knn_probs = knn.predict_proba(X_test_scaled)[:,1]
knn_infer_time = time.time() - start_pred

# Evaluation
knn_acc = accuracy_score(y_test, knn_preds)
knn_f1 = f1_score(y_test, knn_preds)
knn_roc = roc_auc_score(y_test, knn_probs)
knn_pr = average_precision_score(y_test, knn_probs)

print("\nk-NN Results")
print("Accuracy:", knn_acc)
print("F1 Score:", knn_f1)
print("ROC-AUC:", knn_roc)
print("PR-AUC:", knn_pr)
print("Training Time:", knn_train_time)
print("Inference Time:", knn_infer_time)

# Linear SVM


# Scale features
scaler_svm = StandardScaler()
X_train_scaled = scaler_svm.fit_transform(X_train)
X_test_scaled = scaler_svm.transform(X_test)

# Linear SVM
svm = SGDClassifier(
    loss="hinge",        # linear SVM objective
    alpha=1e-4,          # regularization strength
    max_iter=1000,
    tol=1e-3,
    random_state=42
)

# Train
start_train = time.time()
svm.fit(X_train_scaled, y_train)
svm_train_time = time.time() - start_train

# Predictions
start_pred = time.time()
svm_preds = svm.predict(X_test_scaled)

# decision_function replaces predict_proba for SVM
svm_scores = svm.decision_function(X_test_scaled)
svm_infer_time = time.time() - start_pred

# Metrics
svm_acc = accuracy_score(y_test, svm_preds)
svm_f1 = f1_score(y_test, svm_preds)
svm_roc = roc_auc_score(y_test, svm_scores)
svm_pr = average_precision_score(y_test, svm_scores)

print("\nLinear SVM (SGD) Results")
print("Accuracy:", svm_acc)
print("F1 Score:", svm_f1)
print("ROC-AUC:", svm_roc)
print("PR-AUC:", svm_pr)
print("Training Time:", svm_train_time)
print("Inference Time:", svm_infer_time)

# XGBoost (Gradient Boosting)

xgb = XGBClassifier(
    n_estimators=200,        # number of trees
    max_depth=6,             # controls complexity
    learning_rate=0.1,
    subsample=0.8,           # row sampling
    colsample_bytree=0.8,    # feature sampling
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

# Train
start_train = time.time()
xgb.fit(X_train, y_train)
xgb_train_time = time.time() - start_train

# Predict
start_pred = time.time()
xgb_preds = xgb.predict(X_test)
xgb_probs = xgb.predict_proba(X_test)[:, 1]
xgb_infer_time = time.time() - start_pred

# Metrics
xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_f1 = f1_score(y_test, xgb_preds)
xgb_roc = roc_auc_score(y_test, xgb_probs)
xgb_pr = average_precision_score(y_test, xgb_probs)

print("\nXGBoost Results")
print("Accuracy:", xgb_acc)
print("F1 Score:", xgb_f1)
print("ROC-AUC:", xgb_roc)
print("PR-AUC:", xgb_pr)
print("Training Time:", xgb_train_time)
print("Inference Time:", xgb_infer_time)

# Hyperparameter tuning

# We use RandomizedSearchCV with 3-fold stratified CV for each model.
# This searches a random subset of the hyperparameter space, which is far
# more practical than exhaustive GridSearch on 1M rows.
import warnings

warnings.filterwarnings("ignore")



cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# --- Decision Tree ---
tree_param_dist = {
    "max_depth":        randint(4, 20),
    "min_samples_leaf": randint(10, 200),
    "criterion":        ["gini", "entropy"],
}
tree_search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    tree_param_dist,
    n_iter=15, cv=cv, scoring="roc_auc",
    n_jobs=-1, random_state=42, verbose=1
)
print("\nTuning Decision Tree...")
t0 = time.time()
tree_search.fit(X_train, y_train)
tree_cv_train_t = time.time() - t0
print("Best params:", tree_search.best_params_)

best_tree = tree_search.best_estimator_
t0 = time.time()
tree_cv_preds = best_tree.predict(X_test)
tree_cv_probs = best_tree.predict_proba(X_test)[:, 1]
tree_cv_infer_t = time.time() - t0


# --- Random Forest ---
rf_param_dist = {
    "n_estimators":     randint(50, 200),
    "max_depth":        randint(5, 20),
    "min_samples_leaf": randint(5, 100),
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(n_jobs=-1, random_state=42),
    rf_param_dist,
    n_iter=10, cv=cv, scoring="roc_auc",
    n_jobs=-1, random_state=42, verbose=1
)
print("\nTuning Random Forest...")
t0 = time.time()
rf_search.fit(X_train, y_train)
rf_cv_train_t = time.time() - t0
print("Best params:", rf_search.best_params_)

best_rf = rf_search.best_estimator_
t0 = time.time()
rf_cv_preds = best_rf.predict(X_test)
rf_cv_probs = best_rf.predict_proba(X_test)[:, 1]
rf_cv_infer_t = time.time() - t0


# --- k-NN ---
knn_param_dist = {
    "n_neighbors": randint(3, 30),
    "metric":      ["euclidean", "manhattan", "minkowski"],
}
knn_search = RandomizedSearchCV(
    KNeighborsClassifier(n_jobs=-1),
    knn_param_dist,
    n_iter=10, cv=cv, scoring="roc_auc",
    n_jobs=-1, random_state=42, verbose=1
)
print("\nTuning k-NN...")
t0 = time.time()
knn_search.fit(X_train_scaled, y_train)
knn_cv_train_t = time.time() - t0
print("Best params:", knn_search.best_params_)

best_knn = knn_search.best_estimator_
t0 = time.time()
knn_cv_preds = best_knn.predict(X_test_scaled)
knn_cv_probs = best_knn.predict_proba(X_test_scaled)[:, 1]
knn_cv_infer_t = time.time() - t0


# --- Linear SVM ---
svm_param_dist = {
    "alpha":    uniform(1e-5, 1e-2),
    "max_iter": randint(500, 2000),
}
svm_search = RandomizedSearchCV(
    SGDClassifier(loss="hinge", random_state=42),
    svm_param_dist,
    n_iter=10, cv=cv, scoring="roc_auc",
    n_jobs=-1, random_state=42, verbose=1
)
print("\nTuning Linear SVM...")
t0 = time.time()
svm_search.fit(X_train_scaled, y_train)
svm_cv_train_t = time.time() - t0
print("Best params:", svm_search.best_params_)

best_svm = svm_search.best_estimator_
t0 = time.time()
svm_cv_preds  = best_svm.predict(X_test_scaled)
svm_cv_scores = best_svm.decision_function(X_test_scaled)
svm_cv_infer_t = time.time() - t0


# --- XGBoost ---
xgb_param_dist = {
    "n_estimators":     randint(100, 400),
    "max_depth":        randint(3, 10),
    "learning_rate":    uniform(0.01, 0.3),
    "subsample":        uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
}
xgb_search = RandomizedSearchCV(
    XGBClassifier(eval_metric="logloss", tree_method="hist",
                  n_jobs=-1, random_state=42),
    xgb_param_dist,
    n_iter=15, cv=cv, scoring="roc_auc",
    n_jobs=-1, random_state=42, verbose=1
)
print("\nTuning XGBoost...")
t0 = time.time()
xgb_search.fit(X_train, y_train)
xgb_cv_train_t = time.time() - t0
print("Best params:", xgb_search.best_params_)

best_xgb = xgb_search.best_estimator_
t0 = time.time()
xgb_cv_preds = best_xgb.predict(X_test)
xgb_cv_probs = best_xgb.predict_proba(X_test)[:, 1]
xgb_cv_infer_t = time.time() - t0

# --- RBF SVM — subsampled to 100k rows ---
# Justification: sklearn's SVC has O(n^2) memory and O(n^2-n^3) training
# complexity. A timing pilot on the full 800k training rows projected
# > 8 hours. Subsampling to 100k keeps runtime manageable while still
# preserving the original class balance (stratified split).
SUBSAMPLE_N = 100_000
sub_idx = np.random.default_rng(42).choice(len(X), size=SUBSAMPLE_N, replace=False)
X_sub = X.iloc[sub_idx].reset_index(drop=True)
y_sub = y.iloc[sub_idx].reset_index(drop=True)

X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
    X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub
)
scaler_sub = StandardScaler()
X_train_sub_sc = scaler_sub.fit_transform(X_train_sub)
X_test_sub_sc  = scaler_sub.transform(X_test_sub)

rbf_param_grid = {
    "C":     [0.1, 1.0, 10.0],
    "gamma": ["scale", "auto"],
}
rbf_search = GridSearchCV(
    SVC(kernel="rbf", probability=False),
    rbf_param_grid,
    cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
)
print("\nTuning RBF SVM (50k subsample)...")
t0 = time.time()
rbf_search.fit(X_train_sub_sc, y_train_sub)
rbf_cv_train_t = time.time() - t0
print("Best params:", rbf_search.best_params_)

best_rbf = rbf_search.best_estimator_
t0 = time.time()
rbf_cv_preds = best_rbf.predict(X_test_sub_sc)
rbf_cv_probs = best_rbf.decision_function(X_test_sub_sc)
rbf_cv_infer_t = time.time() - t0


# Model comparison

def make_row(name, y_true, preds_, scores_, train_t, infer_t):
    return {
        "Model":          name,
        "Accuracy":       round(accuracy_score(y_true, preds_),            4),
        "F1":             round(f1_score(y_true, preds_),                  4),
        "ROC-AUC":        round(roc_auc_score(y_true, scores_),            4),
        "PR-AUC":         round(average_precision_score(y_true, scores_),  4),
        "Train Time (s)": round(train_t, 2),
        "Infer Time (s)": round(infer_t, 2),
    }

results = [
    make_row("Decision Tree",   y_test,     tree_cv_preds, tree_cv_probs, tree_cv_train_t, tree_cv_infer_t),
    make_row("Random Forest",   y_test,     rf_cv_preds,   rf_cv_probs,   rf_cv_train_t,   rf_cv_infer_t),
    make_row("k-NN",            y_test,     knn_cv_preds,  knn_cv_probs,  knn_cv_train_t,  knn_cv_infer_t),
    make_row("Linear SVM",      y_test,     svm_cv_preds,  svm_cv_scores, svm_cv_train_t,  svm_cv_infer_t),
    make_row("XGBoost",         y_test,     xgb_cv_preds,  xgb_cv_probs,  xgb_cv_train_t,  xgb_cv_infer_t),
    make_row("RBF SVM (100k)",  y_test_sub, rbf_cv_preds,  rbf_cv_probs,  rbf_cv_train_t,  rbf_cv_infer_t),
]

df_results = pd.DataFrame(results)
print("\n\n=== 3a — Tuned Model Comparison (Raw Features) ===")
print(df_results.to_string(index=False))


# 3a visualizations


COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

# Metrics bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("3a — Tuned Model Comparison (Raw Features)", fontsize=14, fontweight="bold")

for ax, metric in zip(axes, ["ROC-AUC", "F1", "Accuracy"]):
    ax.bar(df_results["Model"], df_results[metric], color=COLORS)
    ax.set_title(metric, fontweight="bold")
    ax.set_ylim(df_results[metric].min() - 0.05, 1.0)
    ax.set_xticklabels(df_results["Model"], rotation=30, ha="right")
    ax.set_ylabel(metric)
    for i, v in enumerate(df_results[metric]):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("3a_metrics_comparison.png", dpi=150)
plt.show()

# Scalability chart
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("3a — Scalability: Training vs Inference Time", fontsize=14, fontweight="bold")

axes[0].barh(df_results["Model"], df_results["Train Time (s)"], color=COLORS)
axes[0].set_title("Training Time (s)")
axes[0].set_xlabel("Seconds")

axes[1].barh(df_results["Model"], df_results["Infer Time (s)"], color=COLORS)
axes[1].set_title("Inference Time (s)")
axes[1].set_xlabel("Seconds")

plt.tight_layout()
plt.savefig("3a_scalability.png", dpi=150)
plt.show()

print("\n3a Interpretability ranking (most to least interpretable):")
print("  Decision Tree > Random Forest > XGBoost > Linear SVM > k-NN > RBF SVM")
print("  Decision Tree: human-readable rule paths")
print("  RF / XGBoost:  feature importances available, but no single tree to inspect")
print("  SVMs:          rely on support vectors / kernel — largely a black box")

# Part 3b integrate supervised + unsupervised



# Fit PCA and k-Means on X_train only to avoid data leakage into X_test
pca10 = PCA(n_components=10, random_state=42)
X_train_pca = pca10.fit_transform(X_train_scaled)
X_test_pca  = pca10.transform(X_test_scaled)
print(f"\nPCA-10 cumulative variance explained: {pca10.explained_variance_ratio_.sum():.2%}")

# k-Means cluster labels (fit on training set only)
kmeans = MiniBatchKMeans(n_clusters=2, batch_size=10_000, n_init=10, random_state=42)
kmeans.fit(X_train_scaled)
train_cluster = kmeans.predict(X_train_scaled).reshape(-1, 1)
test_cluster  = kmeans.predict(X_test_scaled).reshape(-1, 1)

# Four feature sets we will compare across models
feature_sets = {
    "Raw":              (X_train,                                     X_test),
    "PCA-10":           (X_train_pca,                                 X_test_pca),
    "Raw + Cluster":    (np.hstack([X_train_scaled, train_cluster]),  np.hstack([X_test_scaled,  test_cluster])),
    "PCA-10 + Cluster": (np.hstack([X_train_pca,   train_cluster]),  np.hstack([X_test_pca,     test_cluster])),
}

# Run three scalable models across all four feature sets.
models_3b = {
    "Decision Tree": best_tree,
    "Random Forest": best_rf,
    "XGBoost":       best_xgb,
}

results_3b = []

print("\n3b")
for fs_name, (Xtr, Xte) in feature_sets.items():
    for model_name, base_model in models_3b.items():
        # Deep-copy so each run starts from the same tuned hyperparameters
        clf = copy.deepcopy(base_model)

        t0 = time.time()
        clf.fit(Xtr, y_train)
        tr_t = time.time() - t0

        t0 = time.time()
        preds_ = clf.predict(Xte)
        probs_ = clf.predict_proba(Xte)[:, 1]
        inf_t = time.time() - t0

        row = {
            "Features":       fs_name,
            "Model":          model_name,
            "Accuracy":       round(accuracy_score(y_test, preds_),            4),
            "F1":             round(f1_score(y_test, preds_),                  4),
            "ROC-AUC":        round(roc_auc_score(y_test, probs_),             4),
            "PR-AUC":         round(average_precision_score(y_test, probs_),   4),
            "Train Time (s)": round(tr_t,  2),
            "Infer Time (s)": round(inf_t, 2),
        }
        results_3b.append(row)
        print(f"  [{fs_name}] {model_name}  ROC-AUC={row['ROC-AUC']}")

df_3b = pd.DataFrame(results_3b)
print("\n=== 3b — Full Comparison Table ===")
print(df_3b.to_string(index=False))


# visualizations

feature_order = ["Raw", "PCA-10", "Raw + Cluster", "PCA-10 + Cluster"]
model_order   = list(models_3b.keys())
x     = np.arange(len(feature_order))
bar_w = 0.25

# A. ROC-AUC grouped bar chart
fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle("3b — ROC-AUC by Feature Set and Model", fontsize=14, fontweight="bold")

for i, mname in enumerate(model_order):
    vals = [
        df_3b.loc[(df_3b["Features"] == fs) & (df_3b["Model"] == mname), "ROC-AUC"].values[0]
        for fs in feature_order
    ]
    ax.bar(x + i * bar_w, vals, bar_w, label=mname, color=COLORS[i])
    for j, v in enumerate(vals):
        ax.text(x[j] + i * bar_w, v + 0.001, f"{v:.3f}", ha="center", fontsize=7.5)

ax.set_xticks(x + bar_w)
ax.set_xticklabels(feature_order)
ax.set_ylabel("ROC-AUC")
ax.set_ylim(df_3b["ROC-AUC"].min() - 0.03, 1.0)
ax.legend()
plt.tight_layout()
plt.savefig("3b_roc_auc.png", dpi=150)
plt.show()

# B. Training time grouped bar chart
fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle("3b — Training Time by Feature Set and Model", fontsize=14, fontweight="bold")

for i, mname in enumerate(model_order):
    vals = [
        df_3b.loc[(df_3b["Features"] == fs) & (df_3b["Model"] == mname), "Train Time (s)"].values[0]
        for fs in feature_order
    ]
    ax.bar(x + i * bar_w, vals, bar_w, label=mname, color=COLORS[i])

ax.set_xticks(x + bar_w)
ax.set_xticklabels(feature_order)
ax.set_ylabel("Seconds")
ax.legend()
plt.tight_layout()
plt.savefig("3b_training_time.png", dpi=150)
plt.show()

# C. Accuracy delta — does PCA or cluster label improve things vs raw?
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("3b — Accuracy Change vs Raw Features", fontsize=14, fontweight="bold")

for ax_i, (enhanced_fs, title) in enumerate([
    ("PCA-10",        "Raw → PCA-10"),
    ("Raw + Cluster", "Raw → Raw + Cluster"),
]):
    deltas, labels_ = [], []
    for mname in model_order:
        base = df_3b.loc[(df_3b["Features"] == "Raw")         & (df_3b["Model"] == mname), "Accuracy"].values[0]
        enh  = df_3b.loc[(df_3b["Features"] == enhanced_fs)   & (df_3b["Model"] == mname), "Accuracy"].values[0]
        deltas.append(round(enh - base, 4))
        labels_.append(mname)

    bar_colors = ["#55A868" if d >= 0 else "#C44E52" for d in deltas]
    axes[ax_i].bar(labels_, deltas, color=bar_colors)
    axes[ax_i].axhline(0, color="black", lw=0.8, linestyle="--")
    axes[ax_i].set_title(title, fontweight="bold")
    axes[ax_i].set_ylabel("Δ Accuracy")
    for j, v in enumerate(deltas):
        axes[ax_i].text(j, v + (0.0003 if v >= 0 else -0.001),
                        f"{v:+.4f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("3b_accuracy_delta.png", dpi=150)
plt.show()



print("\nSummary")

best_row = df_results.loc[df_results["ROC-AUC"].idxmax()]
print(f"\n3a Best model (ROC-AUC): {best_row['Model']}  ({best_row['ROC-AUC']})")

fastest_row = df_results.loc[df_results["Train Time (s)"].idxmin()]
print(f"3a Fastest to train:     {fastest_row['Model']}  ({fastest_row['Train Time (s)']}s)")

print("\n3b — Does PCA-10 help vs Raw?")
for mname in model_order:
    raw = df_3b.loc[(df_3b["Features"] == "Raw")    & (df_3b["Model"] == mname), "ROC-AUC"].values[0]
    pca = df_3b.loc[(df_3b["Features"] == "PCA-10") & (df_3b["Model"] == mname), "ROC-AUC"].values[0]
    print(f"  {mname}: Raw={raw}  PCA-10={pca}  delta={pca - raw:+.4f}")

print("\n3b — Does adding cluster label help vs Raw?")
for mname in model_order:
    raw  = df_3b.loc[(df_3b["Features"] == "Raw")           & (df_3b["Model"] == mname), "ROC-AUC"].values[0]
    clus = df_3b.loc[(df_3b["Features"] == "Raw + Cluster") & (df_3b["Model"] == mname), "ROC-AUC"].values[0]
    print(f"  {mname}: Raw={raw}  +Cluster={clus}  delta={clus - raw:+.4f}")

print("\nPlots saved: 3a_metrics_comparison.png, 3a_scalability.png,")
print("             3b_roc_auc.png, 3b_training_time.png, 3b_accuracy_delta.png")

