import pandas as pd
import numpy as np
import time

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

# load data
data = pd.read_csv(
    "Data/HIGGS.csv.gz",
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




#RBF-kernel SVM

rbf_svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True
)

start_train = time.time()
rbf_svm.fit(X_train_scaled, y_train)
rbf_train_time = time.time() - start_train

start_pred = time.time()
rbf_preds = rbf_svm.predict(X_test_scaled)
rbf_probs = rbf_svm.predict_proba(X_test_scaled)[:,1]
rbf_infer_time = time.time() - start_pred

print("RBF SVM ROC-AUC:", roc_auc_score(y_test, rbf_probs))