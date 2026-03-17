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