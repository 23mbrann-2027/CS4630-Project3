import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.tree import DecisionTreeClassifier

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