import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 讀取檔案路徑
csv_path = "./data/ai4i2020.csv"
df = pd.read_csv(csv_path)

# 顯示前五個row，確保DataFrame的正確性
print(df.head())
print(df.columns)


drop_cols = []
for col in df.columns:
    if "id" in col.lower() or col.lower() == "udi":
        drop_cols.append(col)

print("Drop columns:", drop_cols)
df = df.drop(columns=drop_cols)

# 確保目標欄位名稱
target_col = "Machine failure" if "Machine failure" in df.columns else "Machine_failure"

# 類別欄位做 Label Encoding（這裡針對 'Type'）
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
print("Categorical columns:", cat_cols)

encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])
    encoders[c] = le

# 特徵 / 標籤切開
X = df.drop(columns=[target_col])
y = df[target_col]

print("X shape:", X.shape)
print("y value counts:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 依照類別比例分層抽樣，避免嚴重偏差
)

rf = RandomForestClassifier(
    n_estimators=200,      # 樹的數量，越多越穩定但越慢
    max_depth=None,       # 不限制深度，讓模型自己長到適合的深度
    min_samples_split=10, # 控制樹的生長，避免過擬合
    n_jobs=-1,            # 使用所有CPU核心
    random_state=42,
    class_weight="balanced"  # 對付類別不平衡：給少數類別較大權重
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)

print("Accuracy:", acc)
print("\nConfusion matrix:\n", cm)
print("\nClassification report:\n", report)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature importance (descending):")
for i in indices:
    print(f"{X.columns[i]:25s}: {importances[i]:.4f}")

plt.figure(figsize=(8, 4))
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), X.columns[indices], rotation=45, ha="right")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()
