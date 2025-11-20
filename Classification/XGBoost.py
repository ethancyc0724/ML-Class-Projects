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
from xgboost import XGBClassifier      

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

def clean_col(name: str) -> str:
    # 確保是字串
    name = str(name)
    # 去掉不允許的符號或改寫
    name = name.replace("[", "").replace("]", "").replace("<", "lt")
    # 可順便處理空白與單位
    name = name.replace(" ", "_")
    return name

df.columns = [clean_col(c) for c in df.columns]

print(df.columns)

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

pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
if pos > 0:
    scale_pos_weight = neg / pos
else:
    scale_pos_weight = 1.0

print(f"\nPositive (1) count: {pos}, Negative (0) count: {neg}")
print(f"scale_pos_weight 設定為: {scale_pos_weight:.3f}")

# 定義與訓練 XGBoost 模型
xgb_model = XGBClassifier(
    n_estimators=300,          # 樹的數量
    max_depth=5,              # 樹的最大深度
    learning_rate=0.1,        # 學習率
    subsample=0.8,            # 每棵樹使用的樣本比例
    colsample_bytree=0.8,     # 每棵樹使用的特徵比例
    objective="binary:logistic",
    eval_metric="logloss",    # 固定評估指標，避免警告
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,                # 使用全部 CPU core
    random_state=42
)

print("\n開始訓練 XGBoost 模型...")
xgb_model.fit(X_train, y_train)

# 評估模型
y_pred = xgb_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)

print("\n=== Evaluation: XGBoost on AI4I 2020 Dataset ===")
print(f"Accuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)