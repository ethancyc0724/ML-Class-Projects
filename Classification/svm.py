import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score
)

# 讀取資料集
data_path = "./data/airfoil_self_noise.dat"

col_names = [
    "frequency",
    "attack_angle",
    "chord_length",
    "free_stream_velocity",
    "suction_side_displacement_thickness",
    "scaled_sound_pressure_level"
]

df = pd.read_csv(
    data_path,
    sep=r"\s+",       # 一個或多個空白當分隔
    header=None,      # 檔案裡沒有欄名，所以我們自己給
    names=col_names
)

print("[Data] shape:", df.shape)
print(df.head())

# 任務設定共用
feature_cols = [
    "frequency",
    "attack_angle",
    "chord_length",
    "free_stream_velocity",
    "suction_side_displacement_thickness"
]

X = df[feature_cols].values
y_reg = df["scaled_sound_pressure_level"].values

# (A) SVM 分類：預測「高噪音 or 低噪音」
# 建立二元分類標籤：以中位數當門檻
threshold = np.average(y_reg)
df["noisy"] = (df["scaled_sound_pressure_level"] > threshold).astype(int)
y_clf = df["noisy"].values

print(f"[Classification] threshold (average) = {threshold:.2f} dB")
print(df["noisy"].value_counts())

# 切分訓練 / 測試集（分層抽樣，維持 0/1 比例）
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X,
    y_clf,
    test_size=0.2,
    random_state=42,
    stratify=y_clf
)

# 特徵標準化（SVM 很需要）
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

# 建立並訓練 SVM 分類器（RBF kernel）
clf = SVC(
    kernel="rbf",  # 非線性邊界，適合這種實驗資料
    C=10.0,        # 邊界寬 vs 錯誤容忍的 trade-off
    gamma="scale"  # 預設 gamma，根據資料自動縮放
)
clf.fit(X_train_clf_scaled, y_train_clf)

# 評估分類結果
y_pred_clf = clf.predict(X_test_clf_scaled)

print("\n=== SVM Classification Results ===")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_clf))
print("Confusion matrix:\n", confusion_matrix(y_test_clf, y_pred_clf))
print("Classification report:\n", classification_report(y_test_clf, y_pred_clf, digits=4))

# 簡單預測
sample = np.array([[800, 5.0, 0.3, 60.0, 0.002]])  # (freq, angle, chord, U, delta*)
sample_clf = scaler_clf.transform(sample)

noise_class = clf.predict(sample_clf)[0]

print("\n=== Example Prediction ===")
print("Predicted class (0=acceptable, 1=noisy):", int(noise_class))
