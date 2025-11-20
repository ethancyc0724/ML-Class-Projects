import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
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

# SVR 迴歸：預測實際噪音值 (dB)
# 切分訓練 / 測試集
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X,
    y_reg,
    test_size=0.2,
    random_state=42
)

# 特徵標準化
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# 建立並訓練 SVR（同樣用 RBF kernel）
reg = SVR(
    kernel="rbf",
    C=10.0,      # 對誤差的懲罰程度
    epsilon=0.5, # epsilon tube，容許小誤差不罰
    gamma="scale"
)
reg.fit(X_train_reg_scaled, y_train_reg)

# 評估迴歸效果
y_pred_reg = reg.predict(X_test_reg_scaled)

rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
r2 = r2_score(y_test_reg, y_pred_reg)

print("\n=== SVR Regression Results ===")
print("RMSE:", rmse)
print("R^2:", r2)

# 簡單預測
sample = np.array([[800, 5.0, 0.3, 60.0, 0.002]])  # (freq, angle, chord, U, delta*)
sample_reg = scaler_reg.transform(sample)

noise_value = reg.predict(sample_reg)[0]

print("\n=== Example Prediction ===")
print("Predicted SPL (dB):", noise_value)
