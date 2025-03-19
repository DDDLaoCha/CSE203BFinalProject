from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import time


# 1️⃣ 读取数据
file_path = "California/housing.csv"  # 修改为你的实际路径
df = pd.read_csv(file_path)

# 2️⃣ 数据预处理
# 去掉 ocean_proximity（非数值特征）
df = df.drop(columns=["ocean_proximity"])

# 提取特征 X 和目标 Y（房价）
X = df.drop(columns=["median_house_value"]).values  # 所有特征
Y = df["median_house_value"].values  # 目标变量（房价）

# 标准化特征（提高优化收敛性）
scaler = StandardScaler()
X = scaler.fit_transform(X)
Y = Y / 100000  # 归一化房价（避免数值过大）

# 1️⃣ 找出 X 或 Y 中包含 NaN 的行索引
nan_mask = np.isnan(X).any(axis=1) | np.isnan(Y)

# 2️⃣ 仅保留非 NaN 的行
X_clean = X[~nan_mask]
Y_clean = Y[~nan_mask]

# 划分训练集 & 测试集
X_train, X_test, y_train, y_test = train_test_split(X_clean, Y_clean, test_size=0.2, random_state=42)

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

startTime = time.time()

# 训练 SVR 模型
svr_model = SVR(kernel='linear', C=1.0, gamma='scale')
# svr_model = SVR(kernel='rbf', C=1.0, gamma='scale')
svr_model.fit(X_train, y_train)

endTime = time.time()

print("Time: ", endTime - startTime)

# 预测
y_pred = scaler_y.inverse_transform(svr_model.predict(X_test).reshape(-1, 1))

w = svr_model.coef_  # w 是 (1, d) 维
b = svr_model.intercept_
alpha = np.zeros(X_train.shape[0])
alpha[svr_model.support_] = svr_model.dual_coef_


# print(y_pred[:5])  # 输出部分预测值

mse = mean_squared_error(y_test, y_pred)


print(f"SVR测试集均方误差 (MSE): {mse}")
