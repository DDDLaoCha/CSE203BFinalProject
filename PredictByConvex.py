import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# 3️⃣ **构建凸优化问题**
N, d = X_train.shape  # 训练样本数 & 特征数
w = cp.Variable(d)  # 回归权重
b = cp.Variable()   # 偏置
xi = cp.Variable(N, nonneg=True)  # 松弛变量 ξ
epsilon = 0.1  # 误差容忍度
C = 1.0  # 正则化参数（控制误差与复杂度）

# **目标函数：最小化 ||w||^2 + C * ∑ ξ**
objective = cp.Minimize(0.5 * cp.norm(w, 2) ** 2 + C * cp.sum(xi))

# **约束条件**
constraints = [
    y_train - (X_train @ w + b) <= epsilon + xi,  # 误差上界
    (X_train @ w + b) - y_train <= epsilon + xi,  # 误差下界
    xi >= 0  # 松弛变量非负
]

startTime = time.time()

# **求解优化问题**
problem = cp.Problem(objective, constraints)
problem.solve()

print(f"w.value: {w.value}")  # 查看优化后的值（若优化成功）
print(f"b.value: {b.value}")  # 查看优化后的值（若优化成功）

endTime = time.time()

print("Time: ", endTime - startTime)

# 4️⃣ **预测**
y_pred = X_test @ w.value + b.value  # 使用优化解进行预测

# 5️⃣ **评估性能**
mse = mean_squared_error(y_test, y_pred)


print(f"测试集均方误差 (MSE): {mse}")

# 6️⃣ **可视化：真实 vs. 预测**
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Real Price (100k Dollars)")
plt.ylabel("Predict Price (100k Dollars)")
plt.title("Reality vs. Predict")
plt.show()
