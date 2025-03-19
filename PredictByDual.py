import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1️⃣ 读取数据
file_path = "California/housing.csv"  # 修改为你的实际路径
df = pd.read_csv(file_path)

# 2️⃣ 数据预处理
df = df.drop(columns=["ocean_proximity"])
X = df.drop(columns=["median_house_value"]).values
Y = df["median_house_value"].values

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
Y = Y / 100000  # 归一化

# 处理 NaN 数据
nan_mask = np.isnan(X).any(axis=1) | np.isnan(Y)
X_clean = X[~nan_mask]
Y_clean = Y[~nan_mask]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_clean, Y_clean, test_size=0.2, random_state=42)

# 3️⃣ **对偶优化求解 SVR**
N, d = X_train.shape
alpha = cp.Variable(N)
alpha_star = cp.Variable(N)

epsilon = 0.1
C = 1.0

# 计算核矩阵（线性核）
K = X_train @ X_train.T

# **对偶问题最大化 J(α)，我们在 CVXPY 里使用 `Minimize(-J(α))`**
objective = cp.Minimize(
    - cp.sum(y_train * (alpha - alpha_star)) + epsilon * cp.sum(alpha + alpha_star)
    + 0.5 * cp.quad_form(alpha - alpha_star, K)
)

# **约束条件**
constraints = [
    cp.sum(alpha - alpha_star) == 0,
    alpha >= 0, alpha <= C,
    alpha_star >= 0, alpha_star <= C
]

# **求解**
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.ECOS)

# **计算 w 和 b**
w_value = (alpha.value - alpha_star.value) @ X_train  # 计算 w
support_vectors = (alpha.value - alpha_star.value) != 0  # 选择支持向量
b_value = np.mean(y_train[support_vectors] - X_train[support_vectors] @ w_value)

print("w:", w_value)
print("b:", b_value)
