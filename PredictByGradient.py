import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# 1️⃣ 读取数据
file_path = "California/housing.csv"  # 修改为你的实际路径
df = pd.read_csv(file_path)

# 2️⃣ 数据预处理
df = df.drop(columns=["ocean_proximity"])  # 删除分类特征
X = df.drop(columns=["median_house_value"]).values  # 提取特征
Y = df["median_house_value"].values / 100000  # 目标变量归一化（单位变为 100k 美元）

# 3️⃣ 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4️⃣ 处理 NaN
nan_mask = np.isnan(X).any(axis=1) | np.isnan(Y)
X_clean, Y_clean = X[~nan_mask], Y[~nan_mask]

# 5️⃣ 训练集 & 测试集拆分
X_train, X_test, y_train, y_test = train_test_split(X_clean, Y_clean, test_size=0.2, random_state=42)

# 6️⃣ 初始化参数
n_features = X_train.shape[1]  # 获取特征数
w = np.random.randn(n_features)  # 初始化权重
b = np.random.randn()  # 初始化偏置

# 7️⃣ 设置超参数
learning_rate = 0.01
epochs = 1000  # 迭代次数
m = X_train.shape[0]  # 训练样本数
loss_history = []  # 存储每轮 MSE

startTime = time.time()

# 8️⃣ 训练（梯度下降优化）
for epoch in range(epochs):
    y_pred = X_train @ w + b  # 预测值
    loss = np.mean((y_pred - y_train) ** 2)  # MSE 损失
    loss_history.append(loss)  # 记录损失

    # 计算梯度
    dw = (2 / m) * (X_train.T @ (y_pred - y_train))  # 计算 w 的梯度
    db = (2 / m) * np.sum(y_pred - y_train)  # 计算 b 的梯度

    # 更新参数
    w -= learning_rate * dw
    b -= learning_rate * db

    # 每 100 轮打印一次损失
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

endTime = time.time()

print("Time: ", endTime - startTime)

# 9️⃣ 绘制 MSE 下降曲线
plt.figure(figsize=(8, 6))
plt.plot(loss_history, label="MSE Loss")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("MSE Loss Curve during Gradient Descent")
plt.legend()
plt.show()
