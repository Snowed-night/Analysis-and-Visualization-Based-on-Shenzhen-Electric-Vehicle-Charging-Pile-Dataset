import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("plot_data.csv")

# 假设 "Proposed" 是预测值，"PAG-" 是真实值
y_true = data["PAG-"]
y_pred = data["Proposed"]

# 计算残差
residuals = y_pred - y_true

# =======================
# 1. 预测值与真实值对比曲线
# =======================
plt.figure(figsize=(10,5))
plt.plot(y_true.values, label="True (PAG-)")
plt.plot(y_pred.values, label="Predicted (Proposed)", linestyle="--")
plt.title("Predicted vs True Values")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# =======================
# 2. 残差分布直方图
# =======================
plt.figure(figsize=(8,5))
plt.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
