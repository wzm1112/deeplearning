import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 数据加载与清洗
file_path = '../datasets/es[7]-vs[6]-random-train.csv'  # 替换为实际路径
ventilation_data = pd.read_csv(file_path)
ventilation_data.fillna(ventilation_data.mean(), inplace=True)

# 特征选择与标准化
selected_features = ["q'1", "q'2", "q'3", "q'4", "q'5", "q'6", "q'7"]
X = ventilation_data[selected_features]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=selected_features)

# 目标变量（多维回归目标）
target_columns = ["h'1", "h'2", "h'3", "h'4", "h'5", "h'6", "h'7"]
y = ventilation_data[target_columns]
y_scaled = scaler.fit_transform(y)
y = torch.tensor(y_scaled, dtype=torch.float)

# 固定边索引
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 2, 4],  # 起点索引
    [1, 2, 3, 3, 4, 5, 5]   # 终点索引
], dtype=torch.long)

# 构建多状态图数据对象
edge_attr = torch.tensor(X.values, dtype=torch.float)
node_features = torch.ones((6, 1))

# 构造图数据对象
data = Data(
    x=node_features,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=y
)

# 定义支持回归任务的 GNN 模型
class GNNRegressionModel(torch.nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, output_dim):
        super(GNNRegressionModel, self).__init__()
        self.conv1 = GCNConv(node_input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_fc = Linear(3, hidden_dim)  # 确保输入维度与拼接的 edge_features 一致
        self.graph_fc = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attrs):
        outputs = []
        for edge_attr in edge_attrs:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)  # 确保 edge_attr 是二维张量

            edge_start = x[edge_index[0]]
            edge_end = x[edge_index[1]]

            edge_features = torch.cat([edge_start, edge_end, edge_attr], dim=1)
            edge_output = F.relu(self.edge_fc(edge_features))
            graph_features = torch.mean(edge_output, dim=0, keepdim=True)
            outputs.append(self.graph_fc(graph_features))
        return torch.cat(outputs, dim=0)

# 初始化模型
model = GNNRegressionModel(
    node_input_dim=1,
    edge_input_dim=edge_attr.size(1),
    hidden_dim=8,
    output_dim=y.size(1)
)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# 加载之前的模型权重
model_save_path = "gcn_model.pth"
try:
    model.load_state_dict(torch.load(model_save_path))
    print(f"Model loaded from {model_save_path}")
except FileNotFoundError:
    print("No existing model found. Starting fresh training.")

# 确保模型处于训练模式
model.train()

# 设置训练轮数
new_epochs = 20  # 继续训练的轮数
save_interval = 500  # 每 500 次保存一次模型
losses = []

for epoch in range(1, new_epochs + 1):  # 继续训练
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    # 打印损失
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{new_epochs}, Loss: {loss.item():.4f}")

# 保存最终模型
torch.save(model.state_dict(), model_save_path)
print(f"Final model saved to {model_save_path}")

# 可视化训练损失
plt.figure(figsize=(12, 6))
plt.plot(range(1, new_epochs + 1), losses, label="Training Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.show()

# 模型评估
model.eval()  # 切换到评估模式

# 评估模型
with torch.no_grad():
    predictions = model(data.x, data.edge_index, data.edge_attr)
    predictions = predictions.cpu().numpy()
    true_values = data.y.cpu().numpy()

# 计算评估指标
mse = mean_squared_error(true_values, predictions)
mae = mean_absolute_error(true_values, predictions)
r2 = r2_score(true_values, predictions)

print("Evaluation Results:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# 可视化 1: 真实值 vs 预测值
plt.figure(figsize=(10, 6))
plt.scatter(range(len(true_values)), true_values[:, 0], label="True Values", color="blue", alpha=0.6)
plt.scatter(range(len(predictions)), predictions[:, 0], label="Predictions", color="red", alpha=0.6)
plt.title("True Values vs Predictions (First Output Dimension)")
plt.xlabel("Sample Index")
plt.ylabel("Target Value")
plt.legend()
plt.grid(True)
plt.show()

# 可视化 2: 残差图
residuals = predictions[:, 0] - true_values[:, 0]
plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuals)), residuals, label="Residuals", color="purple", alpha=0.6)
plt.axhline(0, color="red", linestyle="--", label="Zero Residual")
plt.title("Residual Plot")
plt.xlabel("Sample Index")
plt.ylabel("Residual")
plt.legend()
plt.grid(True)
plt.show()

# 可视化 3: 分布对比
plt.figure(figsize=(10, 6))
plt.hist(true_values[:, 0], bins=20, alpha=0.6, label="True Values", color="blue")
plt.hist(predictions[:, 0], bins=20, alpha=0.6, label="Predictions", color="orange")
plt.title("Distribution of True Values and Predictions")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

# 可视化 4: R² 分布（多维目标）
r2_scores = [r2_score(true_values[:, i], predictions[:, i]) for i in range(true_values.shape[1])]
plt.figure(figsize=(10, 6))
plt.bar(range(len(r2_scores)), r2_scores, color="skyblue")
plt.title("R² Scores Across Output Dimensions")
plt.xlabel("Output Dimension")
plt.ylabel("R² Score")
plt.ylim(0, 1)
plt.grid(True)
plt.show()
