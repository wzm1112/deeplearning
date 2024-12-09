import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

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

# 数据分析：特征与目标的相关性
sns.heatmap(ventilation_data[selected_features + target_columns].corr(), annot=True, cmap="coolwarm")
plt.title("Feature-Target Correlation")
plt.show()

# 固定边索引
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 2, 4],  # 起点索引
    [1, 2, 3, 3, 4, 5, 5]   # 终点索引
], dtype=torch.long)

# 构建多状态图数据对象
num_states = X.shape[0]
edge_attr = torch.tensor(X.values, dtype=torch.float)

# 确保 edge_attr 的形状与边数量匹配
if edge_attr.shape[0] == 1 and edge_attr.shape[1] == len(edge_index[0]):
    edge_attr = edge_attr.T
elif edge_attr.dim() == 1:
    edge_attr = edge_attr.unsqueeze(1)

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
        self.conv3 = GCNConv(hidden_dim, hidden_dim)  # 增加一层卷积
        self.edge_fc = Linear(hidden_dim * 2 + edge_input_dim, hidden_dim)  # 修正为实际输入维度
        self.graph_fc = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attrs):
        outputs = []
        for edge_attr in edge_attrs:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)  # 确保 edge_attr 是二维张量

            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)  # Dropout
            x = F.relu(self.conv2(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)  # Dropout
            x = F.relu(self.conv3(x, edge_index))  # 新增卷积层

            edge_start = x[edge_index[0]]
            edge_end = x[edge_index[1]]
            edge_features = torch.cat([edge_start, edge_end, edge_attr], dim=1)

            # 验证 edge_features 的形状
            print("edge_features shape:", edge_features.shape)

            edge_output = F.relu(self.edge_fc(edge_features))
            graph_features = torch.mean(edge_output, dim=0, keepdim=True)
            outputs.append(self.graph_fc(graph_features))
        return torch.cat(outputs, dim=0)

# 初始化模型
model = GNNRegressionModel(
    node_input_dim=1,
    edge_input_dim=edge_attr.size(1),
    hidden_dim=16,  # 增大隐藏层维度
    output_dim=y.size(1)
)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# 加载之前的模型权重
model_save_path = "gcn_model.pth"
try:
    pretrained_dict = torch.load(model_save_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"Model partially loaded from {model_save_path}")
except FileNotFoundError:
    print("No existing model found. Starting fresh training.")

# 确保模型处于训练模式
model.train()

# 设置训练轮数
new_epochs = 1
save_interval = 500  # 每 500 次保存一次模型
losses = []

for epoch in range(1, new_epochs + 1):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{new_epochs}, Loss: {loss.item():.4f}")

    if epoch % save_interval == 0:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model updated and saved at epoch {epoch} to {model_save_path}")

# 保存最终模型
torch.save(model.state_dict(), model_save_path)
print(f"Final model saved to {model_save_path}")

# 可视化训练损失
plt.figure(figsize=(12, 6))
plt.plot(range(1, new_epochs + 1), losses, label="Training Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve (Continued Training)")
plt.legend()
plt.grid(True)
plt.show()

# 模型评估
model.eval()

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

# 可视化真实值和预测值
plt.figure(figsize=(10, 6))
plt.scatter(range(len(true_values)), true_values[:, 0], label="True Values", color="blue", alpha=0.6)
plt.scatter(range(len(predictions)), predictions[:, 0], label="Predictions", color="red", alpha=0.6)
plt.title("True Values vs Predictions (After Continued Training)")
plt.xlabel("Sample Index")
plt.ylabel("Target Value (First Output Dimension)")
plt.legend()
plt.grid(True)
plt.show()
