import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

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

# 加载保存的模型权重
model_save_path = "gcn_model.pth"  # 替换为保存的模型路径
model.load_state_dict(torch.load(model_save_path))
model.eval()  # 切换到评估模式

# 使用模型进行评估
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
plt.title("True Values vs Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Target Value (First Output Dimension)")
plt.legend()
plt.grid(True)
plt.show()

