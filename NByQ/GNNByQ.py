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

#import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Transformer 模型定义
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers=2):
        super(TransformerModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nhead = nhead
        self.num_layers = num_layers

        # Ensure input_dim is divisible by nhead
        if input_dim % nhead != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by nhead ({nhead})")

        # 定义 Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )

        # 线性层映射到输出
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 输入 x 的形状为 (batch_size, seq_len, input_dim)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, input_dim) -> (seq_len, batch_size, input_dim)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # 将输出映射到目标维度
        x = x.mean(dim=0)  # 按时间步求均值，(batch_size, seq_len, hidden_dim) -> (batch_size, hidden_dim)
        x = self.fc(x)  # 映射到输出
        return x

# 加载并处理数据
file_path = '../datasets/es[7]-vs[6]-random-train.csv'  # 替换为你的实际数据路径
data = pd.read_csv(file_path)

# 假设选择一些特征和目标
selected_features = ["q'1", "q'2", "q'3", "q'4", "q'5", "q'6", "q'7"]
target_columns = ["ei"]  # 目标变量

# 特征选择与标准化
X = data[selected_features]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=selected_features)

# 目标变量（确保是整数类型）
y = data[target_columns].astype(int)

# 转换为 PyTorch Tensor
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)  # CrossEntropyLoss 需要目标是 long 类型

# 增加额外的维度以适应 Transformer 输入
X_tensor = X_tensor.unsqueeze(1)  # (batch_size, 7) -> (batch_size, seq_len=1, 7)
print(X_tensor.shape)  # 输出 X_tensor 的形状，应该是 (batch_size, 1, 7)

# 创建模型
model = TransformerModel(
    input_dim=7,  # 7 个特征
    hidden_dim=64,
    output_dim=5,  # 分类的类别数
    nhead=1
)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # 对于分类问题，使用 CrossEntropyLoss

# 数据加载器
batch_size = 32
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in data_loader:
        optimizer.zero_grad()
        # 确保 batch_y 是一维张量
        batch_y = batch_y.squeeze()  # 将目标张量转为一维
        batch_y = batch_y - 2  # 调整目标，使其从 0 开始

        # 前向传播
        y_pred = model(batch_X)

        # 计算损失
        loss = criterion(y_pred, batch_y)

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 每 10 个 epoch 打印一次损失
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data_loader):.4f}")

# 评估模型
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor)  # 使用模型进行预测
    y_pred = y_pred.cpu().numpy()  # 转换为 NumPy 数组
    true_values = y_tensor.cpu().numpy()  # 转换为 NumPy 数组

# 计算评估指标
mse = mean_squared_error(true_values, y_pred)
mae = mean_absolute_error(true_values, y_pred)
r2 = r2_score(true_values, y_pred)

print("Evaluation Results:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# 可视化真实值和预测值
plt.figure(figsize=(10, 6))
plt.scatter(range(len(true_values)), true_values[:, 0], label="True Values", color="blue", alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred[:, 0], label="Predictions", color="red", alpha=0.6)
plt.title("True Values vs Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Target Value")
plt.legend()
plt.grid(True)
plt.show()


