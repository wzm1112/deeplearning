import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# 标签映射为连续整数
label_mapping = {label: idx for idx, label in enumerate(sorted(ventilation_data['ei'].unique()))}
ventilation_data['ei'] = ventilation_data['ei'].map(label_mapping)
print("Updated labels:", ventilation_data['ei'].unique())

# 固定边索引
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 2, 4],  # 起点索引
    [1, 2, 3, 3, 4, 5, 5]   # 终点索引
], dtype=torch.long).to(device)  # 将边索引迁移到 GPU

# 构建多状态图数据对象
num_states = X.shape[0]  # 状态数量等于行数
edge_attr = torch.tensor(X.values, dtype=torch.float).to(device)  # 边特征迁移到 GPU
node_features = torch.ones((6, 8)).to(device)  # 节点特征迁移到 GPU
data_y = torch.tensor(ventilation_data['ei'].values, dtype=torch.long).to(device)  # 标签迁移到 GPU

# 构造多状态图数据对象
data = Data(
    x=node_features,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=data_y
)

# 定义支持批量处理的模型
class GNNModel(torch.nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(node_input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_fc = Linear(hidden_dim * 2 + edge_input_dim, hidden_dim)  # 输入维度修正
        self.graph_fc = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attrs):
        outputs = []
        for edge_attr in edge_attrs:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)  # 保证 edge_attr 为 2D
            edge_start = x[edge_index[0]]
            edge_end = x[edge_index[1]]
            edge_features = torch.cat([edge_start, edge_end, edge_attr], dim=1)
            edge_output = F.relu(self.edge_fc(edge_features))
            graph_features = torch.mean(edge_output, dim=0, keepdim=True)
            outputs.append(self.graph_fc(graph_features))
        return torch.cat(outputs, dim=0)

# 初始化模型并迁移到 GPU
model = GNNModel(
    node_input_dim=8,
    edge_input_dim=1,
    hidden_dim=8,
    output_dim=len(label_mapping)
).to(device)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# 加载之前的模型（可选）
model_path = "../modelsByQ/gcn_model_continued_1000_10000.pth"  # 替换为实际模型路径
model.load_state_dict(torch.load(model_path, map_location=device))
print(f"Model loaded from {model_path}")

# 继续训练
additional_epochs = 10000
save_interval = 200  # 每 200 次保存一次
train_losses = []

for epoch in range(additional_epochs):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr.unsqueeze(-1))  # 输入图数据
    train_loss = criterion(out, data.y)  # 计算损失
    train_loss.backward()
    optimizer.step()  # 更新权重
    train_losses.append(train_loss.item())

    # 每 10 次打印一次训练损失
    if (epoch + 1) % 10 == 0:
        print(f"Continued Training Epoch {epoch + 1}/{additional_epochs}, Loss: {train_loss.item():.4f}")

    # 每 200 次保存模型到同一个文件
    if (epoch + 1) % save_interval == 0:
        torch.save(model.state_dict(), model_path)
        print(f"Model updated at epoch {epoch + 1} and saved to {model_path}")

# 保存最终模型
torch.save(model.state_dict(), model_path)
print(f"Final model saved to {model_path}")
