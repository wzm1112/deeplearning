import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample, class_weight
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Step 1: 数据加载与清洗
file_path = '/mnt/data/es[7]-vs[6]-random-train.csv'  # 替换为实际路径
ventilation_data = pd.read_csv(file_path)
ventilation_data.fillna(ventilation_data.mean(), inplace=True)

# Step 2: 数据增强（过采样类别 4）
data_class_4 = ventilation_data[ventilation_data['ei'] == 4]
data_other = ventilation_data[ventilation_data['ei'] != 4]
data_class_4_upsampled = resample(data_class_4, replace=True, n_samples=len(data_other), random_state=42)
balanced_data = pd.concat([data_other, data_class_4_upsampled])
print(f"Balanced Target Value Distribution:\n{balanced_data['ei'].value_counts()}")

# Step 3: 特征选择与标准化
selected_features = [
    'deltaq1', 'deltaq2', 'deltaq3', 'deltaq4', 'deltaq5', 'deltaq6', 'deltaq7',
    'deltap1', 'deltap2', 'deltap3', 'deltap4', 'deltap5', 'deltap6',
    'deltah1', 'deltah2', 'deltah3', 'deltah4', 'deltah5', 'deltah6', 'deltah7',
    'delta_r0'
]
X = balanced_data[selected_features]
y = balanced_data['ei']  # 目标列

# 标准化特征
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=selected_features)

# Step 4: 构建图数据
node_features = torch.tensor(X.values, dtype=torch.float)
num_nodes = len(node_features)
edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes - 1)], dtype=torch.long).t()
edge_labels = torch.tensor(y.iloc[:len(edge_index[0])].values - 2, dtype=torch.long)
edge_attr = torch.tensor(X.iloc[:len(edge_index[0]), :2].values, dtype=torch.float)
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=edge_labels)

# 检查 data.y 的大小以确认索引有效性
print(f"Data y size: {data.y.size()}")

# Step 5: 数据集划分
valid_indices = np.arange(len(data.y))  # 确保索引范围在 data.y 的大小内
train_indices, test_indices = train_test_split(
    valid_indices, test_size=0.2, random_state=42
)

# 检查 train_indices 和 test_indices 的范围
print(f"Train indices range: {train_indices.min()} to {train_indices.max()}")
print(f"Test indices range: {test_indices.min()} to {test_indices.max()}")


# 映射 edge_index 到局部索引范围的函数
def map_edge_index(edge_index, valid_indices):
    node_map = {global_idx: local_idx for local_idx, global_idx in enumerate(valid_indices)}
    mapped_edge_index = torch.zeros_like(edge_index)
    valid_edges = []

    for i in range(edge_index.size(1)):
        start, end = edge_index[0, i].item(), edge_index[1, i].item()
        if start in node_map and end in node_map:
            mapped_edge_index[0, i] = node_map[start]
            mapped_edge_index[1, i] = node_map[end]
            valid_edges.append(i)

    # 过滤掉未映射的边
    mapped_edge_index = mapped_edge_index[:, valid_edges]
    print(f"Filtered edge_index size: {mapped_edge_index.size()}")

    return mapped_edge_index


# 构建训练数据集
train_edge_index = map_edge_index(data.edge_index, train_indices)
train_data = Data(
    x=data.x[train_indices],
    edge_index=train_edge_index,
    edge_attr=data.edge_attr[:train_edge_index.size(1)],
    y=data.y[train_indices]
)

# 构建测试数据集
test_edge_index = map_edge_index(data.edge_index, test_indices)
test_data = Data(
    x=data.x[test_indices],
    edge_index=test_edge_index,
    edge_attr=data.edge_attr[:test_edge_index.size(1)],
    y=data.y[test_indices]
)

# 检查训练和测试数据集的节点数量
print(f"Train data nodes: {train_data.x.size(0)}")
print(f"Test data nodes: {test_data.x.size(0)}")

# Step 6: 类别权重
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(data.y.numpy()), y=data.y.numpy())
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(f"Class Weights: {class_weights_tensor}")


# Step 7: 定义 GNN 模型
class GNNModel(torch.nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(node_input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.edge_fc = Linear(edge_input_dim + hidden_dim * 2, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        edge_start = x[edge_index[0]]
        edge_end = x[edge_index[1]]
        edge_features = torch.cat([edge_start, edge_end, edge_attr], dim=1)
        edge_output = self.edge_fc(edge_features)
        return F.log_softmax(edge_output, dim=1)


# 初始化模型
input_dim = data.x.size(1)
hidden_dim = 32
num_classes = len(torch.unique(data.y))
model = GNNModel(input_dim, data.edge_attr.size(1), hidden_dim, num_classes)

# Step 8: 优化器与损失函数
optimizer = Adam(model.parameters(), lr=0.0005)
criterion = torch.nn.NLLLoss(weight=class_weights_tensor)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

# Step 9: 模型训练
epochs = 500
losses = []
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(train_data)
    loss = criterion(out, train_data.y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Step 10: 模型评估
model.eval()
with torch.no_grad():
    pred = torch.argmax(model(test_data), dim=1)
    print("\nClassification Report:")

