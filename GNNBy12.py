import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# Step 1: 数据加载与清洗
file_path = 'datasets/es[17]-vs[12]-random-train.csv'  # 替换为实际路径
ventilation_data = pd.read_csv(file_path)
ventilation_data.fillna(ventilation_data.mean(), inplace=True)

# Step 2: 特征选择与标准化
selected_features = [f"q'{i+1}" for i in range(17)]
X = ventilation_data[selected_features]
y = ventilation_data['ei']  # 目标列

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=selected_features)

# Step 3: 构建图数据
node_features = torch.tensor(X.values, dtype=torch.float)
num_nodes = len(node_features)
edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes - 1)], dtype=torch.long).t()
edge_attr = torch.tensor(X.iloc[:len(edge_index[0]), :2].values, dtype=torch.float)
labels = torch.tensor(y.values - 2, dtype=torch.long)

# Step 4: 划分训练集和测试集
num_labels = labels.size(0)
train_idx, test_idx = train_test_split(np.arange(num_labels), test_size=0.2, random_state=42)

train_idx_set = set(train_idx)
test_idx_set = set(test_idx)

train_mask = [(i in train_idx_set and j in train_idx_set) for i, j in zip(edge_index[0].numpy(), edge_index[1].numpy())]
test_mask = [(i in test_idx_set and j in test_idx_set) for i, j in zip(edge_index[0].numpy(), edge_index[1].numpy())]

train_edge_index = edge_index[:, train_mask]
test_edge_index = edge_index[:, test_mask]

train_edge_attr = edge_attr[train_mask]
test_edge_attr = edge_attr[test_mask]

# 修正索引偏移
train_idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(train_idx)}
test_idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(test_idx)}

train_edge_index = torch.tensor(
    [[train_idx_mapping[i.item()] for i in train_edge_index[0]],
     [train_idx_mapping[i.item()] for i in train_edge_index[1]]],
    dtype=torch.long,
)
test_edge_index = torch.tensor(
    [[test_idx_mapping[i.item()] for i in test_edge_index[0]],
     [test_idx_mapping[i.item()] for i in test_edge_index[1]]],
    dtype=torch.long,
)

train_data = Data(
    x=node_features[train_idx],
    edge_index=train_edge_index,
    edge_attr=train_edge_attr,
    y=labels[train_idx],
)

test_data = Data(
    x=node_features[test_idx],
    edge_index=test_edge_index,
    edge_attr=test_edge_attr,
    y=labels[test_idx],
)

# Step 5: 类别权重
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels.numpy()), y=labels.numpy())
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(f"Class Weights: {class_weights_tensor}")

# Step 6: 定义 GNN 模型
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
input_dim = train_data.x.size(1)
hidden_dim = 64
num_classes = len(torch.unique(labels))
model = GNNModel(input_dim, train_data.edge_attr.size(1), hidden_dim, num_classes)

# Step 7: 优化器与损失函数
optimizer = Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

# Step 8: 模型训练
epochs = 1000
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
    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Step 9: 模型评估
model.eval()
with torch.no_grad():
    train_pred = torch.argmax(model(train_data), dim=1)
    test_pred = torch.argmax(model(test_data), dim=1)

    print("\nTrain Classification Report:")
    print(classification_report(train_data.y.numpy(), train_pred.numpy()))
    print("\nTest Classification Report:")
    print(classification_report(test_data.y.numpy(), test_pred.numpy()))

# 绘制训练损失曲线
plt.figure(figsize=(12, 6))
plt.plot(losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()

# 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay.from_predictions(
    test_data.y.numpy(),
    test_pred.numpy(),
    cmap='viridis',
    colorbar=True,
    ax=ax
)
plt.title("Confusion Matrix: Test Data")
plt.show()
