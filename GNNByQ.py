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
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt

# Step 1: 数据加载与清洗
file_path = 'datasets/es[17]-vs[12]-random-train.csv'  # 替换为实际路径
ventilation_data = pd.read_csv(file_path)
ventilation_data.fillna(ventilation_data.mean(), inplace=True)

# Step 2: （已移除数据增强）

# Step 3: 特征选择与标准化
selected_features = [f"q'{i+1}" for i in range(17)]

X = ventilation_data[selected_features]
y = ventilation_data['ei']  # 目标列

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

# Step 5: 类别权重
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(data.y.numpy()), y=data.y.numpy())
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(f"Class Weights: {class_weights_tensor}")

# Step 6: 定义 GNN 模型
class GNNModel(torch.nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(node_input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)  # 新增一层
        self.conv4 = GCNConv(hidden_dim, hidden_dim)  # 添加更多 GCN 层

        self.edge_fc = Linear(edge_input_dim + hidden_dim * 2, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv3(x, edge_index))  # 新增层的前向传播
        edge_start = x[edge_index[0]]
        edge_end = x[edge_index[1]]
        edge_features = torch.cat([edge_start, edge_end, edge_attr], dim=1)
        edge_output = self.edge_fc(edge_features)
        return F.log_softmax(edge_output, dim=1)

# 初始化模型
input_dim = data.x.size(1)
hidden_dim = 64  # 增加隐藏层维度
num_classes = len(torch.unique(data.y))
model = GNNModel(input_dim, data.edge_attr.size(1), hidden_dim, num_classes)

# Step 7: 优化器与损失函数
optimizer = Adam(model.parameters(), lr=0.001)
# criterion = torch.nn.NLLLoss(weight=class_weights_tensor)
# 换一个损失函数
criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)  # 每 100 轮减小学习率一半

# Step 8: 模型训练
epochs = 1000  # 增加训练次数
losses = []
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    scheduler.step()  # 更新学习率
    losses.append(loss.item())
    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Step 9: 模型评估
model.eval()
with torch.no_grad():
    pred = torch.argmax(model(data), dim=1)
    print("\nClassification Report:")
    print(classification_report(data.y.numpy(), pred.numpy(), target_names=[str(i) for i in range(2, 17)]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(data.y.numpy(), pred.numpy()))

# 绘制训练损失曲线
plt.figure(figsize=(12, 6))
plt.plot(losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()

# 绘制混淆矩阵
categories = [{i+2} for i in range(15)]
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay.from_predictions(
    data.y.numpy(),
    pred.numpy(),
    display_labels=categories,
    cmap='viridis',
    colorbar=True,
    ax=ax
)
plt.title("Confusion Matrix: True Labels vs Predicted Labels")
plt.show()

# 绘制测量标签与实际标签的对比条形图
true_labels_count = np.bincount(data.y.numpy(), minlength=len(categories))
predicted_labels_count = np.bincount(pred.numpy(), minlength=len(categories))
x = np.arange(len(categories))  # 分类索引
width = 0.35  # 柱状图宽度

fig, ax = plt.subplots(figsize=(12, 6))
bar1 = ax.bar(x - width/2, true_labels_count, width, label='True Labels')
bar2 = ax.bar(x + width/2, predicted_labels_count, width, label='Predicted Labels')

ax.set_xlabel('Categories')
ax.set_ylabel('Count')
ax.set_title('Comparison of True and Predicted Labels')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

for bar in bar1 + bar2:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.show()
