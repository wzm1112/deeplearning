import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
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

# 标签映射为连续整数
label_mapping = {label: idx for idx, label in enumerate(sorted(ventilation_data['ei'].unique()))}
ventilation_data['ei'] = ventilation_data['ei'].map(label_mapping)
print("Updated labels:", ventilation_data['ei'].unique())

# 固定边索引
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 2, 4],  # 起点索引
    [1, 2, 3, 3, 4, 5, 5]   # 终点索引
], dtype=torch.long)

# 构建多状态图数据对象
num_states = X.shape[0]  # 状态数量等于行数
edge_attr = torch.tensor(X.values, dtype=torch.float)  # 所有状态的边特征
node_features = torch.ones((6, 8))  # 假设节点特征固定为 6 个节点，每节点 8 维特征

# 构造多状态图数据对象
data = Data(
    x=node_features,           # 节点特征
    edge_index=edge_index,     # 边索引
    edge_attr=edge_attr,       # 所有状态的边特征
    y=torch.tensor(ventilation_data['ei'].values, dtype=torch.long)  # 所有状态的标签
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

# 初始化模型
model = GNNModel(
    node_input_dim=8,              # 节点特征维度
    edge_input_dim=1,              # 边特征维度
    hidden_dim=8,                  # 隐藏层维度
    output_dim=len(label_mapping)  # 类别数
)

# 训练模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

epochs = 10000
losses = []

model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr.unsqueeze(-1))  # 保证 edge_attr 为 3D
    loss = criterion(out, data.y)  # 对每个状态的标签计算损失
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Epoch {epoch + 10}/{epochs}, Loss: {loss.item():.4f}")

# 可视化训练损失
plt.figure(figsize=(12, 6))
plt.plot(range(1, epochs + 1), losses, label="Training Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.show()

# 模型评估
model.eval()
with torch.no_grad():
    predictions = model(data.x, data.edge_index, data.edge_attr.unsqueeze(-1))
    predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
    true_labels = data.y.cpu().numpy()

# 生成混淆矩阵
cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_mapping.keys()))
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()

# 分类报告
report = classification_report(
    true_labels,
    predicted_labels,
    target_names=[str(label) for label in label_mapping.keys()]
)
print("Classification Report:")
print(report)
# 保存模型
model_save_path = "../modelsByQ/gcn_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

