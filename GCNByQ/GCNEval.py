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

# 检查设备
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
], dtype=torch.long).to(device)

# 构建多状态图数据对象
num_states = X.shape[0]
edge_attr = torch.tensor(X.values, dtype=torch.float).to(device)
node_features = torch.ones((6, 8)).to(device)
data_y = torch.tensor(ventilation_data['ei'].values, dtype=torch.long).to(device)

# 构造多状态图数据对象
data = Data(
    x=node_features,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=data_y
)

# 定义模型
class GNNModel(torch.nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(node_input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_fc = Linear(hidden_dim * 2 + edge_input_dim, hidden_dim)
        self.graph_fc = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attrs):
        outputs = []
        for edge_attr in edge_attrs:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)
            edge_start = x[edge_index[0]]
            edge_end = x[edge_index[1]]
            edge_features = torch.cat([edge_start, edge_end, edge_attr], dim=1)
            edge_output = F.relu(self.edge_fc(edge_features))
            graph_features = torch.mean(edge_output, dim=0, keepdim=True)
            outputs.append(self.graph_fc(graph_features))
        return torch.cat(outputs, dim=0)

# 加载模型
model_path = "../modelsByQ/gcn_model_continued_1000_10000.pth"  # 替换为模型路径
model = GNNModel(
    node_input_dim=8,
    edge_input_dim=1,
    hidden_dim=8,
    output_dim=len(label_mapping)
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Model loaded from {model_path}")

# 模型测试
with torch.no_grad():
    predictions = model(data.x, data.edge_index, data.edge_attr.unsqueeze(-1))
    predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
    true_labels = data.y.cpu().numpy()

# 计算准确率
accuracy = (predicted_labels == true_labels).sum() / len(true_labels)
print(f"Model Accuracy: {accuracy:.4f}")

# 混淆矩阵
cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_mapping.keys()))
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()

# 分类报告
report = classification_report(
    true_labels,
    predicted_labels,
    target_names=[str(label) for label in label_mapping.keys()],
    zero_division=0
)
print("Classification Report:")
print(report)

# 打印部分预测结果
print("Sample Predictions:")
for i in range(min(1000, len(true_labels))):  # 打印前 10 条预测
    print(f"True: {true_labels[i]}, Predicted: {predicted_labels[i]}")
