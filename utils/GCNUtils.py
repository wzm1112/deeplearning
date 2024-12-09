import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt


def load_and_process_data(file_path, selected_features, label_column):
    """
    加载并预处理数据，包括填充缺失值、特征标准化和标签映射。
    """
    # 数据加载与清洗
    data = pd.read_csv(file_path)
    data.fillna(data.mean(), inplace=True)

    # 特征选择与标准化
    X = data[selected_features]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=selected_features)

    # 标签映射为连续整数
    label_mapping = {label: idx for idx, label in enumerate(sorted(data[label_column].unique()))}
    data[label_column] = data[label_column].map(label_mapping)

    return data, X, label_mapping


def build_graph_data(X, labels, edge_index, node_feature_dim):
    """
    构建图数据对象，包括节点特征、边特征和标签。
    """
    num_states = X.shape[0]
    edge_attr = torch.tensor(X.values, dtype=torch.float)
    node_features = torch.ones((edge_index.max().item() + 1, node_feature_dim))

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(labels, dtype=torch.long)
    )


class GNNModel(torch.nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, output_dim):
        """
        定义支持批量处理的 GNN 模型，包括两层图卷积和全连接层。
        """
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


def train_gnn_model(model, data, epochs=1000, learning_rate=0.001):
    """
    训练 GNN 模型并返回训练损失。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr.unsqueeze(-1))
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    return losses


def evaluate_gnn_model(model, data, label_mapping):
    """
    评估 GNN 模型，包括生成混淆矩阵和分类报告。
    """
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index, data.edge_attr.unsqueeze(-1))
        predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
        true_labels = data.y.cpu().numpy()

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
        target_names=[str(label) for label in label_mapping.keys()]
    )
    print("Classification Report:")
    print(report)

def plot_training_loss(losses):
    """
    绘制训练损失曲线。
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(losses) + 1), losses, label="Training Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


def save_model(model, path):
    """
    保存训练好的模型。
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")



