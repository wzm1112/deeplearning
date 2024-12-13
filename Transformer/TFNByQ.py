import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Transformer 模型定义
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers=2):
        super(TransformerModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nhead = nhead
        self.num_layers = num_layers

        # 确保输入维度可以被 nhead 整除
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
        self.fc = nn.Linear(hidden_dim, output_dim)  # 输出到目标类别

    def forward(self, x):
        # 输入 x 的形状为 (batch_size, seq_len, input_dim)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, input_dim) -> (seq_len, batch_size, input_dim)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # 选择最后一个时间步的输出作为特征
        x = x.mean(dim=0)  # 按时间步求均值，(seq_len, batch_size, hidden_dim) -> (batch_size, hidden_dim)



        # 将变换后的结果输入到全连接层
        x = self.fc(x)  # 映射到输出类别
        return x


# 加载并处理数据
file_path = '../datasets/es[7]-vs[6]-random-train.csv'  # 替换为你的实际数据路径
data = pd.read_csv(file_path)

# 假设选择一些特征和目标
selected_features = ["q'1", "q'2", "q'3", "q'4", "q'5", "q'6", "q'7"]
target_columns = ["ei"]  # 假设目标是多分类问题的标签

# 特征选择与标准化
X = data[selected_features]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=selected_features)

# 目标变量（这里假设目标是多分类问题）
y = data[target_columns]

# 确保标签为整数类型（用于多分类任务）
y = y.astype(int)

# 转换为 PyTorch Tensor
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values.flatten(), dtype=torch.long)  # CrossEntropyLoss 需要标签是 long 类型

# 增加额外的维度以适应 Transformer 输入
X_tensor = X_tensor.unsqueeze(-1)  # 转换为 (batch_size, seq_len, features)

# 创建模型
model = TransformerModel(
    input_dim=1,  # 获取特征维度
    hidden_dim=1,
    output_dim=7,  # 根据标签的种类数来设置输出维度
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
    y_pred = model(X_tensor)

    # 获取预测的类别
    _, predicted_classes = torch.max(y_pred, 1)

# 打印结果
print("预测结果: ", predicted_classes[:5])
print("真实值: ", y_tensor[:5])

# 计算准确率
correct = (predicted_classes == y_tensor).sum().item()
accuracy = correct / len(y_tensor)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
