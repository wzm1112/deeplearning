import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Transformer 模型定义
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers=2):
        super(TransformerModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nhead = nhead
        self.num_layers = num_layers

        if input_dim % nhead != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by nhead ({nhead})")

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x

# 加载数据并处理
file_path = '../../datasets/es[17]-vs[12]-random-train.csv'
data = pd.read_csv(file_path)

# 特征选择与标准化
selected_features = [f"h'{i+1}" for i in range(17)]
target_columns = ["ei"]
X = data[selected_features]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=selected_features)

# 目标变量
y = data[target_columns].astype(int)

X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

X_tensor = X_tensor.unsqueeze(1)
print(X_tensor.shape)

# 创建模型
model = TransformerModel(input_dim=17, hidden_dim=32, output_dim=17, nhead=1)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 数据加载器
batch_size = 32
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练模型
num_epochs = 400
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in data_loader:
        optimizer.zero_grad()
        batch_y = batch_y.squeeze()  # 将目标张量转为一维


        # 前向传播
        y_pred = model(batch_X)

        # 计算损失
        loss = criterion(y_pred, batch_y)

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data_loader):.4f}")

# 保存模型
torch.save(model.state_dict(), "transformer_model_byH_17.pth")
print("Model saved successfully!")

# 评估模型
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor)
    _, predicted = torch.max(y_pred, 1)



# 生成分类报告
from sklearn.metrics import classification_report, confusion_matrix
print("Classification Report:")
print(classification_report(y_tensor.numpy(), predicted.numpy()))
print("Confusion Matrix:")
print(confusion_matrix(y_tensor.numpy(), predicted.numpy()))

# 加载模型（用于后续使用）
model_loaded = TransformerModel(input_dim=17, hidden_dim=32, output_dim=17, nhead=1)
model_loaded.load_state_dict(torch.load("transformer_model_byH_17.pth"))
model_loaded.eval()
print("Model loaded successfully!")
