from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.load_data import series_to_supervised

# 数据加载
dataset = read_csv('datasets/generated_time_series_with_anomalies.csv', header=0)

# 特征选择和目标选择
selected_features = [f"{metric}_edge{edge}" for edge in range(1, 8) for metric in ["Q", "P", "R"]]
values = dataset[selected_features].values
target_columns = [f"Anomaly_edge{edge}" for edge in range(1, 8)]
targets = dataset[target_columns].values

# 将异常节点编码为分类标签
def encode_anomalies(targets):
    labels = []
    for row in targets:
        if np.any(row == 1):
            labels.append(np.argmax(row) + 1)
        else:
            labels.append(0)
    return np.array(labels)

labels = encode_anomalies(targets)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# 设置时间窗口
n_hours = 10
n_features = len(selected_features)
n_classes = len(target_columns) + 1

# 构造监督学习数据
reframed = series_to_supervised(scaled, n_hours, 1)
reframed_labels = labels[n_hours:]

# 数据集划分
values = reframed.values
train_size = int(len(reframed) * 0.6)
val_size = int(len(reframed) * 0.2)
train_X, train_y = values[:train_size, :], reframed_labels[:train_size]
val_X, val_y = values[train_size:train_size + val_size, :], reframed_labels[train_size:train_size + val_size]
test_X, test_y = values[train_size + val_size:, :], reframed_labels[train_size + val_size:]

# 调整和重塑数据
def adjust_and_reshape(data, n_hours):
    n_samples = data.shape[0]
    total_features = data.shape[1]

    if total_features % n_hours != 0:
        adjusted_columns = (total_features // n_hours) * n_hours
        data = data[:, :adjusted_columns]

    n_real_features = data.shape[1] // n_hours
    return data.reshape((n_samples, n_hours, n_real_features))

train_X = adjust_and_reshape(train_X, n_hours)
val_X = adjust_and_reshape(val_X, n_hours)
test_X = adjust_and_reshape(test_X, n_hours)

n_features = train_X.shape[2]
train_y = to_categorical(train_y, num_classes=n_classes)
val_y = to_categorical(val_y, num_classes=n_classes)
test_y = to_categorical(test_y, num_classes=n_classes)

# 计算类别权重
class_weights = compute_class_weight('balanced', classes=np.unique(reframed_labels), y=reframed_labels)
class_weights_dict = dict(enumerate(class_weights))

# 设计双向 LSTM 模型
model = Sequential([
    Bidirectional(LSTM(100, activation='relu', return_sequences=True), input_shape=(n_hours, n_features)),
    Dropout(0.3),
    LSTM(50, activation='relu'),
    Dropout(0.3),
    Dense(n_classes, activation='softmax')
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# 模型训练
callbacks = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(
    train_X, train_y,
    epochs=200,  # 增加迭代次数
    batch_size=32,
    validation_data=(val_X, val_y),
    class_weight=class_weights_dict,
    callbacks=[callbacks],
    verbose=2
)

# 模型评估
loss, accuracy = model.evaluate(test_X, test_y, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 预测故障节点
y_pred = model.predict(test_X)
y_pred_labels = np.argmax(y_pred, axis=1)
test_labels = np.argmax(test_y, axis=1)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(test_labels, label="True Labels", linestyle='-')
plt.plot(y_pred_labels, label="Predicted Labels", linestyle='--')
plt.title("Fault Detection Using BiLSTM")
plt.xlabel("Time Step")
plt.ylabel("Node Class")
plt.legend()
plt.show()
