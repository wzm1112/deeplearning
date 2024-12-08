from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from model.model_fun import generate_lstm_model

# 数据加载
dataset = read_csv('datasets/generated_time_series_with_anomalies.csv', header=0)
# 选择特征列（以 Q_edge1 和 P_edge1 为例）
selected_features = ["Q_edge1", "P_edge1", "Anomaly_edge1"]
values = dataset[selected_features].values

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# 设置时间步长和预测目标
n_hours = 10  # 使用过去 10 个时间步
n_features = len(selected_features)  # 使用所有选定特征
n_predict = 1  # 预测未来 1 个时间步

# 构造监督学习数据
def series_to_supervised(data, n_in, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'var{j+1}(t-{i})') for j in range(n_vars)]
    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(f'var{j+1}(t)') for j in range(n_vars)]
        else:
            names += [(f'var{j+1}(t+{i})') for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

reframed = series_to_supervised(scaled, n_hours, n_predict)

# 切分训练集和测试集
values = reframed.values
train_size = int(len(values) * 0.6)
val_size = int(len(values) * 0.2)
train = values[:train_size, :]
val = values[train_size:train_size + val_size, :]
test = values[train_size + val_size:, :]

# 提取输入输出
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features:]
val_X, val_y = val[:, :n_obs], val[:, -n_features:]
test_X, test_y = test[:, :n_obs], test[:, -n_features:]

# 转换为LSTM需要的3D输入格式
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
val_X = val_X.reshape((val_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

# 设计LSTM模型
model = generate_lstm_model(n_input=n_hours, n_out=n_features, n_features=n_features)
callbacks = EarlyStopping(monitor='val_loss', patience=5)
model.compile(optimizer='adam', loss='mse')

# 训练模型
history = model.fit(train_X, train_y, epochs=50, batch_size=32, validation_data=(val_X, val_y), callbacks=[callbacks], verbose=2)

# 预测
y_pred = model.predict(test_X)

# 逆归一化结果
y_pred_original = scaler.inverse_transform(y_pred)
test_y_original = scaler.inverse_transform(test_y)

# 绘制预测结果
plt.figure(figsize=(12, 6))
plt.plot(test_y_original[:, 0], label="True Q_edge1", linestyle='-')
plt.plot(y_pred_original[:, 0], label="Predicted Q_edge1", linestyle='--')
plt.title("LSTM Prediction of Q_edge1")
plt.xlabel("Time")
plt.ylabel("Flow Rate")
plt.legend()
plt.show()
