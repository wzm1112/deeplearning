import sys
from pathlib import Path

# 动态添加 utils 目录到模块搜索路径
utils_path = Path(__file__).resolve().parent.parent / "utils"
sys.path.append(str(utils_path))

from GCNUtils import (
    load_and_process_data,
    build_graph_data,
    GNNModel,
    train_gnn_model,
    plot_training_loss,
    evaluate_gnn_model,
    save_model
)
import torch

def main():
    # 配置参数
    file_path = '../datasets/es[7]-vs[6]-random-train.csv'  # 替换为实际数据路径
    selected_features = ["h'1", "h'2", "h'3", "h'4", "h'5", "h'6", "h'7"]  # 输入特征
    label_column = 'ei'  # 输出目标列
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 2, 4],  # 起点索引
        [1, 2, 3, 3, 4, 5, 5]   # 终点索引
    ], dtype=torch.long)
    node_feature_dim = 8  # 节点特征维度
    hidden_dim = 8  # 隐藏层维度
    additional_epochs = 5                                                                                                        # 总训练轮数
    update_interval = 500  # 每 500 次更新模型
    initial_model_path = ("../GCNByH/gcn_model.pth")  # 初始模型路径
    updated_model_path = "../GCNByH/gcn_model.pth"  # 更新模型保存路径

    # 加载数据并构建图数据对象
    ventilation_data, X, label_mapping = load_and_process_data(file_path, selected_features, label_column)
    data = build_graph_data(X, ventilation_data[label_column].values, edge_index, node_feature_dim)

    # 初始化模型
    model = GNNModel(
        node_input_dim=node_feature_dim,
        edge_input_dim=1,
        hidden_dim=hidden_dim,
        output_dim=len(label_mapping)
    )

    # 加载之前保存的模型
    try:
        model.load_state_dict(torch.load(initial_model_path))
        print(f"Model loaded from {initial_model_path}")
    except FileNotFoundError:
        print(f"No existing model found at {initial_model_path}. Starting fresh training.")

    # 继续训练模型，每 500 次保存一次
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    model.train()
    for epoch in range(1, additional_epochs + 1):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr.unsqueeze(-1))
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{additional_epochs}, Loss: {loss.item():.4f}")

        # 每 500 次保存一次模型
        if epoch % update_interval == 0:
            save_model(model, updated_model_path)
            print(f"Model updated and saved at epoch {epoch} to {updated_model_path}")

    # 保存最终模型
    save_model(model, updated_model_path)
    print(f"Final model saved to {updated_model_path}")

    # 绘制损失曲线
    plot_training_loss(losses)

    # 模型评估
    evaluate_gnn_model(model, data, label_mapping)

if __name__ == "__main__":
    main()
