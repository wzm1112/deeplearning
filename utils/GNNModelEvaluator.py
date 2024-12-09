import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np


class GNNModelEvaluator:
    def __init__(self, model, data, label_mapping):
        """
        初始化评估工具类。

        Args:
            model (torch.nn.Module): 已训练的GNN模型。
            data (torch_geometric.data.Data): 包含图结构和标签的图数据对象。
            label_mapping (dict): 标签到索引的映射字典。
        """
        self.model = model
        self.data = data
        self.label_mapping = label_mapping

    def evaluate(self):
        """
        评估模型的性能，包括准确率、混淆矩阵和分类报告。
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.data.x, self.data.edge_index, self.data.edge_attr.unsqueeze(-1))
            predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
            true_labels = self.data.y.cpu().numpy()

        # 计算准确率
        accuracy = self.calculate_accuracy(predicted_labels, true_labels)
        print(f"Model Accuracy: {accuracy:.4f}")

        # 显示混淆矩阵
        self.plot_confusion_matrix(true_labels, predicted_labels)

        # 打印分类报告
        self.print_classification_report(true_labels, predicted_labels)

    @staticmethod
    def calculate_accuracy(predicted_labels, true_labels):
        """
        计算准确率。

        Args:
            predicted_labels (numpy.ndarray): 模型预测的标签。
            true_labels (numpy.ndarray): 真实标签。

        Returns:
            float: 准确率。
        """
        return np.mean(predicted_labels == true_labels)

    def plot_confusion_matrix(self, true_labels, predicted_labels):
        """
        绘制混淆矩阵。

        Args:
            true_labels (numpy.ndarray): 真实标签。
            predicted_labels (numpy.ndarray): 模型预测的标签。
        """
        cm = confusion_matrix(true_labels, predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(self.label_mapping.keys()))
        disp.plot(cmap='Blues', xticks_rotation='vertical')
        plt.title("Confusion Matrix")
        plt.show()

    def print_classification_report(self, true_labels, predicted_labels):
        """
        打印分类报告。

        Args:
            true_labels (numpy.ndarray): 真实标签。
            predicted_labels (numpy.ndarray): 模型预测的标签。
        """
        report = classification_report(
            true_labels,
            predicted_labels,
            target_names=[str(label) for label in self.label_mapping.keys()]
        )
        print("Classification Report:")
        print(report)
