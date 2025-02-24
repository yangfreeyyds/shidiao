#验证脚本
#freeyyds

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from snownlp import SnowNLP
import seaborn as sns
import matplotlib.pyplot as plt

# 解决乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def validate(ground_truth_file='labeled_data.csv'):
    """验证情感分析准确率"""
    df = pd.read_csv(ground_truth_file)
    true_labels = df['label'].values
    preds = []

    for text in df['text']:
        try:
            score = SnowNLP(text).sentiments
            pred = 1 if score > 0.6 else (0 if score < 0.4 else 2)
        except:
            pred = 2
        preds.append(pred)

    # 过滤中性结果
    mask = [p != 2 for p in preds]
    filtered_true = true_labels[mask]
    filtered_pred = [p for p in preds if p != 2]

    # 输出报告
    print(f"准确率: {accuracy_score(filtered_true, filtered_pred):.2%}")
    print(classification_report(filtered_true, filtered_pred))

    # 绘制混淆矩阵
    cm = pd.crosstab(filtered_true, filtered_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig('confusion_matrix.png', dpi=300)


if __name__ == '__main__':
    validate()
