# sentiment_analysis.py
import os
import sys
import warnings
from snownlp import SnowNLP, sentiment
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import preprocess_text
from data import comments

# 配置环境
os.environ['PYTHONUTF8'] = '1'  # 启用UTF-8编码模式
pd.set_option('display.unicode.east_asian_width', True)  # 优化中文显示


# 自定义异常
class InsufficientDataError(Exception):
    """标注数据不足异常"""

    def __init__(self, positive, negative):
        msg = f"正样本: {len(positive)}, 负样本: {len(negative)} - 需要至少各2条"
        super().__init__(msg)


class ModelSaveError(Exception):
    """模型保存失败异常"""
    pass


def safe_path_creation(path):
    """安全创建文件路径"""
    try:
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            print(f"创建目录: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)

        # 验证目录可写
        test_file = os.path.join(dir_path, 'write_test.tmp')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        return True
    except Exception as e:
        print(f"路径创建失败: {str(e)}")
        return False


def load_labeled_data():
    """加载并预处理标注数据"""
    positive, negative = [], []

    for text, label in comments:
        try:
            cleaned = preprocess_text(text).strip()
            if len(cleaned) < 2:
                continue

            if label == 1:
                positive.append((cleaned, 1))
            elif label == -1:
                negative.append((cleaned, 0))
        except Exception as e:
            print(f"数据预处理失败: {text[:20]}... | 错误: {str(e)}")

    print("\n[数据统计]")
    print(f"有效正面: {len(positive):>4}条")
    print(f"有效负面: {len(negative):>4}条")
    print(f"正负比例: {len(positive) / len(negative):.2f}:1")

    return positive, negative


def create_balanced_split(positive, negative, test_size=0.8):
    """创建平衡的数据划分"""
    min_test = 2
    pos_test = max(min_test, int(len(positive) * test_size))
    neg_test = max(min_test, int(len(negative) * test_size))

    if len(positive) < pos_test or len(negative) < neg_test:
        raise InsufficientDataError(positive, negative)

    pos_train, pos_test = train_test_split(positive, test_size=pos_test, random_state=42)
    neg_train, neg_test = train_test_split(negative, test_size=neg_test, random_state=42)

    # 合并并打乱
    train = pd.DataFrame(pos_train + neg_train).sample(frac=1).values.tolist()
    test = pd.DataFrame(pos_test + neg_test).sample(frac=1).values.tolist()

    print("\n[数据划分]")
    print(f"训练集: {len(train)}条 (正:{len(pos_train)} 负:{len(neg_train)})")
    print(f"测试集: {len(test)}条 (正:{len(pos_test)} 负:{len(neg_test)})")

    return train, test


def train_sentiment_model(model_path):
    """执行模型训练"""
    try:
        print(f"\n[训练启动] 目标模型路径: {os.path.abspath(model_path)}")

        # 路径安全验证
        if not safe_path_creation(model_path):
            raise ModelSaveError("路径不可用")

        # 数据准备
        positive, negative = load_labeled_data()
        train_data, test_data = create_balanced_split(positive, negative)

        # 生成临时训练文件
        temp_file = os.path.join(os.path.dirname(model_path), "TEMP_train.txt")
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                for text, label in train_data:
                    f.write(f"{text}___LABEL_{label}\n")

            # 执行训练
            print("\n[训练日志]")
            sentiment.train(temp_file, model_path)

            # 验证模型文件
            if not os.path.isfile(model_path):
                raise ModelSaveError("模型文件未生成")
            print(f"模型保存成功 ({os.path.getsize(model_path) / 1024:.1f}KB)")

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        return test_data

    except Exception as e:
        print(f"\n[训练异常] {str(e)}")
        if os.path.exists(model_path):
            os.remove(model_path)
        raise


def evaluate_model(test_data, model_path):
    """模型性能评估"""
    try:
        sentiment.load(model_path)
        y_true, y_pred = [], []

        for text, label in test_data:
            try:
                s = SnowNLP(text)
                pred = 1 if s.sentiments > 0.5 else 0
            except:
                pred = 0
            y_true.append(label)
            y_pred.append(pred)

        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        print("\n[评估结果]")
        print(f"准确率: {accuracy:.2%}")
        print("混淆矩阵:")
        print(pd.crosstab(pd.Series(y_true, name='实际'),
                          pd.Series(y_pred, name='预测'),
                          margins=True))

    except Exception as e:
        print(f"\n[评估失败] {str(e)}")
        raise


if __name__ == '__main__':
    # 配置参数
    MODEL_PATH = os.path.expanduser('~/sentiment_model/sentiment.marshal')  # 推荐路径

    try:
        # 执行训练
        test_set = train_sentiment_model(MODEL_PATH)

        # 评估模型
        evaluate_model(test_set, MODEL_PATH)

    except Exception as e:
        print(f"\n[主程序异常] {str(e)}")
        sys.exit(1)

    finally:
        print("\n[程序执行结束]")
