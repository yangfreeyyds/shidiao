import re
import jieba
from collections import Counter

from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# 在分词处理前添加
jieba.add_word('四个现代化', freq=1000, tag='n')  # 设置高词频确保优先识别

# 自定义停用词列表（可根据需要补充）
STOPWORDS = {
    '的', '了', '和', '是', '在', '我', '有', '就', '也', '不', '都', '还',
    '这个', '一个', '吗', '要', '说', '很', '没有', '但', '我们', '好',
    '这', '那', '你', '我', '他', '她', '它', '啊', '哦', '嗯', '吧','捂脸','这么','这是','还有','这里','什么 ','怎么','不是','这是','还是','四个','一个月','666',
'就是',    # 排名7 (113次) 连接词
    '感觉',    # 排名18 (65次) 模糊情感词   # 排名19 (63次) 助动词
    '看到',    # 排名20 (62次) 通用动词
    '这样',    # 排名22 (58次) 指示代词
    '应该',    # 排名24 (58次) 情态动词
    '真的',    # 排名34 (50次) 程度副词
    '什么',    # 排名39 (45次) 疑问代词
    '现在',    # 排名40 (45次) 时间副词
    '地方',     # 排名28 (53次) 泛指名词
'画面',     # 排名15 (70次) 场景泛称，信息量低
    '知道',     # 排名32 (42次) 通用认知动词
    '场景',     # 排名33 (42次) 环境泛称
    '时候',     # 排名35 (41次) 时间副词
    '视频',     # 排名40 (36次) 媒介载体（若分析对象非媒体内容）   # 排名26 (51次) 时间状语
    '一块',   # 排名35 量词无具体指代
       '多少',   # 排名36 疑问代词
       '课本',   # 排名37 非常规主题词
       '一边',   # 排名38 方位副词
       '以为'    # 排名40 主观判断动词
}

def preprocess_text(text):
    """文本预处理函数"""
    # 去除特殊字符和表情
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    # 合并空格
    return re.sub(r'\s+', ' ', text).strip()

# 读取文件
input_file = 'main.txt'
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 分词处理
all_words = []
for line in lines:
    cleaned = preprocess_text(line.strip())
    if not cleaned:
        continue
    words = jieba.cut(cleaned, cut_all=False)
    # 过滤停用词和单字
    filtered_words = [
        word for word in words
        if word.strip() and
           len(word) > 1 and
           word not in STOPWORDS
    ]
    all_words.extend(filtered_words)

# 词频统计
word_counter = Counter(all_words)
top_40 = word_counter.most_common(150)

# 打印结果
print("【TOP40 关键词词频统计】")
print("{:<8}{:<10}{:<8}".format('排名', '关键词', '频次'))
print('-' * 30)
for idx, (word, count) in enumerate(top_40, 1):
    print("{:<10}{:<12}{:<8}".format(idx, word, count))

# 保存结果到文件
output_file = 'word_frequency.csv'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("关键词,出现次数\n")
    for word, count in top_40:
        f.write(f"{word},{count}\n")
print(f"\n结果已保存到 {output_file}")

# 在文件顶部添加新的导入
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# 在保存结果到文件之后添加以下代码（在print语句之前）
# 生成词云数据
word_freq_dict = dict(top_40)
from PIL import Image
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 在生成词云对象前添加蒙版配置
# 设置蒙版图片路径（需准备PNG格式形状图）
mask_path = "../数据分析代码/img_1.png"  # 例如：中国地图、圆形、动物等形状

try:
    # 加载蒙版图片并转换为数组
    mask_image = Image.open(mask_path)
    mask_array = np.array(mask_image)
    font_path = 'msyh.ttc'  # 微软雅黑字体，需下载放入程序目录
    # 修改词云配置
    wc = WordCloud(
        font_path=font_path,
        mask=mask_array,  # 添加蒙版参数
        background_color='white',
        max_words=10000,
        max_font_size=1000,
        contour_width=2,  # 轮廓线宽
        contour_color='steelblue',  # 轮廓颜色
        colormap='viridis'
    )

    # 生成词云（其余代码保持不变）
    wordcloud = wc.generate_from_frequencies(word_freq_dict)

except FileNotFoundError:
    print(f"蒙版图片{mask_path}未找到，将生成标准矩形词云")
    wordcloud = wc.generate_from_frequencies(word_freq_dict)
except Exception as e:
    print(f"蒙版加载失败：{str(e)}")
    wordcloud = wc.generate_from_frequencies(word_freq_dict)
# 配置词云参数

# 生成词云对象
wordcloud = wc.generate_from_frequencies(word_freq_dict)

# 保存词云图片
wordcloud.to_file('img.png')

# 可选：显示词云
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

print(f"词云图已保存至 wordcloud.png")

# 在文件顶部添加新导入
# 新增导入
from snownlp import SnowNLP

# 在预处理后添加情感分析代码
# 初始化统计变量
sentiment_stats = {
    'positive': 0,  # 积极（>0.6）
    'neutral': 0,  # 中性（0.4-0.6）
    'negative': 0  # 消极（<0.4）
}
detailed_sentiments = []

# 在原有循环中添加情感分析（建议在分词前处理）
for line in lines:
    cleaned = preprocess_text(line.strip())
    if not cleaned:
        continue

    # 情感分析（新增部分）
    try:
        s = SnowNLP(cleaned)
        score = s.sentiments
        detailed_sentiments.append(score)

        if score > 0.6:
            sentiment_stats['positive'] += 1
        elif score < 0.4:
            sentiment_stats['negative'] += 1
        else:
            sentiment_stats['neutral'] += 1
    except:
        continue

    # 原有分词处理代码
    words = jieba.cut(cleaned, cut_all=False)
    # ... 保持原有过滤逻辑 ...


# 新增情感分析结果输出
# 在文件顶部添加新导入
import seaborn as sns
import numpy as np


# 替换原来的print_sentiment_results函数
def print_sentiment_results(stats, scores):
    """绘制情感分布曲线图"""
    plt.figure(figsize=(10, 6))

    # 使用seaborn的kdeplot绘制平滑曲线
    ax = sns.kdeplot(scores,
                     bw_adjust=0.5,  # 带宽调整参数，越大越平滑
                     color='#2c7fb8',
                     linewidth=2.5,
                     fill=True,
                     alpha=0.3)

    # 设置坐标轴范围
    plt.xlim(0, 1)
    plt.ylim(0, None)

    # 添加辅助线
    plt.axvline(0.4, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(0.6, color='gray', linestyle='--', alpha=0.7)

    # 设置标签和标题
    plt.xlabel('情感得分', fontsize=12, labelpad=10)
    plt.ylabel('密度估计', fontsize=12, labelpad=10)
    plt.title('情感得分分布曲线 (核密度估计)', fontsize=14, pad=20)

    # 添加分布特征标注
    text_str = f"""分布特征：
    - 均值：{np.mean(scores):.2f}
    - 标准差：{np.std(scores):.2f}
    - 偏度：{float(pd.Series(scores).skew()):.2f}"""
    plt.text(0.65, 0.95 * plt.ylim()[1], text_str,
             bbox=dict(facecolor='white', alpha=0.8),
             fontsize=10)

    # 保存图片
    plt.savefig('sentiment_distribution.png', bbox_inches='tight', dpi=300)
    plt.show()

    # 控制台输出统计信息
    print("\n【情感分布统计】")
    print(f"平均得分: {np.mean(scores):.2f}")
    print(f"得分中位数: {np.median(scores):.2f}")
    print(f"得分标准差: {np.std(scores):.2f}")
    print(f"最低得分: {np.min(scores):.2f}")
    print(f"最高得分: {np.max(scores):.2f}")


def validate_accuracy(ground_truth_file='labeled_data.csv'):
    """
    准确率验证函数
    参数：
    ground_truth_file : 包含人工标注的CSV文件，格式为 text,label (0:负向, 1:正向)
    """
    try:
        # 读取标注数据
        df = pd.read_csv(ground_truth_file)
        if {'text', 'label'}.issubset(df.columns):
            texts = df['text'].tolist()
            true_labels = df['label'].tolist()
        else:
            raise ValueError("CSV文件需要包含 'text' 和 'label' 列")

        # 模型预测
        pred_labels = []
        for text in texts:
            try:
                s = SnowNLP(text)
                # 使用连续值进行阈值分类（保持与之前相同的标准）
                score = s.sentiments
                pred = 1 if score > 0.6 else (0 if score < 0.4 else 2)  # 2表示中性
            except:
                pred = 2  # 解析失败视为中性
            pred_labels.append(pred)

        # 过滤中性样本（根据评估需求）
        filtered_true = []
        filtered_pred = []
        for t, p in zip(true_labels, pred_labels):
            if p != 2:  # 只评估明确分类的样本
                filtered_true.append(t)
                filtered_pred.append(p)

        # 计算指标
        accuracy = accuracy_score(filtered_true, filtered_pred)
        f1 = f1_score(filtered_true, filtered_pred, average='weighted')
        cm = confusion_matrix(filtered_true, filtered_pred)

        # 打印报告
        print("\n【验证报告】")
        print(f"评估样本数：{len(filtered_true)}")
        print(f"准确率：{accuracy:.2%}")
        print(f"加权F1：{f1:.2%}")
        print("\n分类报告：")
        print(classification_report(filtered_true, filtered_pred, target_names=['负面', '正面']))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['负面', '正面'],
                    yticklabels=['负面', '正面'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.savefig('confusion_matrix.png', dpi=300)
        plt.show()

        # 返回详细结果
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'report': classification_report(filtered_true, filtered_pred, output_dict=True)
        }

    except FileNotFoundError:
        print(f"标注文件 {ground_truth_file} 未找到")
        return None
    except Exception as e:
        print(f"验证过程出错：{str(e)}")
        return None

# 在文件顶部添加pandas导入（用于偏度计算）
import pandas as pd

# 调用输出函数
print_sentiment_results(sentiment_stats, detailed_sentiments)

# 保存详细情感数据（新增）
sentiment_output = 'sentiment_scores.csv'
with open(sentiment_output, 'w', encoding='utf-8') as f:
    f.write("text,sentiment_score\n")
    for line, score in zip(lines, detailed_sentiments):
        f.write(f'"{line.strip()}",{score}\n')
print(f"\n详细情感数据已保存到 {sentiment_output}")


