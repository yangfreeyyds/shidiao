import numpy as np
import matplotlib.pyplot as plt
from snownlp import SnowNLP, sentiment
from matplotlib.font_manager import FontProperties

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

def analyze_sentiments(data):
    sentiments = []
    for line in data:
        text = line.strip()
        if not text:
            continue  # 跳过空行
        s = SnowNLP(text)
        sentiments.append(s.sentiments)
    return sentiments

def plot_sentiment_distribution(sentiments):
    sentiments = np.array(sentiments)
    bins = np.linspace(0, 1, 6)
    counts, bin_edges = np.histogram(sentiments, bins=bins)

    # Plotting
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    y = counts

    font_path = 'C:/Windows/Fonts/simhei.ttf'  # 在Windows中设置合适的中文字体路径
    font_prop = FontProperties(fname=font_path)

    plt.figure(figsize=(10, 6))
    plt.fill_between(x, y, color='lightgreen', alpha=0.6)

    plt.xticks(x, [f'({bins[i]:.1f}, {bins[i+1]:.1f}]' for i in range(len(bins)-1)], fontproperties=font_prop)
    plt.yticks(fontproperties=font_prop)
    plt.xlabel('情感得分区间', fontproperties=font_prop)
    plt.ylabel('频数', fontproperties=font_prop)
    plt.title('情感得分分布', fontproperties=font_prop, fontsize=14)

    for i, txt in enumerate(y):
        plt.annotate(f'{txt}', (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center', fontproperties=font_prop)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main():
    original_data = load_data("original_text.txt")

    if not original_data:
        print("文本数据为空。请检查文件内容。")
        return

    sentiment.load("sentiment.marshal")  # Load the pre-trained model
    sentiments = analyze_sentiments(original_data)

    plot_sentiment_distribution(sentiments)

if __name__ == '__main__':
    main()
