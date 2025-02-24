#词频统计
#freeyyds

from collections import Counter
from utils import preprocess_text, STOPWORDS
import jieba


def get_word_frequency(lines):
    """核心词频统计逻辑"""
    all_words = []
    for line in lines:
        cleaned = preprocess_text(line.strip())
        if not cleaned: continue
        words = jieba.cut(cleaned, cut_all=False)
        filtered_words = [word for word in words if word not in STOPWORDS and len(word) > 1]
        all_words.extend(filtered_words)
    return Counter(all_words)


def save_word_frequency(counter, filename='词频数据.csv', top_n=150):
    """保存词频结果"""
    top_items = counter.most_common(top_n)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("关键词,出现次数\n")
        for word, count in top_items:
            f.write(f"{word},{count}\n")
    return top_items


if __name__ == '__main__':
    with open('original_text.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    counter = get_word_frequency(lines)
    top_40 = save_word_frequency(counter)

    # 打印结果
    print("【TOP40 关键词词频统计】")
    for idx, (word, count) in enumerate(top_40, 1):
        print(f"{idx:<10}{word:<12}{count:<8}")
