import re
import jieba


def preprocess_text(text):
    """文本预处理函数"""
    pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9]')
    text = pattern.sub(' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# 读取文件并处理前5条
with open('123.txt', 'r', encoding='utf-8') as f:
    count = 0
    for line in f:
        if count >= 5:
            break

        original = line.strip()
        if not original:  # 跳过空行
            continue

        # 预处理和分词
        cleaned = preprocess_text(original)
        seg_list = jieba.cut(cleaned, cut_all=False)
        processed = ' '.join([w for w in seg_list if w.strip()])

        # 格式化输出
        print(f"原始：{original}")
        print(f"分词：{processed}\n{'-' * 40}")
        count += 1
