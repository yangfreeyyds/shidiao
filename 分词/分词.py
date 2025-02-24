import re
import jieba

def preprocess_text(text):
    """
    文本预处理函数
    1. 去除所有非中文字符、英文字母和数字
    2. 去除多余空格
    """
    # 匹配非中文字符、英文、数字的正则表达式
    pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9]')
    # 用空格替换特殊字符
    text = pattern.sub(' ', text)
    # 将多个空格合并为一个
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 读取txt文件（请替换为实际文件路径）
file_path = '123.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    raw_text = f.read()

# 数据预处理
cleaned_text = preprocess_text(raw_text)

# 使用jieba进行分词
seg_list = jieba.cut(cleaned_text, cut_all=False)

# 转换为列表并过滤空字符串
word_list = [word for word in seg_list if word.strip()]

# 输出结果
print("原始文本长度：", len(raw_text))
print("清洗后文本：", cleaned_text)
print("分词结果：", word_list)

# 如果需要保存结果
with open('processed_words.txt', 'w', encoding='utf-8') as f:
    f.write(' '.join(word_list))
