import os
import re
from collections import defaultdict


def load_existing_labels(pos_path, neg_path):
    """加载已有标注数据"""
    labeled = defaultdict(set)
    try:
        with open(pos_path, 'r', encoding='utf-8') as f:
            labeled['pos'] = set(line.strip() for line in f)
    except FileNotFoundError:
        labeled['pos'] = set()

    try:
        with open(neg_path, 'r', encoding='utf-8') as f:
            labeled['neg'] = set(line.strip() for line in f)
    except FileNotFoundError:
        labeled['neg'] = set()

    return labeled


def preprocess_text(text):
    """文本预处理（与之前保持一致）"""
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# 文件配置
pos_file = 'positive.txt'
neg_file = 'negative.txt'
source_file = 'main.txt'  # 原始评论文件

# 初始化
labeled_data = load_existing_labels(pos_file, neg_file)
all_comments = [line.strip() for line in open(source_file, 'r', encoding='utf-8')]
total = len(all_comments)
current_index = 0
stats = {'pos': len(labeled_data['pos']), 'neg': len(labeled_data['neg'])}

print(f"【共富工坊评论标注工具】")
print(f"总评论数: {total} | 已标注: {stats['pos'] + stats['neg']}")
print(f"积极: {stats['pos']} | 消极: {stats['neg']}\n")

# 开始标注流程
try:
    while current_index < total:
        comment = all_comments[current_index]
        cleaned = preprocess_text(comment)

        # 跳过空评论和已标注评论
        if not cleaned or comment in labeled_data['pos'] or comment in labeled_data['neg']:
            current_index += 1
            continue

        # 显示待标注内容
        print(f"\n【第 {current_index + 1}/{total} 条】")
        print(f"原文: {comment}")
        print("-" * 40)
        print("1. 积极 | 2. 消极 | 0. 跳过 | q. 退出")
        choice = input("请选择标注类型：").strip().lower()

        # 处理用户输入
        if choice == 'q':
            break
        elif choice == '1':
            labeled_data['pos'].add(comment)
            stats['pos'] += 1
            current_index += 1
        elif choice == '2':
            labeled_data['neg'].add(comment)
            stats['neg'] += 1
            current_index += 1
        elif choice == '0':
            current_index += 1
            continue
        else:
            print("输入无效，请重新选择")
            continue

        # 实时保存结果
        with open(pos_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(labeled_data['pos']))
        with open(neg_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(labeled_data['neg']))

        print(f"当前统计：积极 {stats['pos']} | 消极 {stats['neg']}")

except KeyboardInterrupt:
    print("\n用户中断操作，正在保存已标注数据...")

finally:
    # 最终保存
    with open(pos_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(labeled_data['pos']))
    with open(neg_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(labeled_data['neg']))

    print("\n【标注结果】")
    print(f"积极评论数: {stats['pos']}")
    print(f"消极评论数: {stats['neg']}")
    print(f"未标注评论数: {total - (stats['pos'] + stats['neg'])}")
    print(f"标注文件已保存至：{pos_file} 和 {neg_file}")
