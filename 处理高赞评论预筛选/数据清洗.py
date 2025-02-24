import re
from pathlib import Path
import hashlib
from typing import List, Tuple
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)


class CommentCleaner:
    def __init__(self,
                 max_empty_lines: int = 1,
                 min_comment_length: int = 4,
                 deduplicate: bool = True):
        """
        初始化清洗器
        :param max_empty_lines: 允许保留的最大连续空行数
        :param min_comment_length: 有效评论最小长度（中文字符）
        :param deduplicate: 是否去重
        """
        self.max_empty_lines = max_empty_lines
        self.min_length = min_comment_length * 2  # 中文字符UTF-8长度
        self.deduplicate = deduplicate

    def load_comments(self, input_path: str) -> Tuple[List[str], dict]:
        """加载并预处理原始数据"""
        try:
            raw_text = Path(input_path).read_text(encoding='utf-8')
            stats = {
                'original_size': len(raw_text),
                'original_lines': raw_text.count('\n') + 1
            }
            return raw_text.split('\n'), stats
        except Exception as e:
            logging.error(f"文件加载失败: {str(e)}")
            raise

    def clean_comment(self, comment: str) -> str:
        """单条评论清洗"""
        # 去除特殊字符
        cleaned = re.sub(r'[\x00-\x1F\x7F]', '', comment)
        # 合并连续空格
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # 去除首尾空白
        return cleaned.strip()

    def process(self, input_path: str, output_path: str) -> dict:
        """完整处理流程"""
        # 读取数据
        raw_lines, stats = self.load_comments(input_path)

        # 处理流程
        comments = []
        current_comment = []
        empty_counter = 0
        hash_set = set()

        for line in raw_lines:
            stripped = line.strip()

            if not stripped:  # 空行处理
                empty_counter += 1
                if empty_counter <= self.max_empty_lines:
                    current_comment.append('')
            else:
                # 处理连续空行
                if empty_counter > 0:
                    current_comment.extend([''] * min(empty_counter, self.max_empty_lines))
                    empty_counter = 0

                # 清洗内容
                cleaned = self.clean_comment(stripped)
                if cleaned:
                    current_comment.append(cleaned)

                # 段落结束判断
                if not re.search(r'[。！？…]$', cleaned):
                    continue

                # 生成完整评论
                full_comment = '\n'.join(current_comment)
                if len(full_comment.encode('utf-8')) >= self.min_length:
                    # 去重处理
                    comment_hash = hashlib.md5(full_comment.encode('utf-8')).hexdigest()
                    if not self.deduplicate or comment_hash not in hash_set:
                        comments.append(full_comment)
                        hash_set.add(comment_hash)
                current_comment = []
                empty_counter = 0

        # 保存结果
        self.save_results(comments, output_path)

        # 生成统计报告
        stats.update({
            'cleaned_comments': len(comments),
            'removed_duplicates': len(hash_set) - len(comments) if self.deduplicate else 0,
            'avg_length': sum(len(c) for c in comments) / len(comments) if comments else 0
        })
        return stats

    def save_results(self, comments: List[str], output_path: str):
        """保存清洗结果"""
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(comments))  # 用两个换行分隔评论

            logging.info(f"成功保存清洗结果到: {output_path}")
        except Exception as e:
            logging.error(f"结果保存失败: {str(e)}")
            raise


if __name__ == "__main__":
    # 配置参数
    config = {
        'input_file': 'original_text.txt',  # 原始数据文件路径
        'output_file': 'original_text.txt',  # 输出文件路径
        'max_empty_lines': 1,  # 允许保留的连续空行数
        'min_comment_length': 4,  # 有效评论最小长度（中文字数）
        'deduplicate': True  # 是否去重
    }

    # 执行清洗
    cleaner = CommentCleaner(
        max_empty_lines=config['max_empty_lines'],
        min_comment_length=config['min_comment_length'],
        deduplicate=config['deduplicate']
    )

    try:
        stats = cleaner.process(config['input_file'], config['output_file'])

        # 打印统计报告
        print("\n=== 清洗报告 ===")
        print(f"原始数据行数: {stats['original_lines']}")
        print(f"清洗后评论数: {stats['cleaned_comments']}")
        print(f"移除重复评论: {stats['removed_duplicates']}")
        print(f"平均评论长度: {stats['avg_length']:.1f}字符")
        print(f"输出文件大小: {Path(config['output_file']).stat().st_size / 1024:.2f}KB")
    except Exception as e:
        print(f"处理失败: {str(e)}")
