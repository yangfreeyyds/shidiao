import re
import hashlib
from pathlib import Path
import logging
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


class CommentProcessor:
    def __init__(self,
                 min_length: int = 4,
                 deduplicate: bool = True):
        """
        :param min_length: 最小有效评论长度（中文字数）
        :param deduplicate: 是否去重
        """
        self.min_length = min_length * 2  # 按字节计算
        self.deduplicate = deduplicate

    def process_file(self, input_path: str, output_path: str) -> Dict:
        """处理文件并生成清洗报告"""
        try:
            # 读取文件
            content = Path(input_path).read_text(encoding='utf-8')
            lines = content.split('\n')

            # 处理流程
            comments = []
            current_comment = []
            hash_set = set()

            for line in lines:
                stripped = line.strip()

                if not stripped:  # 遇到空行
                    if current_comment:  # 结束当前评论
                        self._add_comment(current_comment, comments, hash_set)
                        current_comment = []
                else:
                    # 清洗并添加内容
                    cleaned = self._clean_line(stripped)
                    current_comment.append(cleaned)

                    # 标点结尾则分段
                    if re.search(r'[。！？…]$', cleaned):
                        self._add_comment(current_comment, comments, hash_set)
                        current_comment = []

            # 处理最后一条评论
            if current_comment:
                self._add_comment(current_comment, comments, hash_set)

            # 保存结果
            self._save_output(comments, output_path)

            return self._generate_report(lines, comments, input_path, output_path)

        except Exception as e:
            logging.error(f"处理失败: {str(e)}")
            raise

    def _clean_line(self, line: str) -> str:
        """行级清洗"""
        # 移除特殊字符
        cleaned = re.sub(r'[\x00-\x1F\x7F]', '', line)
        # 合并连续空格
        return re.sub(r'\s+', ' ', cleaned).strip()

    def _add_comment(self,
                     parts: List[str],
                     comments: List[str],
                     hash_set: set):
        """添加有效评论"""
        full_comment = '\n'.join(parts)
        if len(full_comment.encode('utf-8')) < self.min_length:
            return

        # 去重检查
        comment_hash = hashlib.md5(full_comment.encode()).hexdigest()
        if self.deduplicate and comment_hash in hash_set:
            return

        comments.append(full_comment)
        hash_set.add(comment_hash)

    def _save_output(self, comments: List[str], output_path: str):
        """保存清洗结果"""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(comments))  # 用两个换行分隔评论
        logging.info(f"已保存 {len(comments)} 条评论到 {output_path}")

    def _generate_report(self,
                         original_lines: List[str],
                         cleaned_comments: List[str],
                         input_path: str,
                         output_path: str) -> Dict:
        """生成统计报告"""
        return {
            "input_info": {
                "path": input_path,
                "lines": len(original_lines),
                "empty_lines": sum(1 for l in original_lines if not l.strip())
            },
            "output_info": {
                "path": output_path,
                "comments": len(cleaned_comments),
                "avg_length": f"{sum(len(c) for c in cleaned_comments) / len(cleaned_comments):.1f}" if cleaned_comments else "0.0"
            }
        }


# 使用示例
if __name__ == "__main__":
    processor = CommentProcessor(
        min_length=2,  # 至少2个汉字
        deduplicate=True
    )

    report = processor.process_file(
        input_path="comments.txt",
        output_path="cleaned_comments.txt"
    )

    print("\n=== 清洗报告 ===")
    print(f"输入文件: {report['input_info']['path']}")
    print(f"原始行数: {report['input_info']['lines']}")
    print(f"空行数量: {report['input_info']['empty_lines']}")
    print(f"\n输出文件: {report['output_info']['path']}")
    print(f"有效评论: {report['output_info']['comments']}")
    print(f"平均长度: {report['output_info']['avg_length']}字符")
