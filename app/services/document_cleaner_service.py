"""文档清洗服务模块 - 在切分前清理原始文本中的脏字符、页眉页脚、重复段落"""

import re
import unicodedata
from collections import Counter

from loguru import logger

# 零宽空格(0x200B)、零宽连接符(0x200C)、零宽非连接符(0x200D)、BOM(0xFEFF)
# ——NFKC 规范化处理不到的不可见字符；用 chr(整数编码) 构造，避免把不可见字符本身打进源码里
_INVISIBLE_CODEPOINTS = [0x200B, 0x200C, 0x200D, 0xFEFF]
INVISIBLE_CHARS_PATTERN = re.compile("[" + "".join(chr(c) for c in _INVISIBLE_CODEPOINTS) + "]")
EXCESSIVE_NEWLINES_PATTERN = re.compile(r"\n{3,}")


class DocumentCleanerService:
    """文档清洗服务 - 对加载器读出的原始文本做统一清理，供分割器使用"""

    def clean(self, text: str) -> str:
        """
        清洗原始文本

        流程：Unicode规范化 -> 清理不可见字符 -> 行级空白清理
              -> 段落去重 -> 去页眉页脚 -> 合并多余空行

        注意：段落去重必须在去页眉页脚之前——如果某一整段正文被意外重复
        （比如复制粘贴多贴了一次），段落内每一行也会跟着重复出现，此时如果先
        按“行重复次数”识别页眉页脚，会把这些正文行误判为噪声一并删掉。先做
        段落去重，把整段重复消掉，页眉页脚检测就只会命中真正跨位置重复的行。
        """
        if not text or not text.strip():
            return text

        original_length = len(text)

        text = self._normalize_unicode(text)
        text = self._remove_invisible_chars(text)
        text = self._normalize_line_whitespace(text)
        text = self._deduplicate_paragraphs(text)
        text = self._remove_headers_footers(text)
        text = self._collapse_excessive_newlines(text)

        logger.debug(f"文档清洗完成: {original_length} -> {len(text)} 字符")
        return text

    def _normalize_unicode(self, text: str) -> str:
        """NFKC 规范化：全角转半角、不间断空格转普通空格等"""
        return unicodedata.normalize("NFKC", text)

    def _remove_invisible_chars(self, text: str) -> str:
        """清理零宽空格等 NFKC 处理不到的不可见字符"""
        return INVISIBLE_CHARS_PATTERN.sub("", text)

    def _normalize_line_whitespace(self, text: str) -> str:
        """统一换行符为 \\n，清理每行行尾多余空格"""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines)

    def _remove_headers_footers(
        self, text: str, min_repeat: int = 2, max_line_length: int = 50
    ) -> str:
        """
        按频率启发式去除页眉页脚

        短行（<=max_line_length）且在全文重复出现 >=min_repeat 次，
        判定为页眉/页脚一类的噪声行（主要针对 PDF 提取出的文本生效）。

        min_repeat 默认 2 而不是 3：页眉页脚每页都会出现一次，一份 2 页的
        文档页眉也只会重复 2 次，阈值设成 3 会导致短文档完全清不到页眉页脚。
        """
        lines = text.split("\n")
        line_counts = Counter(line.strip() for line in lines if line.strip())
        noise_lines = {
            line
            for line, count in line_counts.items()
            if count >= min_repeat and len(line) <= max_line_length
        }
        if noise_lines:
            logger.debug(f"识别到疑似页眉页脚 {len(noise_lines)} 处: {noise_lines}")
        return "\n".join(line for line in lines if line.strip() not in noise_lines)

    def _deduplicate_paragraphs(self, text: str) -> str:
        """
        去除完全重复的段落（按空行分段，保留首次出现的顺序）

        只对多行段落去重；单行的"段落"（比如独占一行、前后又是空行的页眉页脚）
        留给 _remove_headers_footers 按重复频率处理——如果这里连单行也一起去重，
        一份 2 页文档的页眉只会剩 1 份，后面按"重复 >= 2 次"判断页眉页脚时反而
        因为只剩 1 次而识别不出来，两个清洗步骤会互相打架。
        """
        paragraphs = text.split("\n\n")
        seen: set[str] = set()
        result = []
        for para in paragraphs:
            normalized = para.strip()
            if not normalized:
                continue
            if "\n" not in normalized:
                # 单行段落不参与去重，原样保留，交给去页眉页脚那一步判断
                result.append(para)
                continue
            if normalized not in seen:
                seen.add(normalized)
                result.append(para)
        return "\n\n".join(result)

    def _collapse_excessive_newlines(self, text: str) -> str:
        """合并 3 个以上连续换行为 2 个（最多保留一行空行），并掐头去尾"""
        text = EXCESSIVE_NEWLINES_PATTERN.sub("\n\n", text)
        return text.strip()


# 全局单例
document_cleaner_service = DocumentCleanerService()
