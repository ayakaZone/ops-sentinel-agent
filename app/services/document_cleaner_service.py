"""文档清洗服务模块 - 在切分前清理原始文本中的脏字符、页眉页脚、重复段落"""

import re
import unicodedata
from collections import Counter
from math import ceil

from loguru import logger

from app.services.document_loader_service import PDF_PAGE_SEPARATOR

# 零宽空格(0x200B)、零宽连接符(0x200C)、零宽非连接符(0x200D)、BOM(0xFEFF)
# ——NFKC 规范化处理不到的不可见字符；用 chr(整数编码) 构造，避免把不可见字符本身打进源码里
_INVISIBLE_CODEPOINTS = [0x200B, 0x200C, 0x200D, 0xFEFF]
INVISIBLE_CHARS_PATTERN = re.compile("[" + "".join(chr(c) for c in _INVISIBLE_CODEPOINTS) + "]")
EXCESSIVE_NEWLINES_PATTERN = re.compile(r"\n{3,}")

# 只规范化“明确长得像页码”的文本。不能把正文中的所有数字都替换成占位符，
# 否则像“CPU 超过 80%”“重试 3 次”这类正常业务内容也可能被误判为重复页脚。
CHINESE_PAGE_NUMBER_PATTERN = re.compile(r"第\s*\d+\s*页")
ENGLISH_PAGE_NUMBER_PATTERN = re.compile(r"(?i)\bpage\s+\d+\b")


class DocumentCleanerService:
    """文档清洗服务 - 对加载器读出的原始文本做统一清理，供分割器使用"""

    def clean(self, text: str, file_extension: str = "") -> str:
        """
        清洗原始文本

        通用流程：Unicode规范化 -> 清理不可见字符 -> 行级空白清理
                  -> 相邻重复段落去重 -> 合并多余空行

        PDF 专用流程：在相邻段落去重之前，按页面位置识别跨页重复的页眉页脚。
        Markdown/TXT 没有页面概念；python-docx 当前只读取 Word 正文段落，
        不会读取页眉页脚，因此这两个格式不运行 PDF 页眉页脚算法。
        """
        if not text or not text.strip():
            return text

        original_length = len(text)

        text = self._normalize_unicode(text)
        text = self._remove_invisible_chars(text)
        text = self._normalize_line_whitespace(text)
        # 只有 PDF 才会保留 PDF_PAGE_SEPARATOR；只有它需要按页面位置清理。
        # file_extension 设置默认值，是为了兼容现有调用和测试：
        # clean("普通文本") 仍然可以直接使用，只是不运行 PDF 专用逻辑。
        if file_extension.lower() == ".pdf":
            text = self._remove_pdf_headers_footers(text)

        text = self._deduplicate_adjacent_paragraphs(text)
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

    def _remove_pdf_headers_footers(
        self,
        text: str,
        edge_line_count: int = 2,
        repeat_ratio: float = 0.6,
    ) -> str:
        """
        按“页面位置 + 跨页重复”规则删除 PDF 页眉页脚。

        旧算法只要发现一条短文本在全文重复两次，就删除所有同样的行。这个规则
        会误删正文中重复出现的 ``工具: query_logs``、Markdown 代码围栏和参数名。

        新算法分四步：

        1. 通过加载器留下的 PDF_PAGE_SEPARATOR 把文本拆回每一页；
        2. 每页只取顶部和底部各 edge_line_count 个“非空行”作为候选；
        3. 分别统计候选行在多少个页面的顶部/底部出现；
        4. 仅删除“位于同类边缘位置且在足够多页面重复”的行。

        因此，一句在每页正文中间重复出现的工具说明，即使文字完全相同，也不会
        被删除；同样一句话必须同时满足“在页边缘”和“跨页重复”两个条件。

        Args:
            text: 包含 PDF_PAGE_SEPARATOR 的 PDF 原始文本。
            edge_line_count: 每页顶部、底部分别检查多少个非空行。
            repeat_ratio: 判定为重复页眉页脚所需的最小页面比例。
                例如 3 页 PDF、比例 0.6 时，至少在 ceil(3 * 0.6)=2 页出现。

        Returns:
            清理页眉页脚后的普通文本；返回值不再包含换页符。
        """
        # split("\f") 会将“第一页\f第二页”重新拆成 ["第一页", "第二页"]。
        # 空页不参与统计，避免连续分隔符使页面总数被错误放大。
        pages = [page for page in text.split(PDF_PAGE_SEPARATOR) if page.strip()]

        # 单页没有“跨页重复”可判断，不能凭单页顶部的一行文字猜测它是页眉。
        # 同时把可能残留的换页符转换成正常的页面间空行。
        if len(pages) < 2:
            return "\n\n".join(pages)

        # ceil 是“向上取整”。
        # 例：3 页 * 0.6 = 1.8，向上取整为 2，表示至少需要 2 页重复。
        # max(2, ...) 再保证：即使是 2 页或比例很小的文档，也不能只凭 1 页删除。
        minimum_repeat_pages = max(2, ceil(len(pages) * repeat_ratio))

        # Counter 可以理解为“字符串 -> 出现次数”的字典。
        # 页眉和页脚分开统计，避免“第一页顶部、第二页底部”的同一句话被误判。
        header_counts: Counter[str] = Counter()
        footer_counts: Counter[str] = Counter()

        # 这里保存后续真正删除时需要的信息：
        # - lines：这一页的所有原始行（包括空行，保证格式尽量不变）
        # - header_indexes：顶部候选行在 lines 中的下标集合
        # - footer_indexes：底部候选行在 lines 中的下标集合
        page_details: list[tuple[list[str], set[int], set[int]]] = []

        for page in pages:
            lines = page.split("\n")
            header_indexes, footer_indexes = self._get_page_edge_indexes(
                lines,
                edge_line_count=edge_line_count,
            )
            page_details.append((lines, header_indexes, footer_indexes))

            # 同一页中的同样文本只能为计数贡献一次。
            # 否则一页里不小心重复两次的行会把“跨页重复”计数抬高，造成误删。
            current_page_headers = {
                self._normalize_page_number(lines[index].strip())
                for index in header_indexes
            }
            current_page_footers = {
                self._normalize_page_number(lines[index].strip())
                for index in footer_indexes
            }

            header_counts.update(current_page_headers)
            footer_counts.update(current_page_footers)

        # 只保留出现页数达到阈值的候选。这里得到的是“规范化后的特征文本”，
        # 例如“第 1 页”和“第 2 页”都会变成“第 {page} 页”。
        repeated_headers = {
            line
            for line, count in header_counts.items()
            if count >= minimum_repeat_pages
        }
        repeated_footers = {
            line
            for line, count in footer_counts.items()
            if count >= minimum_repeat_pages
        }

        logger.debug(
            "PDF 页眉页脚候选完成: "
            f"页数={len(pages)}, 最小重复页数={minimum_repeat_pages}, "
            f"页眉={repeated_headers}, 页脚={repeated_footers}"
        )

        cleaned_pages: list[str] = []

        for lines, header_indexes, footer_indexes in page_details:
            cleaned_lines: list[str] = []

            # enumerate 同时给出“行下标”和“原始行文本”。
            # index 用于判断这行是不是位于页面顶部/底部候选区域。
            for index, original_line in enumerate(lines):
                normalized_line = self._normalize_page_number(original_line.strip())

                # 注意两个条件必须同时成立：
                # 1. 这行确实在当前页顶部候选区域；
                # 2. 它的规范化文本在多个页面顶部重复出现。
                is_repeated_header = (
                    index in header_indexes
                    and normalized_line in repeated_headers
                )

                # 页脚判断同理，但只检查当前页底部候选区域。
                is_repeated_footer = (
                    index in footer_indexes
                    and normalized_line in repeated_footers
                )

                if is_repeated_header or is_repeated_footer:
                    # continue 表示“跳过当前这一行，直接进入下一轮循环”。
                    # 因此这行不会被添加到 cleaned_lines，等价于从输出中删除。
                    continue

                # 不是确定的页眉/页脚时，宁可保留，避免损坏业务正文。
                cleaned_lines.append(original_line)

            cleaned_pages.append("\n".join(cleaned_lines))

        # 此处页面边界已经完成使命，使用正常空行连接页面即可。
        # 后续的段落去重、切分器和 Embedding 都不会看到 \f。
        return "\n\n".join(cleaned_pages)

    def _get_page_edge_indexes(
        self,
        lines: list[str],
        edge_line_count: int,
    ) -> tuple[set[int], set[int]]:
        """获取一页顶部和底部候选行的原始下标。"""
        # 只记录非空行的下标：空行不应占用“顶部前两行”或“底部后两行”的名额。
        non_empty_indexes = [
            index
            for index, line in enumerate(lines)
            if line.strip()
        ]

        # set（集合）适合后续做 “index in header_indexes” 判断，
        # 并且会自动去重。对两三行这样的小集合来说，可读性也很好。
        header_indexes = set(non_empty_indexes[:edge_line_count])
        footer_indexes = set(non_empty_indexes[-edge_line_count:])
        return header_indexes, footer_indexes

    def _normalize_page_number(self, line: str) -> str:
        """把明确的中英文页码格式转换为统一占位符，便于跨页计数。"""
        # “第 1 页”“第2页” -> “第 {page} 页”。
        normalized = CHINESE_PAGE_NUMBER_PATTERN.sub("第 {page} 页", line)
        # “Page 1”“page 2” -> “Page {page}”。
        return ENGLISH_PAGE_NUMBER_PATTERN.sub("Page {page}", normalized)

    def _deduplicate_adjacent_paragraphs(self, text: str) -> str:
        """
        去除“相邻的”完全重复多行段落。

        旧实现使用全局 seen 集合：一段多行正文只要之前出现过，后面无论相隔多远
        都会删除。这会误伤不同章节中有意复用的操作说明。

        新实现只删除紧挨在一起的重复多行段落，这更符合“复制粘贴多了一份”的
        常见脏数据特征。单行标题、工具名和参数名不参与本步骤的去重。
        """
        paragraphs = text.split("\n\n")
        result: list[str] = []
        previous_normalized: str | None = None
        for para in paragraphs:
            normalized = para.strip()
            if not normalized:
                continue
            if "\n" not in normalized:
                # 单行段落不参与去重：标题、工具名等短文本在正文重复很常见。
                result.append(para)
                previous_normalized = normalized
                continue

            # 只有“当前多行段落”和“紧挨着的上一个保留段落”完全相同，才删除。
            # 如果中间夹了其他正文，previous_normalized 已经变化，后面的同样内容会保留。
            if normalized == previous_normalized:
                continue

            result.append(para)
            previous_normalized = normalized
        return "\n\n".join(result)

    def _collapse_excessive_newlines(self, text: str) -> str:
        """合并 3 个以上连续换行为 2 个（最多保留一行空行），并掐头去尾"""
        text = EXCESSIVE_NEWLINES_PATTERN.sub("\n\n", text)
        return text.strip()


# 全局单例
document_cleaner_service = DocumentCleanerService()
