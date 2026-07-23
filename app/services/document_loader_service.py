"""文档加载服务模块 - 按文件类型将原始文件读取为纯文本，供后续清洗/切分复用"""

from pathlib import Path

import pdfplumber
from docx import Document
from loguru import logger

# \f 是标准的“换页符”（form feed）。
#
# PDF 在加载时需要保留“这一段文字来自第几页”的边界：后面的清洗器只有知道
# 每一页从哪里开始、在哪里结束，才能区分“每页顶部重复的页眉”与“正文中
# 恰好重复的工具名”。普通空行只表示段落间隔，无法可靠表示换页，因此这里
# 使用一个不常出现在正常正文中的控制字符作为内部标记。
#
# 注意：这个标记只在“加载 -> 清洗”这一小段流程中存在。清洗完成后会被移除，
# 不会进入切分器、Embedding 或 Milvus。
PDF_PAGE_SEPARATOR = "\f"


class DocumentLoaderService:
    """文档加载服务 - 根据文件扩展名分发到对应的加载逻辑，统一返回纯文本"""

    def load(self, file_path: str) -> str:
        """
        读取文件内容为纯文本

        Args:
            file_path: 文件路径

        Returns:
            str: 文件的纯文本内容
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._load_pdf(path)
        elif ext == ".docx":
            return self._load_docx(path)
        else:
            # .md/.txt 本身就是纯文本，直接读取
            return path.read_text(encoding="utf-8")

    def _load_pdf(self, path: Path) -> str:
        """用 pdfplumber 逐页提取文字，并用换页符保留页面边界。"""
        page_texts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    page_texts.append(text)
                else:
                    logger.warning(
                        f"PDF 第 {page.page_number} 页未提取到文字（可能是扫描件图片，"
                        f"需要 OCR 才能处理，本次跳过）: {path}"
                    )
        # 不使用 "\n\n"（普通空行）连接页面，因为空行在正文中也很常见。
        # 后续的 DocumentCleanerService 会根据 PDF_PAGE_SEPARATOR 分页，
        # 只检查每页顶部/底部的候选行；清洗后再把页面重新合并成普通文本。
        return PDF_PAGE_SEPARATOR.join(page_texts)

    def _load_docx(self, path: Path) -> str:
        """用 python-docx 按顺序读取正文段落（不含页眉页脚，正文之外的内容天然不会读入）"""
        document = Document(path)
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)


# 全局单例
document_loader_service = DocumentLoaderService()
