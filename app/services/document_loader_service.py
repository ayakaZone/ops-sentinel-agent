"""文档加载服务模块 - 按文件类型将原始文件读取为纯文本，供后续清洗/切分复用"""

from pathlib import Path

import pdfplumber
from docx import Document
from loguru import logger


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
        """用 pdfplumber 逐页提取文字，页与页之间用空行分隔"""
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
        return "\n\n".join(page_texts)

    def _load_docx(self, path: Path) -> str:
        """用 python-docx 按顺序读取正文段落（不含页眉页脚，正文之外的内容天然不会读入）"""
        document = Document(path)
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)


# 全局单例
document_loader_service = DocumentLoaderService()
