"""document_cleaner_service.clean() 的单元测试

纯函数，零外部依赖：不 mock 任何东西，直接构造输入、断言输出。
里面几个 case 是这次开发过程中真实踩过的 bug（min_repeat 阈值、
段落去重与去页眉页脚的执行顺序），写成用例防止以后回归。
"""

from app.services.document_cleaner_service import document_cleaner_service


def test_clean_empty_text_returns_as_is():
    """空文本/纯空白文本直接原样返回，不应该抛异常"""
    assert document_cleaner_service.clean("") == ""
    assert document_cleaner_service.clean("   \n  ") == "   \n  "


def test_clean_removes_invisible_chars():
    """零宽空格等不可见字符要被清理掉"""
    dirty = "Hello​World﻿"  # 零宽空格 + BOM
    result = document_cleaner_service.clean(dirty)
    assert "​" not in result
    assert "﻿" not in result
    assert result == "HelloWorld"


def test_clean_normalizes_fullwidth_characters():
    """NFKC 规范化：全角字符转半角"""
    # "Ａ" 是全角 A，规范化后应变成半角 "A"
    result = document_cleaner_service.clean("ＡＢＣ１２３")
    assert result == "ABC123"


def test_clean_collapses_excessive_newlines():
    """3 个以上连续换行要折叠成最多 2 个（保留一行空行）"""
    result = document_cleaner_service.clean("第一段\n\n\n\n第二段")
    assert result == "第一段\n\n第二段"


def test_clean_removes_header_repeated_only_twice():
    """
    回归测试：min_repeat 默认值曾经是 3，导致一份只有 2 页（页眉只重复 2 次）
    的短文档页眉完全清不掉。这里模拟一份 2 页文档验证页眉确实被清理。
    """
    text = (
        "运维手册 v1.0\n"  # 页眉，重复 2 次
        "第一页正文内容，介绍系统架构。\n\n"
        "运维手册 v1.0\n"  # 页眉第二次出现
        "第二页正文内容，介绍部署流程。"
    )
    result = document_cleaner_service.clean(text)
    assert "运维手册 v1.0" not in result
    assert "第一页正文内容" in result
    assert "第二页正文内容" in result


def test_clean_does_not_misjudge_duplicated_body_paragraph_as_header():
    """
    回归测试：降低 min_repeat 阈值后曾经引发的新 bug——一段被复制粘贴
    重复了一次的正文（多行段落），不应该被误判成页眉页脚删掉。
    """
    duplicated_paragraph = "处理步骤：\n1. 检查磁盘\n2. 重启服务"
    text = f"{duplicated_paragraph}\n\n{duplicated_paragraph}\n\n其他正文"
    result = document_cleaner_service.clean(text)
    # 段落去重只保留一份，但内容本身必须还在，不能被误删
    assert "处理步骤" in result
    assert "检查磁盘" in result
    assert result.count("处理步骤") == 1  # 去重生效，只剩一份


def test_clean_deduplicates_repeated_multiline_paragraph():
    """完全重复的多行段落只保留第一次出现"""
    text = "重复内容第一行\n重复内容第二行\n\n重复内容第一行\n重复内容第二行\n\n独有内容"
    result = document_cleaner_service.clean(text)
    assert result.count("重复内容第一行") == 1
    assert "独有内容" in result


def test_clean_normalizes_line_endings():
    """统一 \\r\\n / \\r 为 \\n，并清理行尾多余空格"""
    text = "第一行  \r\n第二行\r第三行"
    result = document_cleaner_service.clean(text)
    assert "\r" not in result
    assert "第一行" in result and "第二行" in result and "第三行" in result
