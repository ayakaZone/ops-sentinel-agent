"""knowledge_tool 里 _expand_query / _rerank_documents 的单元测试

这两个函数都真实调用外部 API（DashScope 的查询改写模型 / 精排模型），
用 mocker 打桩替换掉真实调用，只验证"给定 API 返回结果后，我们自己的
处理逻辑对不对"，包括 API 失败时的降级逻辑。
"""

import pytest
from langchain_core.documents import Document

# 注意：不在文件顶部 import knowledge_tool，而是延迟到每个测试函数体内部才
# import——knowledge_tool.py 顶部会触发 app.services.vector_store_manager 的
# 模块级单例真实连接 Milvus（不是等真正检索才连）。如果写在文件顶部，pytest
# 收集（collect）这个文件时就会立刻触发连接，哪怕这些测试因为标记被跳过不执行
# 也拦不住——collect 阶段必须先 import 整个文件才能知道里面有哪些测试。写在
# 函数体内部，只有测试真的被执行时才会 import，配合 pytestmark 才能做到
# `pytest -m "not integration"` 时真正零 Milvus 依赖。
pytestmark = pytest.mark.integration


def test_expand_query_returns_rewritten_and_alternatives(mocker):
    """
    正常情况：改写后的主查询 + 2 个角度不同的变体，共 3 条

    注意：这里没有用 mocker.patch("...._expansion_chain.invoke", ...) 去 patch
    实例方法——_expansion_chain 是 LangChain 的 RunnableSequence，本质是个
    Pydantic 模型，mock 在测试结束时清理（__exit__）会尝试 delattr 打上去的
    补丁属性，但 Pydantic 模型不允许随意 delattr 非定义字段，会抛
    AttributeError。改成直接把 _expansion_chain 这个模块级名字整个换成
    一个假对象，绕开对 Pydantic 实例内部属性动刀子的问题。
    """
    from app.tools.knowledge_tool import _expand_query

    fake_expansion = mocker.Mock(
        rewritten_query="CPU 使用率过高的排查方法",
        alternative_queries=["CPU 高负载的常见原因", "CPU 性能瓶颈定位"],
    )
    fake_chain = mocker.Mock()
    fake_chain.invoke.return_value = fake_expansion
    mocker.patch("app.tools.knowledge_tool._expansion_chain", fake_chain)

    result = _expand_query("CPU老是很高怎么办")

    assert result == [
        "CPU 使用率过高的排查方法",
        "CPU 高负载的常见原因",
        "CPU 性能瓶颈定位",
    ]


def test_expand_query_falls_back_to_original_on_failure(mocker):
    """改写调用异常时，应该优雅降级为只用原始查询，而不是让整个工具报错"""
    from app.tools.knowledge_tool import _expand_query

    fake_chain = mocker.Mock()
    fake_chain.invoke.side_effect = RuntimeError("DashScope 超时")
    mocker.patch("app.tools.knowledge_tool._expansion_chain", fake_chain)

    result = _expand_query("CPU老是很高怎么办")

    assert result == ["CPU老是很高怎么办"]


def test_rerank_documents_reorders_by_relevance(mocker):
    """精排应该按 API 返回的 index 顺序重新排列文档，并截断到 top_n"""
    from app.tools.knowledge_tool import _rerank_documents

    docs = [
        Document(page_content="今天天气不错"),
        Document(page_content="CPU 使用率过高的排查步骤"),
        Document(page_content="内存泄漏的常见原因"),
    ]

    fake_response = mocker.Mock()
    fake_response.status_code = 200
    fake_response.output.results = [
        mocker.Mock(index=1, relevance_score=0.92),  # 最相关：CPU 那篇排第一
        mocker.Mock(index=2, relevance_score=0.55),  # 次相关：内存那篇排第二
    ]
    mocker.patch("app.tools.knowledge_tool.TextReRank.call", return_value=fake_response)

    result = _rerank_documents("CPU 相关问题", docs, top_n=2)

    assert len(result) == 2
    assert result[0].page_content == "CPU 使用率过高的排查步骤"
    assert result[1].page_content == "内存泄漏的常见原因"


def test_rerank_documents_falls_back_on_api_error_status(mocker):
    """API 返回非 200 状态码时，应该降级为按原始顺序截取前 top_n 篇"""
    from app.tools.knowledge_tool import _rerank_documents

    docs = [Document(page_content=f"文档{i}") for i in range(5)]

    fake_response = mocker.Mock()
    fake_response.status_code = 400
    fake_response.message = "invalid api key"
    mocker.patch("app.tools.knowledge_tool.TextReRank.call", return_value=fake_response)

    result = _rerank_documents("查询", docs, top_n=3)

    assert result == docs[:3]


def test_rerank_documents_falls_back_on_exception(mocker):
    """API 调用抛异常（超时/网络错误）时，同样应该降级，不能让整个检索工具崩掉"""
    from app.tools.knowledge_tool import _rerank_documents

    docs = [Document(page_content=f"文档{i}") for i in range(5)]

    mocker.patch(
        "app.tools.knowledge_tool.TextReRank.call",
        side_effect=ConnectionError("网络不可达"),
    )

    result = _rerank_documents("查询", docs, top_n=3)

    assert result == docs[:3]
