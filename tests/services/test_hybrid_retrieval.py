"""混合检索接线的回归测试。

这些测试不验证 Milvus 的 BM25 算法本身（那是 Milvus 服务端已经实现的能力），
而是验证我们自己的业务代码有没有：
1. 明确要求使用 RRF 融合；
2. 在 RAG 工具中真正调用混合检索入口。

导入 vector_store_manager 会在模块加载阶段连接真实 Milvus，因此仍标记为 integration，
避免日常 ``pytest -m "not integration"`` 产生数据库依赖。
"""

import pytest
from langchain_core.documents import Document


pytestmark = pytest.mark.integration


def test_hybrid_search_uses_rrf_ranker(mocker):
    """混合检索必须把两路排名交给 Milvus 的 RRF，而不是默认加权融合。"""
    from app.services.vector_store_manager import RRF_K, VectorStoreManager

    # 不调用 __init__：该构造函数会连接 Milvus；本测试只关心 Python 参数是否正确传递。
    manager = object.__new__(VectorStoreManager)
    fake_vector_store = mocker.Mock()
    fake_vector_store.similarity_search.return_value = [
        Document(page_content="CLOSE_WAIT 文档")
    ]
    manager.vector_store = fake_vector_store

    result = manager.hybrid_search("CLOSE_WAIT 如何排查", k=6)

    assert result == [Document(page_content="CLOSE_WAIT 文档")]
    fake_vector_store.similarity_search.assert_called_once_with(
        "CLOSE_WAIT 如何排查",
        k=6,
        ranker_type="rrf",
        ranker_params={"k": RRF_K},
    )


def test_retrieve_knowledge_uses_hybrid_candidates_before_rerank(mocker):
    """查询扩展后的每个变体都应走 hybrid_search，再统一去重、交给精排。"""
    from app.tools.knowledge_tool import RerankOutcome, retrieve_knowledge

    first_doc = Document(page_content="文件句柄耗尽", metadata={"_file_name": "a.docx"})
    duplicate_doc = Document(page_content="文件句柄耗尽", metadata={"_file_name": "a.docx"})
    second_doc = Document(page_content="连接池耗尽", metadata={"_file_name": "b.docx"})

    mocker.patch(
        "app.tools.knowledge_tool._expand_query",
        return_value=["CLOSE_WAIT", "too many open files"],
    )
    hybrid_search = mocker.patch(
        "app.tools.knowledge_tool.vector_store_manager.hybrid_search",
        side_effect=[[first_doc], [duplicate_doc, second_doc]],
    )
    rerank = mocker.patch(
        "app.tools.knowledge_tool._rerank_documents",
        return_value=RerankOutcome(
            documents=[first_doc, second_doc],
            relevance_scores=[0.9, 0.8],
            rerank_succeeded=True,
        ),
    )

    # @tool 装饰后 retrieve_knowledge 是 StructuredTool；.func 才是原始 Python 函数。
    context, artifacts = retrieve_knowledge.func("文件句柄异常怎么处理")

    assert hybrid_search.call_count == 2
    assert all(
        call.kwargs["k"] == 6 for call in hybrid_search.call_args_list
    )
    # 内容相同的 first_doc / duplicate_doc 只能保留一份后送入精排。
    rerank.assert_called_once()
    assert len(rerank.call_args.args[1]) == 2
    assert artifacts == [first_doc, second_doc]
    assert "a.docx" in context
    assert "b.docx" in context


def test_retrieve_knowledge_refuses_low_relevance_context(mocker):
    """所有精排分数低于阈值时，应返回明确引导而不是把噪声交给大模型。"""
    from app.tools.knowledge_tool import (
        NO_RELEVANT_KNOWLEDGE_MESSAGE,
        RerankOutcome,
        retrieve_knowledge,
    )

    low_relevance_doc = Document(page_content="与问题关联很弱", metadata={"_file_name": "noise.md"})
    mocker.patch("app.tools.knowledge_tool._expand_query", return_value=["无关问题"])
    mocker.patch(
        "app.tools.knowledge_tool.vector_store_manager.hybrid_search",
        return_value=[low_relevance_doc],
    )
    mocker.patch(
        "app.tools.knowledge_tool._rerank_documents",
        return_value=RerankOutcome(
            documents=[low_relevance_doc],
            relevance_scores=[0.1],
            rerank_succeeded=True,
        ),
    )

    content, artifacts = retrieve_knowledge.func("无关问题")

    assert content == NO_RELEVANT_KNOWLEDGE_MESSAGE
    assert artifacts == []
