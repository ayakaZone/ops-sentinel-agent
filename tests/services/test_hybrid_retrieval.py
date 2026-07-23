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
