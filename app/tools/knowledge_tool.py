"""知识检索工具 - 从向量数据库中检索相关信息"""

from http import HTTPStatus
from textwrap import dedent
from typing import List, Tuple

from dashscope import TextReRank
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_qwq import ChatQwen
from loguru import logger
from pydantic import BaseModel, Field

from app.config import config
from app.services.vector_store_manager import vector_store_manager


class QueryExpansion(BaseModel):
    """查询改写与多角度扩展结果"""

    rewritten_query: str = Field(description="改写后的主查询，措辞更贴近专业文档书面语")
    alternative_queries: List[str] = Field(
        description="从不同角度提出的检索查询，用于扩大召回覆盖面",
        min_length=2,
        max_length=2,
    )


# 查询改写与扩展的提示词
expansion_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                你是一个专业的检索查询优化助手，任务是帮助用户的原始问题在运维知识库中获得更好的检索效果。

                请完成两件事：
                1. 把用户的原始问题改写成更贴近专业运维文档书面语的表述（rewritten_query）
                2. 从另外两个不同的角度，提出两个新的检索查询（alternative_queries），
                   帮助覆盖用户问题可能涉及的其他相关方面

                注意：改写和扩展都要保持和原始问题相关，不要引入无关的主题。
            """).strip(),
        ),
        ("human", "{query}"),
    ]
)

# 专门用一个更便宜的模型做查询改写/扩展这个子任务，跟主对话模型分开
_expansion_llm = ChatQwen(
    model="qwen-flash",
    api_key=config.dashscope_api_key,
    temperature=0,
)
_expansion_chain = expansion_prompt | _expansion_llm.with_structured_output(QueryExpansion)


def _expand_query(query: str) -> List[str]:
    """
    查询改写 + 多角度扩展

    Args:
        query: 用户的原始查询

    Returns:
        List[str]: 用于检索的查询列表（改写后的主查询 + 2 个角度不同的变体），
                   如果改写失败则退化为只使用原始查询
    """
    try:
        expansion = _expansion_chain.invoke({"query": query})
        queries = [expansion.rewritten_query, *expansion.alternative_queries]
        logger.info(f"查询改写与扩展完成: {query!r} -> {queries}")
        return queries
    except Exception as e:
        logger.warning(f"查询改写与扩展失败，退化为仅使用原始查询: {e}")
        return [query]


def _rerank_documents(query: str, docs: List[Document], top_n: int) -> List[Document]:
    """
    用 DashScope 精排对候选文档重新排序

    Args:
        query: 用户的原始问题（精排用交叉编码器，对措辞不敏感，不需要用改写后的版本）
        docs: 多角度检索去重合并后的候选文档
        top_n: 最终保留的文档数量

    Returns:
        List[Document]: 按相关性排序、截断到 top_n 篇的文档列表；
                        精排失败时退化为按原始检索顺序截取前 top_n 篇
    """
    try:
        response = TextReRank.call(
            model="gte-rerank-v2",
            query=query,
            documents=[doc.page_content for doc in docs],
            top_n=top_n,
            api_key=config.dashscope_api_key,
        )
        if response.status_code != HTTPStatus.OK:
            logger.warning(f"精排调用失败，退化为使用原始检索顺序: {response.message}")
            return docs[:top_n]

        reranked = [docs[result.index] for result in response.output.results]
        scores = [round(result.relevance_score, 3) for result in response.output.results]
        logger.info(f"精排完成: {len(docs)} -> {len(reranked)} 篇，分数: {scores}")
        return reranked
    except Exception as e:
        logger.warning(f"精排调用异常，退化为使用原始检索顺序: {e}")
        return docs[:top_n]


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题

    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。

    Args:
        query: 用户的问题或查询

    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识检索工具被调用: query='{query}'")

        # 查询改写 + 多角度扩展，得到多个用于检索的查询
        queries = _expand_query(query)

        # 从向量存储中检索相关文档
        vector_store = vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(
            search_kwargs={"k": config.rag_top_k}
        )

        # 对每个查询分别检索，按文档内容去重合并（同一分片被不同角度的查询命中时只保留一份）
        seen_content = set()
        docs: List[Document] = []
        for q in queries:
            for doc in retriever.invoke(q):
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    docs.append(doc)

        if not docs:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []

        logger.info(f"多角度检索去重后共 {len(docs)} 个候选文档，开始精排")

        # 精排：从候选池里挑出真正最相关的 top_k 篇
        docs = _rerank_documents(query, docs, top_n=config.rag_top_k)

        # 格式化文档为上下文
        context = format_docs(docs)

        logger.info(f"检索到 {len(docs)} 个相关文档")
        return context, docs

    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        return f"检索知识时发生错误: {str(e)}", []


def format_docs(docs: List[Document]) -> str:
    """
    格式化文档列表为上下文文本

    Args:
        docs: 文档列表

    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []

    for i, doc in enumerate(docs, 1):
        # 提取元数据
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")

        # 提取标题信息 (如果有)
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])

        header_str = " > ".join(headers) if headers else ""

        # 构建格式化文本
        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)
