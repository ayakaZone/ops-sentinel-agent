"""知识检索工具 - 从向量数据库中检索相关信息"""

from http import HTTPStatus
from dataclasses import dataclass
from textwrap import dedent
from typing import List, Sequence, Tuple, TypedDict

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


# 这不是“系统错误”，而是一次正常的检索决策：候选文档存在，但精排判断它们
# 与用户问题不够相关。把它明确告诉 Agent，可避免模型拿无关上下文强行编造答案。
NO_RELEVANT_KNOWLEDGE_MESSAGE = (
    "当前知识库中没有找到足够相关且可信的资料，因此不能基于知识库给出结论。"
    "请补充具体的告警现象、错误信息、服务名称或时间范围后再试。"
)


@dataclass
class RerankOutcome:
    """精排后的业务结果，保留分数以支持阈值判断与后续观测。

    documents 和 relevance_scores 的相同下标表示同一篇文档。例如：
    documents[0] 是第一名文档时，relevance_scores[0] 就是它的精排相关性分数。
    精排 API 失败时，无法取得可信分数，rerank_succeeded 会是 False。
    """

    documents: List[Document]
    relevance_scores: List[float]
    rerank_succeeded: bool


class SourceReference(TypedDict):
    """写入工作流状态的精简知识库来源。

    State 会被 LangGraph 持久化，因此只保存最终展示需要的文件名和标题层级，
    不保存完整 Document 正文，避免检查点中重复存储大段文本。
    """

    file_name: str
    headers: List[str]


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


def _rerank_documents(query: str, docs: List[Document], top_n: int) -> RerankOutcome:
    """
    用 DashScope 精排对候选文档重新排序

    Args:
        query: 用户的原始问题（精排用交叉编码器，对措辞不敏感，不需要用改写后的版本）
        docs: 多角度检索去重合并后的候选文档
        top_n: 最终保留的文档数量

    Returns:
        RerankOutcome: 按相关性排序后的文档、与其一一对应的分数、以及精排是否成功。
                       精排失败时退化为原始检索顺序，但不伪造相关性分数。
    """
    try:
        response = TextReRank.call(
            model=config.rag_rerank_model,
            query=query,
            documents=[doc.page_content for doc in docs],
            top_n=top_n,
            api_key=config.dashscope_api_key,
        )
        if response.status_code != HTTPStatus.OK:
            logger.warning(f"精排调用失败，退化为使用原始检索顺序: {response.message}")
            return RerankOutcome(docs[:top_n], [], False)

        # gte-rerank-v2 的结果位于 response.output.results；当前 qwen3-rerank
        # 则直接位于 response.results。兼容两种结构，便于历史版本平滑迁移。
        output = getattr(response, "output", None)
        results = getattr(output, "results", None) if output is not None else None
        if results is None:
            results = response.results

        reranked = [docs[result.index] for result in results]
        scores = [float(result.relevance_score) for result in results]
        logger.info(f"精排完成: {len(docs)} -> {len(reranked)} 篇，分数: {scores}")
        return RerankOutcome(reranked, scores, True)
    except Exception as e:
        logger.warning(f"精排调用异常，退化为使用原始检索顺序: {e}")
        return RerankOutcome(docs[:top_n], [], False)


def _filter_documents_by_relevance(rerank_outcome: RerankOutcome) -> List[Document]:
    """按精排分数过滤文档；精排失败时不误用阈值拒答。

    阈值只对“精排成功且有分数”的情况生效。否则若因网络超时、限流等原因
    拿不到分数，不能把技术故障误判成“用户问题没有知识库答案”。
    """
    if not rerank_outcome.rerank_succeeded:
        logger.warning("精排未成功，本次跳过相关性阈值过滤，使用召回降级结果")
        return rerank_outcome.documents

    relevant_docs = [
        document
        for document, score in zip(
            rerank_outcome.documents, rerank_outcome.relevance_scores
        )
        if score >= config.rag_min_relevance_score
    ]
    logger.info(
        "相关性阈值过滤完成: {} -> {} 篇，阈值: {:.2f}，分数: {}",
        len(rerank_outcome.documents),
        len(relevant_docs),
        config.rag_min_relevance_score,
        [round(score, 3) for score in rerank_outcome.relevance_scores],
    )
    return relevant_docs


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

        # 对每个查询变体做混合检索：
        # 1. 稠密向量通道负责语义相近匹配；
        # 2. BM25 通道负责错误码、命令、工具名等关键词精确匹配；
        # 3. Milvus 使用 RRF 融合两路排名，返回更完整的候选池。
        #
        # rag_candidate_k 是“精排前每个查询变体的候选数”，不能直接用最终的
        # rag_top_k，否则有价值的候选会在精排开始前被过早丢弃。
        seen_content = set()
        docs: List[Document] = []
        for q in queries:
            for doc in vector_store_manager.hybrid_search(
                q, k=config.rag_candidate_k
            ):
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    docs.append(doc)

        if not docs:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []

        logger.info(f"多角度检索去重后共 {len(docs)} 个候选文档，开始精排")

        # 精排：从候选池里挑出真正最相关的 top_k 篇，同时保留分数。
        rerank_outcome = _rerank_documents(query, docs, top_n=config.rag_top_k)
        docs = _filter_documents_by_relevance(rerank_outcome)

        # 所有候选文档都低于阈值时，明确拒绝使用这些无关上下文回答。
        # 返回空 artifact 也能让后续 Agent 知道“没有可信来源”，而不是误以为有资料。
        if not docs:
            logger.info("知识库拒答: 最高精排分数未达到阈值 {:.2f}", config.rag_min_relevance_score)
            return NO_RELEVANT_KNOWLEDGE_MESSAGE, []

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


def build_source_references(docs: Sequence[Document]) -> List[SourceReference]:
    """从原始 Document 提取可写入状态的精简来源记录。"""
    source_records: List[SourceReference] = []
    seen_sources = set()

    for doc in docs:
        metadata = doc.metadata
        file_name = str(metadata.get("_file_name", "未知来源"))
        headers = [
            str(metadata[key]).strip()
            for key in ["h1", "h2", "h3"]
            if metadata.get(key)
        ]

        source_key = (file_name, *headers)
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        source_records.append({"file_name": file_name, "headers": headers})

    return source_records


def format_source_reference_records(references: Sequence[SourceReference]) -> str:
    """将 State 中的来源记录格式化为最终回答末尾的来源清单。

    同一来源可能来自多个 Planner / Executor 节点，所以在最终展示前再次去重。
    这是一层安全兜底，确保 State 的追加式合并不会造成重复来源。
    """
    source_lines: List[str] = []
    seen_sources = set()

    for reference in references:
        file_name = reference["file_name"]
        headers = reference["headers"]
        source_key = (file_name, *headers)
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)

        source_path = " > ".join([file_name, *headers])
        source_lines.append(f"- {source_path}")

    if not source_lines:
        return ""

    return "\n\n---\n### 参考来源\n\n" + "\n".join(source_lines)


def format_source_references(docs: Sequence[Document]) -> str:
    """根据真实检索文档生成最终回答末尾的来源清单。"""
    return format_source_reference_records(build_source_references(docs))
