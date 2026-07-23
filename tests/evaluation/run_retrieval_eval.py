"""检索质量评估脚本 —— Hit Rate@K / MRR

对比四种检索配置的真实效果：
    baseline              纯向量检索，不做任何优化
    +查询扩展              只加查询改写/多角度扩展（#6），不做精排
    +精排                 只加精排（#2），不做查询扩展
    +查询扩展+精排         当前生产环境 knowledge_tool.retrieve_knowledge() 的真实流程

用 tests/evaluation/golden_set.json 里人工标注的"问题 -> 应该命中的文档"跑一遍，
输出每种配置的 Hit Rate@K 和 MRR，量化验证每个优化点实际带来的提升。

会真实调用 DashScope（embedding / 查询改写 / 精排），有 API 开销，
不放进日常 pytest / CI，手动执行：
    python tests/evaluation/run_retrieval_eval.py
"""

import json
from pathlib import Path
from typing import Callable, List

from langchain_core.documents import Document

from app.config import config
from app.services.vector_store_manager import vector_store_manager
from app.tools.knowledge_tool import _expand_query, _rerank_documents

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


def retrieve_baseline(query: str, top_k: int) -> List[Document]:
    """纯向量检索，不做任何优化——对照组"""
    return vector_store_manager.similarity_search(query, k=top_k)


def retrieve_with_expansion(query: str, top_k: int) -> List[Document]:
    """只加查询改写 + 多角度扩展，不做精排"""
    queries = _expand_query(query)
    seen, docs = set(), []
    for q in queries:
        for doc in vector_store_manager.similarity_search(q, k=top_k):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                docs.append(doc)
    return docs[:top_k]  # 没有精排环节，按检索原始顺序直接截断


def retrieve_with_rerank(query: str, top_k: int) -> List[Document]:
    """只加精排，不做查询扩展——单查询先捞一个更大的候选池，交给精排收窄"""
    candidates = vector_store_manager.similarity_search(query, k=top_k * 3)
    return _rerank_documents(query, candidates, top_n=top_k)


def retrieve_with_hybrid(query: str, top_k: int) -> List[Document]:
    """只启用“向量 + BM25 + RRF”混合召回，不做查询扩展和外部精排。"""
    return vector_store_manager.hybrid_search(query, k=top_k)


def retrieve_with_both(query: str, top_k: int) -> List[Document]:
    """当前生产环境 retrieve_knowledge() 的真实流程：查询扩展 + 精排"""
    queries = _expand_query(query)
    seen, docs = set(), []
    for q in queries:
        for doc in vector_store_manager.similarity_search(q, k=top_k):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                docs.append(doc)
    return _rerank_documents(query, docs, top_n=top_k)


def evaluate(golden_set: list[dict], retrieve_fn: Callable, top_k: int) -> dict:
    """跑一遍 golden_set，算 Hit Rate@K 和 MRR"""
    hits, reciprocal_ranks = 0, []

    for item in golden_set:
        docs = retrieve_fn(item["query"], top_k)
        file_names = [doc.metadata.get("_file_name") for doc in docs]

        if item["expected_file"] in file_names:
            hits += 1
            rank = file_names.index(item["expected_file"]) + 1  # 排名从 1 开始
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)  # 没命中，倒数排名记 0

    n = len(golden_set)
    return {"hit_rate": hits / n, "mrr": sum(reciprocal_ranks) / n, "n": n}


def main():
    golden_set = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    top_k = config.rag_top_k

    configs = {
        "baseline（纯向量检索）": retrieve_baseline,
        "+查询扩展（#6）": retrieve_with_expansion,
        "+精排（#2）": retrieve_with_rerank,
        "+混合检索（向量+BM25+RRF）": retrieve_with_hybrid,
        "+查询扩展+精排（生产环境现状）": retrieve_with_both,
    }

    print(f"\n评测集: {len(golden_set)} 条问题，top_k={top_k}\n")
    print(f"{'配置':<32}{'Hit Rate@K':<14}{'MRR':<10}")
    print("-" * 56)

    results = {}
    for name, fn in configs.items():
        result = evaluate(golden_set, fn, top_k)
        results[name] = result
        print(f"{name:<32}{result['hit_rate']:<14.2%}{result['mrr']:<10.3f}")

    return results


if __name__ == "__main__":
    main()
