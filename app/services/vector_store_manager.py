"""向量存储管理器 - 封装 Milvus VectorStore 操作"""

from pathlib import Path
from typing import List, Optional, Set

from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction, Milvus
from loguru import logger

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service


# 统一使用 biz collection
COLLECTION_NAME = "biz"
DENSE_VECTOR_FIELD = "vector"
SPARSE_VECTOR_FIELD = "sparse_vector"
BM25_FUNCTION_NAME = "bm25_content"
RRF_K = 60


class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self):
        """初始化向量存储管理器"""
        self.vector_store = None
        self.collection_name = COLLECTION_NAME
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """初始化 Milvus VectorStore"""
        try:
            # 必须在 PyMilvus / langchain_milvus 访问 Collection 之前建立连接，
            # 否则会出现 ConnectionNotExistException: should create connection first.
            # （模块导入时就会执行此处，早于 FastAPI lifespan 中的 milvus_manager.connect）
            _ = milvus_manager.connect()

            connection_args = {
                "host": config.milvus_host,
                "port": config.milvus_port,
            }

            # 创建“稠密向量 + BM25 稀疏向量”的混合 VectorStore。
            #
            # vector 是现有 DashScope Embedding 产生的 1024 维稠密向量，擅长语义匹配；
            # sparse_vector 由 Milvus 根据 content 自动生成，擅长错误码、命令、工具名等
            # 关键词精确匹配。两者的 RRF 融合在 hybrid_search（混合检索）中执行。
            self.vector_store = Milvus(
                embedding_function=vector_embedding_service,
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=False,  # 使用自定义 id
                drop_old=False,
                text_field="content",  # 文本内容存储到 content 字段
                vector_field=[DENSE_VECTOR_FIELD, SPARSE_VECTOR_FIELD],
                primary_field="id",  # 主键字段
                metadata_field="metadata",  # 元数据字段
                # 必须与 MilvusClientManager._create_collection() 中注册的 Function
                # 保持同名、同输入输出字段；已有 Collection 时 LangChain 会复用它，
                # 新建 Collection 时则可按该定义创建同样的 BM25 Function。
                builtin_function=BM25BuiltInFunction(
                    input_field_names="content",
                    output_field_names=SPARSE_VECTOR_FIELD,
                    analyzer_params={"type": "chinese"},
                    function_name=BM25_FUNCTION_NAME,
                ),
            )

            logger.info(
                f"VectorStore 初始化成功: {config.milvus_host}:{config.milvus_port}, "
                f"collection: {self.collection_name}"
            )

        except Exception as e:
            logger.error(f"VectorStore 初始化失败: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        批量添加文档到向量存储（自动批量向量化）

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档 ID 列表
        """
        try:
            import time
            import uuid
            start_time = time.time()

            # 为每个文档生成唯一 id（因为 auto_id=False）
            ids = [str(uuid.uuid4()) for _ in documents]

            # LangChain Milvus 的 add_documents 会自动调用 embedding_function
            # 并进行批量处理，性能更好
            result_ids = self.vector_store.add_documents(documents, ids=ids)

            elapsed = time.time() - start_time
            logger.info(
                f"批量添加 {len(documents)} 个文档到 VectorStore 完成, "
                f"耗时: {elapsed:.2f}秒, 平均: {elapsed/len(documents):.2f}秒/个"
            )
            return result_ids
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    def delete_by_source(self, file_path: str) -> int:
        """
        删除指定文件的所有文档

        Args:
            file_path: 文件路径

        Returns:
            int: 删除的文档数量
        """
        try:
            # 使用 milvus_manager 获取已连接的 collection
            collection = milvus_manager.get_collection()

            # 按文件名匹配，不按完整路径匹配——完整路径里带着项目目录名，
            # 项目改名/索引目录调整（uploads -> aiops-docs）都会让路径字符串变化，
            # 导致这里精确匹配不到旧记录，旧片段变成永久删不掉的孤儿数据（实测踩过）。
            # 文件名在同一个知识库目录下具备唯一性，按文件名匹配更稳。
            file_name = Path(file_path).name
            expr = f'metadata["_file_name"] == "{file_name}"'

            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0
            
            logger.info(f"删除文件旧数据: {file_path}, 删除数量: {deleted_count}")
            return deleted_count
            
        except Exception as e:
            logger.warning(f"删除旧数据失败 (可能是首次索引): {e}")
            return 0

    def get_content_hash(self, file_name: str) -> Optional[str]:
        """
        查询某个文件当前已索引内容的哈希值，用于增量索引判断"这个文件变没变"

        同一个文件的所有分片存的 _content_hash 都是同一个值（分片时统一写入），
        所以这里只需要查 1 条就够，不用把这个文件的全部分片都拉出来。

        Args:
            file_name: 文件名（不含目录路径，比如 "cpu_high_usage.md"）

        Returns:
            Optional[str]: 已索引内容的哈希值；如果这个文件从来没被索引过，返回 None
        """
        try:
            collection = milvus_manager.get_collection()

            # collection.query() 是 pymilvus 提供的"按条件查数据"方法，跟 SQL 的
            # SELECT ... WHERE ... LIMIT 1 是一回事。
            # expr 参数是查询条件（字符串形式的表达式），metadata 是 JSON 字段，
            # metadata["_file_name"] 这种写法是 Milvus 专门用来"从 JSON 字段里按 key
            # 取值再比较"的语法。
            # output_fields 指定只要返回 metadata 这一列，不用把 vector（向量，体积大）
            # 也传回来，省流量。
            # limit=1 表示只要 1 条结果，因为同一文件所有分片的哈希值都相同，查 1 条就够。
            results = collection.query(
                expr=f'metadata["_file_name"] == "{file_name}"',
                output_fields=["metadata"],
                limit=1,
            )

            # query() 查不到任何匹配结果时，返回的是空列表 []，不是 None、也不会报错，
            # 所以要先判断列表是不是为空。
            if not results:
                return None

            # results[0] 是查到的第一条（也是唯一一条，因为 limit=1）记录，
            # 它是一个字典，形如 {"metadata": {"_file_name": "...", "_content_hash": "...", ...}}。
            # 用 .get() 取值而不是直接用 [] 取值，是因为老数据可能是在加上这个字段之前
            # 就已经入库的，压根没有 _content_hash 这个 key，直接用 [] 取会报 KeyError，
            # .get() 取不到时会安全地返回 None，不会报错。
            metadata = results[0].get("metadata", {})
            return metadata.get("_content_hash")

        except Exception as e:
            logger.warning(f"查询文件哈希值失败: {file_name}, 错误: {e}")
            return None

    def list_indexed_file_names(self) -> Set[str]:
        """
        查询当前 Milvus 里所有已索引的文件名集合，用于跟磁盘目录做对比、
        找出"库里有、磁盘上已经删掉了"的孤儿文件。

        Returns:
            Set[str]: 已索引的文件名集合（比如 {"cpu_high_usage.md", "disk_high_usage.md", ...}）
        """
        try:
            collection = milvus_manager.get_collection()

            # 先 flush 一下——Milvus 的删除/插入操作默认是"软提交"，不会立刻反映在
            # query() 能查到的结果里，得等内部把变更真正落盘（这个过程叫 flush）之后
            # 才能查到最新状态。这个方法是孤儿清理逻辑的关键判断依据，必须保证读到的是
            # 最新数据，所以查询前主动 flush 一次，不能依赖"过一会儿它自己就同步了"。
            collection.flush()

            # 这里没法只查"文件名"这一个维度就完事——Milvus 的 query 是按"每一条实体
            # (entity，也就是每个文档分片)"为单位返回的，一篇文档有几个分片就会返回几条
            # 重复的 _file_name。所以查出来之后要用 Python 的 set() 去重，
            # 只保留不重复的文件名。
            # limit=10000 是给一个足够大的上限，避免 Milvus 默认的返回条数限制截断结果；
            # 这个项目量级不大，10000 绰绰有余。
            results = collection.query(
                expr='id != ""',
                output_fields=["metadata"],
                limit=10000,
            )

            # 用集合推导式（跟列表推导式 [x for x in ...] 语法一样，只是外层用花括号 {}
            # 表示"结果去重存成集合"）把每条记录的 _file_name 取出来
            return {r.get("metadata", {}).get("_file_name") for r in results}

        except Exception as e:
            logger.warning(f"查询已索引文件名列表失败: {e}")
            return set()

    def get_vector_store(self) -> Milvus:
        """
        获取 VectorStore 实例

        Returns:
            Milvus: VectorStore 实例
        """
        return self.vector_store

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        纯稠密向量相似度搜索（仅用于评测对照组）

        生产 RAG 链路请使用 hybrid_search（混合检索）。这个方法保留为
        "纯向量检索 baseline"，使 tests/evaluation 可以继续与混合方案做公平对比。

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关文档列表
        """
        try:
            # VectorStore 同时配置两个向量字段后，直接调用 similarity_search 会自动
            # 走混合检索；这里为了保留 baseline，显式调用底层 Collection 的 vector
            # 字段做单路检索。用户查询只在本次请求临时向量化，不会写入 Milvus。
            query_vector = vector_embedding_service.embed_query(query)
            collection = milvus_manager.get_collection()
            results = collection.search(
                data=[query_vector],
                anns_field=DENSE_VECTOR_FIELD,
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=k,
                output_fields=["content", "metadata"],
            )

            docs = [
                Document(
                    page_content=hit.entity.get("content"),
                    metadata=hit.entity.get("metadata") or {},
                )
                for hit in results[0]
            ]
            logger.debug(f"纯向量搜索完成: query='{query}', 结果数={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []

    def hybrid_search(self, query: str, k: int = 6) -> List[Document]:
        """执行“稠密语义检索 + BM25 关键词检索 + RRF 融合”。

        Args:
            query: 用户原始问题或查询扩展后的变体。
            k: 两路融合后返回的候选分片数量；它是精排前的候选池大小，
                通常应大于最终回答使用的 rag_top_k。

        Returns:
            RRF 融合排序后的候选 Document 列表。
        """
        try:
            docs = self.vector_store.similarity_search(
                query,
                k=k,
                # RRF 只依据“稠密检索名次”和“BM25 检索名次”融合，避免直接比较
                # 两种含义不同、数值范围也不同的原始分数。
                ranker_type="rrf",
                ranker_params={"k": RRF_K},
            )
            logger.debug(
                f"混合检索完成: query='{query}', RRF k={RRF_K}, 结果数={len(docs)}"
            )
            return docs
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return []


# 全局单例
vector_store_manager = VectorStoreManager()
