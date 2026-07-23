"""Milvus 客户端工厂模块"""

from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusClient,
    connections,
    utility,
    MilvusException,
)

from app.config import config


def _patch_pymilvus_milvus_client_orm_alias() -> None:
    """
    langchain_milvus 内部创建的 MilvusClient 会将 _using 设为 ``cm-{id}``，
    该别名未在 pymilvus.orm.connections 中注册；随后 ORM ``Collection(..., using=...)``
    会抛出 ConnectionNotExistException: should create connection first.

    在已通过 ``connections.connect(alias="default", ...)`` 建立连接后，
    强制让 MilvusClient 使用 ``default`` 别名，与 ORM 一致。
    """
    if getattr(_patch_pymilvus_milvus_client_orm_alias, "_done", False):
        return
    try:
        from pymilvus.milvus_client.milvus_client import MilvusClient
    except ImportError:
        return

    _orig_init = MilvusClient.__init__

    def _wrapped_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        _orig_init(self, *args, **kwargs)
        self._using = "default"

    MilvusClient.__init__ = _wrapped_init  # type: ignore[method-assign]
    setattr(_patch_pymilvus_milvus_client_orm_alias, "_done", True)


class MilvusClientManager:
    """Milvus 客户端管理器"""

    # 常量定义
    COLLECTION_NAME: str = "biz"
    VECTOR_DIM: int = 1024  # 统一使用 1024 维
    DENSE_VECTOR_FIELD: str = "vector"
    SPARSE_VECTOR_FIELD: str = "sparse_vector"
    BM25_FUNCTION_NAME: str = "bm25_content"
    ID_MAX_LENGTH: int = 100
    CONTENT_MAX_LENGTH: int = 8000
    DEFAULT_SHARD_NUMBER: int = 2

    def __init__(self) -> None:
        """初始化 Milvus 客户端管理器"""
        self._client: MilvusClient | None = None
        self._collection: Collection | None = None

    def connect(self) -> MilvusClient:
        """
        连接到 Milvus 服务器并初始化 collection

        Returns:
            MilvusClient: Milvus 客户端实例

        Raises:
            RuntimeError: 连接或初始化失败时抛出
        """
        # 幂等：导入阶段可能已由 VectorStoreManager 等提前连接，避免重复初始化
        if self._collection is not None and self._client is not None:
            logger.debug("Milvus 已连接，跳过重复 connect")
            return self._client

        try:
            _patch_pymilvus_milvus_client_orm_alias()

            logger.info(f"正在连接到 Milvus: {config.milvus_host}:{config.milvus_port}")

            # 建立连接
            connections.connect(
                alias="default",
                host=config.milvus_host,
                port=str(config.milvus_port),
                timeout=config.milvus_timeout / 1000,  # 转换为秒
            )

            # 创建客户端
            uri = f"http://{config.milvus_host}:{config.milvus_port}"
            self._client = MilvusClient(uri=uri)

            logger.info("成功连接到 Milvus")

            # 检查并创建 collection
            if not self._collection_exists():
                logger.info(f"collection '{self.COLLECTION_NAME}' 不存在，正在创建...")
                self._create_collection()
                logger.info(f"成功创建 collection '{self.COLLECTION_NAME}'")
            else:
                logger.info(f"collection '{self.COLLECTION_NAME}' 已存在")
                self._collection = Collection(self.COLLECTION_NAME)
                
                # 混合检索需要在建表时同时声明：
                # 1. 原有的稠密向量字段 vector；
                # 2. BM25 自动生成的稀疏向量字段 sparse_vector；
                # 3. content 字段上的 BM25 Function。
                #
                # Milvus 不能安全地为已存在的 Collection 补加 Function 输出字段，
                # 因此旧 Schema 不在运行时自动删除（避免应用启动时意外清空知识库），
                # 而是明确报错，要求通过受控迁移命令重建可再生的索引数据。
                schema = self._collection.schema
                field_names = {field.name for field in schema.fields}
                required_fields = {
                    "id",
                    self.DENSE_VECTOR_FIELD,
                    self.SPARSE_VECTOR_FIELD,
                    "content",
                    "metadata",
                }
                missing_fields = required_fields - field_names
                if missing_fields:
                    raise RuntimeError(
                        "Milvus Collection Schema 版本过旧，缺少混合检索字段: "
                        f"{sorted(missing_fields)}。请在确认源文档可重建后，"
                        "删除 biz Collection 并重新索引。"
                    )

                # 原有稠密向量仍使用 DashScope 的 1024 维 Embedding，
                # 因此继续校验 vector 字段维度，防止模型维度变化后静默写错数据。
                vector_field = None
                existing_dim = None
                for field in schema.fields:
                    if field.name == self.DENSE_VECTOR_FIELD:
                        vector_field = field
                        break
                
                if vector_field and hasattr(vector_field, 'params') and 'dim' in vector_field.params:
                    existing_dim = vector_field.params['dim']
                    if existing_dim != self.VECTOR_DIM:
                        logger.warning(
                            f"检测到向量维度不匹配！当前 collection 维度: {existing_dim}, 配置维度: {self.VECTOR_DIM}"
                        )
                        raise RuntimeError(
                            "检测到稠密向量维度不匹配。为避免自动删除知识库，"
                            "请确认源文档后手动重建 biz Collection。"
                        )
                    else:
                        logger.info(f"向量维度匹配: {self.VECTOR_DIM}")

            # 加载 collection
            self._load_collection()

            return self._client

        except MilvusException as e:
            logger.error(f"Milvus 操作失败: {e}")
            self.close()
            raise RuntimeError(f"Milvus 操作失败: {e}") from e
        except ConnectionError as e:
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e

    def _collection_exists(self) -> bool:
        """检查 collection 是否存在"""
        # pymilvus 的类型标注可能不准确，实际返回 bool
        result = utility.has_collection(self.COLLECTION_NAME)
        return bool(result)  # type: ignore[arg-type]

    def _create_collection(self) -> None:
        """创建 biz collection"""
        # 定义字段
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=self.ID_MAX_LENGTH,
                is_primary=True,
            ),
            FieldSchema(
                name=self.DENSE_VECTOR_FIELD,
                dtype=DataType.FLOAT_VECTOR,
                dim=self.VECTOR_DIM,
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=self.CONTENT_MAX_LENGTH,
                # BM25 需要先对文本做分词；当前知识库以中文运维文档为主，
                # 同时夹杂 CLOSE_WAIT、OOMKilled 等英文术语，因此选择 Milvus
                # 内置 chinese 分析器，让中文按词切分、英文术语保留为可匹配 token。
                enable_analyzer=True,
                analyzer_params={"type": "chinese"},
            ),
            FieldSchema(
                # 这是 BM25 Function 的输出字段。应用层不会手工写入它：
                # Milvus 会根据 content 自动生成稀疏向量并维护倒排索引。
                name=self.SPARSE_VECTOR_FIELD,
                dtype=DataType.SPARSE_FLOAT_VECTOR,
                is_function_output=True,
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
            ),
        ]

        # 创建 schema
        schema = CollectionSchema(
            fields=fields,
            description="Business knowledge collection",
            enable_dynamic_field=False,
        )

        # 把 content -> sparse_vector 的转换规则注册到 Collection Schema。
        # 最终的 BM25 分数仍然要等用户查询到来后计算；这里保存的是
        # "关键词稀疏表示 + 倒排索引"所需的可检索数据，而不是某个固定分数。
        schema.add_function(
            Function(
                name=self.BM25_FUNCTION_NAME,
                input_field_names=["content"],
                output_field_names=[self.SPARSE_VECTOR_FIELD],
                function_type=FunctionType.BM25,
            )
        )

        # 创建 collection
        self._collection = Collection(
            name=self.COLLECTION_NAME,
            schema=schema,
            num_shards=self.DEFAULT_SHARD_NUMBER,
        )

        # 创建索引
        self._create_index()

    def _create_index(self) -> None:
        """为稠密向量和 BM25 稀疏向量分别创建索引"""
        if self._collection is None:
            raise RuntimeError("Collection 未初始化")

        index_params = {
            "metric_type": "COSINE",  # 余弦相似度
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }

        _ = self._collection.create_index(
            field_name=self.DENSE_VECTOR_FIELD,
            index_params=index_params,
        )

        # BM25 稀疏向量使用倒排索引。Milvus 在写入 content 时自动更新该索引，
        # 因此后续新增/更新知识文档不需要我们额外维护关键词表。
        sparse_index_params = {
            # 注意：普通 SPARSE_FLOAT_VECTOR 常用 IP；但这里是 Milvus BM25
            # Function 的输出字段，服务端会按 BM25 规则计算分数，必须明确指定
            # metric_type="BM25"，否则建索引会被拒绝。
            "metric_type": "BM25",
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"drop_ratio_build": 0.2},
        }
        _ = self._collection.create_index(
            field_name=self.SPARSE_VECTOR_FIELD,
            index_params=sparse_index_params,
        )

        logger.info("成功创建 vector 稠密索引与 sparse_vector BM25 倒排索引")

    def _load_collection(self) -> None:
        """加载 collection 到内存"""
        if self._collection is None:
            self._collection = Collection(self.COLLECTION_NAME)

        # 检查 collection 是否已加载（兼容多版本）
        try:
            # 方法 1: 尝试使用 utility.load_state（新版本）
            load_state = utility.load_state(self.COLLECTION_NAME)
            # load_state 返回字符串或枚举，如 "Loaded" 或 "NotLoad"
            state_name = getattr(load_state, "name", str(load_state))
            if state_name != "Loaded":
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            else:
                logger.info(f"Collection '{self.COLLECTION_NAME}' 已加载")
        except AttributeError:
            # 方法 2: 直接尝试加载，捕获 "already loaded" 异常
            try:
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            except MilvusException as e:
                error_msg = str(e).lower()
                if "already loaded" in error_msg or "loaded" in error_msg:
                    logger.info(f"Collection '{self.COLLECTION_NAME}' 已加载")
                else:
                    raise
        except Exception as e:
            logger.error(f"加载 collection 失败: {e}")
            raise

    def get_collection(self) -> Collection:
        """
        获取 collection 实例

        Returns:
            Collection: collection 实例

        Raises:
            RuntimeError: collection 未初始化时抛出
        """
        if self._collection is None:
            raise RuntimeError("Collection 未初始化，请先调用 connect()")
        return self._collection

    def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: True 表示健康，False 表示异常
        """
        try:
            if self._client is None:
                return False

            # 尝试列出 connections
            _ = connections.list_connections()
            return True

        except (MilvusException, ConnectionError) as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False
        except Exception as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False

    def close(self) -> None:
        """关闭连接"""
        errors = []
        
        try:
            if self._collection is not None:
                self._collection.release()
                self._collection = None
        except Exception as e:
            errors.append(f"释放 collection 失败: {e}")

        try:
            if connections.has_connection("default"):
                connections.disconnect("default")
        except Exception as e:
            errors.append(f"断开连接失败: {e}")

        self._client = None
        
        if errors:
            error_msg = "; ".join(errors)
            logger.error(f"关闭 Milvus 连接时出现错误: {error_msg}")
        else:
            logger.info("已关闭 Milvus 连接")

    def __enter__(self) -> "MilvusClientManager":
        """上下文管理器入口"""
        _ = self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object
    ) -> None:
        """上下文管理器退出"""
        self.close()


# 全局单例
milvus_manager = MilvusClientManager()
