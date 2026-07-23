"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py 在 app/ 目录下，上一级就是项目根目录；用 __file__ 动态推算，
# 不管从哪个工作目录启动进程（命令行 cd 到子目录 / IDE 默认按脚本所在目录运行），
# .env 都能被正确找到，不依赖当前工作目录
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "OpsSentinelAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_top_k: int = 3
    # 混合检索每个查询变体在精排前保留的候选数。最终回答仍只使用 rag_top_k 篇。
    # 分开配置可以避免“召回阶段太早截断”，给精排留下足够的候选池。
    rag_candidate_k: int = 6
    # 精排模型会为“用户问题 + 文档切片”给出 0.0 ~ 1.0 的相关性分数。
    # 只要最高分低于这个阈值，就不把不可靠的知识库内容交给大模型生成答案，
    # 而是返回引导性提示，降低 RAG 幻觉风险。该值应结合后续评测与线上日志校准。
    rag_min_relevance_score: float = 0.5
    # gte-rerank-v2 已下线，使用 DashScope 当前推荐的文本精排模型。
    rag_rerank_model: str = "qwen3-rerank"
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # 对话历史摘要压缩配置（SummarizationMiddleware）
    # qwen-max 最大输入长度 30K，trigger 取其 80%；keep 保留摘要后的原始消息 token 数
    context_summary_trigger_tokens: int = 24000
    context_summary_keep_tokens: int = 4000

    # 软限流配置（按天重置，超限仅在响应末尾附加提示，不拦截请求）
    daily_chat_limit: int = 50
    daily_aiops_limit: int = 20

    # MCP 服务配置（transport: stdio | sse | streamable-http）
    # 腾讯云托管 MCP 的 URL 通常含 /sse/，需使用 sse；本地 FastMCP 使用 streamable-http
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    # Prometheus
    prometheus_base_url: str = "http://127.0.0.1:9090"
    prometheus_request_timeout: float = 10.0

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }


# 全局配置实例
config = Settings()
