"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""

import uuid
from typing import Annotated, Any, AsyncGenerator, Dict, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, before_model
from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.runtime import Runtime
from loguru import logger
from typing_extensions import TypedDict, deprecated
from langchain_qwq import ChatQwen

from app.config import config
from app.tools import DEFAULT_LOCAL_AGENT_TOOLS
from app.tools.knowledge_tool import format_source_references
from app.agent.mcp_client import (
    get_mcp_client_with_retry,
    load_mcp_tools_safe,
    format_exception_chain,
    suggest_mcp_transport,
)
from app.services.usage_tracker import daily_usage_counter

# 阿里千问大模型和langchain集成参考： https://docs.langchain.com/oss/python/integrations/chat/qwen
# 注意：需要配置环境变量 DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1 否则默认访问的是新加坡站点
# 同时也需要配置环境变量 DASHSCOPE_API_KEY=your_api_key


class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# 模拟用户身份的固定值，仅用于验证"跨会话也能读到长期记忆"这个机制本身；
# 项目没有登录体系，没有真实稳定的用户 ID 可用。生产环境应替换成真实登录用户的 user_id，
# 让每个用户的长期记忆落在各自独立的命名空间下。
FIXED_MEMORY_KEY = "demo_user"
MEMORY_NAMESPACE = ("memories", FIXED_MEMORY_KEY)


def _extract_current_turn_knowledge_docs(
    messages: Sequence[BaseMessage], question: str
) -> list[Document]:
    """从本轮消息中提取知识检索工具返回的真实 Document artifact。

    checkpointer 会保存整个会话历史，不能把历史轮次的来源追加到当前回答。
    因此先从后向前找到本次用户问题对应的 HumanMessage，只处理它之后的
    retrieve_knowledge ToolMessage。
    """
    current_turn_start = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage) and message.content == question:
            current_turn_start = index + 1
            break

    # 不能确定本轮起点时宁可不展示来源，也不能误用历史会话的来源。
    if current_turn_start is None:
        return []

    docs: list[Document] = []
    for message in messages[current_turn_start:]:
        if not isinstance(message, ToolMessage) or message.name != "retrieve_knowledge":
            continue

        artifact = getattr(message, "artifact", None)
        # retrieve_knowledge 约定 artifact 是 List[Document]；这里额外判断 list，
        # 防止其他工具或异常返回值被误当成来源。
        if isinstance(artifact, list):
            docs.extend(doc for doc in artifact if isinstance(doc, Document))

    return docs


@tool
async def save_memory(memory: str, runtime: ToolRuntime) -> str:
    """当用户提到自己的习惯、偏好、关注点等值得长期记住的信息时，调用这个工具保存下来

    Args:
        memory: 要保存的记忆内容，用简洁的一句话概括
    """
    await runtime.store.aput(MEMORY_NAMESPACE, str(uuid.uuid4()), {"content": memory})
    logger.info(f"保存长期记忆: {memory}")
    return "已经记住这条信息了"


@before_model
@deprecated(
    "已由官方 SummarizationMiddleware 取代：按消息条数硬截断可能切断 "
    "AIMessage(tool_calls)/ToolMessage 配对，导致 DashScope API 报 400。"
    "保留此实现仅供学习对比，未接入 create_agent。"
)
def trim_messages_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    修剪消息历史，只保留最近的几条消息以适应上下文窗口

    策略：
    - 保留第一条系统消息（System Message）
    - 保留最近的 6 条消息（3 轮对话）
    - 当消息少于等于 7 条时，不做修剪

    Args:
        state: Agent 状态
        runtime: LangGraph 运行时上下文（本函数未使用，仅满足 before_model 钩子签名）

    Returns:
        包含修剪后消息的字典，如果无需修剪则返回 None
    """
    messages = state["messages"]

    # 如果消息数量较少，无需修剪
    if len(messages) <= 7:
        return None

    # 提取第一条系统消息
    first_msg = messages[0]

    # 保留最近的 6 条消息（确保包含完整的对话轮次）
    recent_messages = messages[-6:] if len(messages) % 2 == 0 else messages[-7:]

    # 构建新的消息列表
    new_messages = [first_msg] + list(recent_messages)

    logger.debug(f"修剪消息历史: {len(messages)} -> {len(new_messages)} 条")

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + ChatQwen 原生集成"""

    def __init__(self, streaming: bool = True):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
        """
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()


        self.model = ChatQwen(
            model=self.model_name,
            api_key=config.dashscope_api_key,
            temperature=0.7,
            streaming=streaming,
        )

        # 定义基础工具（与 AIOps Planner/Executor 共用的本地工具 + 仅对话 Agent 专属的长期记忆工具）
        self.tools = list(DEFAULT_LOCAL_AGENT_TOOLS) + [save_memory]

        # MCP 客户端（延迟初始化，使用全局管理）
        self.mcp_tools: list = []

        # 短期记忆（会话历史）+ 长期记忆（跨会话记忆），由应用启动时 configure_memory() 注入，
        # 在此之前是 None——正常运行时 FastAPI lifespan 会在收到第一个请求之前完成注入
        self.checkpointer = None
        self.store = None

        # Agent 初始化（会在异步方法中完成）
        self.agent = None
        self._agent_initialized = False

        logger.info(f"RAG Agent 服务初始化完成 (ChatQwen), model={self.model_name}, streaming={streaming}")

    def configure_memory(self, checkpointer, store):
        """
        注入持久化的短期/长期记忆存储（由 main.py 的 lifespan 在应用启动时调用）

        Args:
            checkpointer: 短期记忆（会话历史）持久化实例，如 AsyncSqliteSaver
            store: 长期记忆（跨会话记忆）持久化实例，如 AsyncSqliteStore
        """
        self.checkpointer = checkpointer
        self.store = store
        # 记忆存储换了，之前用旧 checkpointer/store 构建的 agent 需要重新构建
        self._agent_initialized = False

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具）"""
        if self._agent_initialized:
            return

        for name, server in config.mcp_servers.items():
            hint = suggest_mcp_transport(
                str(server.get("url", "")),
                str(server.get("transport", "")),
            )
            if hint:
                logger.warning(f"MCP 配置 [{name}]: {hint}")

        mcp_client = await get_mcp_client_with_retry()
        mcp_tools, mcp_err = await load_mcp_tools_safe(mcp_client)
        if mcp_err:
            logger.warning(
                f"MCP 工具加载失败，将仅使用本地工具继续运行:\n{mcp_err}"
            )
            self.mcp_tools = []
        else:
            self.mcp_tools = mcp_tools
            logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")

        all_tools = self.tools + self.mcp_tools

        self.agent = create_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
            store=self.store,
            # trim_messages_middleware：旧的硬截断方案，已弃用（见函数上的 @deprecated 说明），保留代码不启用
            # middleware=[trim_messages_middleware],
            middleware=[
                SummarizationMiddleware(
                    model=self.model,
                    trigger=("tokens", config.context_summary_trigger_tokens),
                    keep=("tokens", config.context_summary_keep_tokens),
                )
            ],
        )

        self._agent_initialized = True


        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        注意：LangChain 框架会自动将工具信息传递给 LLM，
        因此系统提示词中无需列举具体的工具列表。

        Returns:
            str: 系统提示词
        """
        from textwrap import dedent

        return dedent("""
            你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。

            工作原则:
            1. 理解用户需求，选择合适的工具来完成任务
            2. 当需要获取实时信息或专业知识时，主动使用相关工具
            3. 基于工具返回的结果提供准确、专业的回答
            4. 如果工具无法提供足够信息，请诚实地告知用户
            5. 当用户提到自己的习惯、偏好、长期关注的问题等值得记住的信息时，
               主动调用 save_memory 工具保存下来，方便以后的对话里回忆起来

            回答要求:
            - 保持友好、专业的语气
            - 回答简洁明了，重点突出
            - 基于事实，不编造信息
            - 如有不确定的地方，明确说明

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()

    async def _get_memory_context(self) -> str:
        """
        检索长期记忆，格式化成可以直接拼进系统提示词的文本

        每次对话开始前无条件执行（不是让 Agent 自己决定要不要查），
        这样才能保证有相关记忆时一定会被用上，不依赖模型主动调用工具去查。
        """
        if not self.store:
            return ""
        try:
            memories = await self.store.asearch(MEMORY_NAMESPACE)
            if not memories:
                return ""
            memory_lines = "\n".join(f"- {m.value.get('content', '')}" for m in memories)
            return f"\n\n以下是关于用户的历史记忆，如果相关请参考：\n{memory_lines}"
        except Exception as e:
            logger.warning(f"检索长期记忆失败: {e}")
            return ""

    async def query(
        self,
        question: str,
        session_id: str,
    ) -> str:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Returns:
            str: 完整答案
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}")

            # 构建消息列表（系统提示 + 长期记忆 + 用户问题）
            memory_context = await self._get_memory_context()
            messages = [
                SystemMessage(content=self.system_prompt + memory_context),
                HumanMessage(content=question)
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            result = await self.agent.ainvoke(
                input=agent_input,
                config=config_dict,
            )

            # 提取最终答案
            messages_result = result.get("messages", [])
            if messages_result:
                last_message = messages_result[-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

                # 不依赖模型输出来源：直接从本轮知识检索工具的 artifact 生成来源清单。
                # 没有调用知识库、检索拒答或工具异常时 docs 为空，不会显示空来源区域。
                source_docs = _extract_current_turn_knowledge_docs(messages_result, question)
                source_footer = format_source_references(source_docs)
                if source_footer:
                    answer += source_footer

                # 记录工具调用
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")

                # 软限流：超过今日调用次数阈值时，在回答末尾附加友好提示（不拦截请求）
                reminder = daily_usage_counter.increment_and_get_reminder("chat", config.daily_chat_limit)
                if reminder:
                    answer += reminder

                logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")
                return answer

            logger.warning(f"[会话 {session_id}] Agent 返回结果为空")
            return ""

        except Exception as e:
            logger.error(
                f"[会话 {session_id}] RAG Agent 查询失败（非流式）: "
                f"{format_exception_chain(e)}"
            )
            raise

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Yields:
            Dict[str, Any]: 包含流式数据的字典
                - type: "content" | "tool_call" | "complete" | "error"
                - data: 具体内容
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}")

            # 构建消息列表（系统提示 + 长期记忆 + 用户问题）
            memory_context = await self._get_memory_context()
            messages = [
                SystemMessage(content=self.system_prompt + memory_context),
                HumanMessage(content=question)
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            # 在当前 generator 生命周期内收集本轮 ToolMessage，避免从整个会话历史
            # 读取来源，导致上一轮对话的文档被错误追加到本轮答案末尾。
            source_docs = []

            async for token, metadata in self.agent.astream(
                input=agent_input,
                config=config_dict,
                stream_mode="messages",
            ):
                node_name = metadata.get('langgraph_node', 'unknown') if isinstance(metadata, dict) else 'unknown'
                message_type = type(token).__name__

                # response_format="content_and_artifact" 会把原始 Document 列表放进
                # ToolMessage.artifact。模型正文不会看到 artifact，但应用层可以可靠使用它。
                if isinstance(token, ToolMessage) and token.name == "retrieve_knowledge":
                    artifact = getattr(token, "artifact", None)
                    if isinstance(artifact, list):
                        source_docs.extend(artifact)

                if message_type in ("AIMessage", "AIMessageChunk"):
                    content_blocks = getattr(token, 'content_blocks', None)

                    if content_blocks and isinstance(content_blocks, list):
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_content = block.get('text', '')
                                if text_content:
                                    yield {
                                        "type": "content",
                                        "data": text_content,
                                        "node": node_name
                                    }

            # 模型正文输出完毕后，由程序统一补充一次真实来源，避免模型重复或伪造来源。
            source_footer = format_source_references(source_docs)
            if source_footer:
                yield {
                    "type": "content",
                    "data": source_footer,
                    "node": "knowledge_sources",
                }

            # 软限流：超过今日调用次数阈值时，多发一段提示内容（不拦截请求）
            reminder = daily_usage_counter.increment_and_get_reminder("chat", config.daily_chat_limit)
            if reminder:
                yield {"type": "content", "data": reminder, "node": "usage_reminder"}

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            yield {"type": "complete"}

        except Exception as e:
            detail = format_exception_chain(e)
            logger.error(
                f"[会话 {session_id}] RAG Agent 查询失败（流式）: {detail}"
            )
            yield {"type": "error", "data": detail}

    async def _get_raw_session_messages(self, session_id: str) -> list:
        """
        从 checkpointer 读取某个会话的原始 LangChain 消息对象列表（内部共用方法）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 原始消息对象列表（BaseMessage 子类），无历史时返回空列表
        """
        # 使用 checkpointer 的 aget_tuple 方法获取最新的检查点（异步版本，AsyncSqliteSaver
        # 要求在主线程/事件循环里必须用异步接口，同步的 get_tuple() 会直接报错）
        # （get_tuple/aget_tuple 才会返回带 .checkpoint 属性的 CheckpointTuple；
        #  get() 直接返回原始 dict，之前误按 CheckpointTuple 的结构解析导致 KeyError）
        config_dict = {"configurable": {"thread_id": session_id}}
        checkpoint_tuple = await self.checkpointer.aget_tuple(config_dict)

        if not checkpoint_tuple:
            return []

        checkpoint_data = checkpoint_tuple.checkpoint
        return checkpoint_data.get("channel_values", {}).get("messages", [])

    async def get_session_history(self, session_id: str) -> list:
        """
        获取会话历史（从持久化 checkpointer 中读取）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        try:
            messages = await self._get_raw_session_messages(session_id)

            if not messages:
                logger.info(f"获取会话历史: {session_id}, 消息数量: 0")
                return []

            # 转换为前端需要的格式
            history = []
            for msg in messages:
                # 跳过系统消息
                if isinstance(msg, SystemMessage):
                    continue
                    
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)
                
                # 提取时间戳（如果有的话）
                timestamp = getattr(msg, 'timestamp', None)
                if timestamp:
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    })
                else:
                    from datetime import datetime
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    })
            
            logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
            return history

        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    async def get_session_token_usage(self, session_id: str) -> dict:
        """
        获取当前会话历史消息的 token 用量（近似值）

        计算口径与 SummarizationMiddleware 触发摘要压缩时完全一致
        （同样调用 count_tokens_approximately + use_usage_metadata_scaling=True），
        保证前端展示的数字和实际触发摘要的判断标准是同一套，不会两边对不上。

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            dict: {"used": 已用 token 数, "limit": 触发摘要的阈值, "percent": 占比(%)}
        """
        limit = config.context_summary_trigger_tokens
        try:
            messages = await self._get_raw_session_messages(session_id)
            used = count_tokens_approximately(messages, use_usage_metadata_scaling=True) if messages else 0
            percent = round(used / limit * 100, 1) if limit else 0.0
            return {"used": used, "limit": limit, "percent": percent}
        except Exception as e:
            logger.error(f"获取会话 token 用量失败: {session_id}, 错误: {e}")
            return {"used": 0, "limit": limit, "percent": 0.0}

    async def clear_session(self, session_id: str) -> bool:
        """
        清空会话历史（从持久化 checkpointer 中删除）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            bool: 是否成功
        """
        try:
            # 使用 checkpointer 的 adelete_thread 方法删除该 thread 的所有检查点（异步版本）
            await self.checkpointer.adelete_thread(session_id)

            logger.info(f"已清除会话历史: {session_id}")
            return True

        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局单例 - 启用流式输出
rag_agent_service = RagAgentService(streaming=True)
