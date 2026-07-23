"""RAG 回答来源提取的回归测试。

该文件延迟 import rag_agent_service：其导入链会加载知识检索工具并初始化 Milvus
管理对象，因此标记为 integration；测试本身只构造假的 ToolMessage，不访问模型或数据库。
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


pytestmark = pytest.mark.integration


def test_extract_current_turn_knowledge_docs_ignores_previous_turn_sources():
    """有会话历史时，只能给当前问题追加本轮检索到的来源。"""
    from app.services.rag_agent_service import _extract_current_turn_knowledge_docs

    old_doc = Document(page_content="旧来源")
    current_doc = Document(page_content="当前来源")
    messages = [
        HumanMessage(content="上一轮问题"),
        ToolMessage(content="旧知识上下文", tool_call_id="old-call", name="retrieve_knowledge", artifact=[old_doc]),
        AIMessage(content="上一轮回答"),
        HumanMessage(content="当前问题"),
        ToolMessage(content="当前知识上下文", tool_call_id="current-call", name="retrieve_knowledge", artifact=[current_doc]),
        AIMessage(content="当前回答"),
    ]

    result = _extract_current_turn_knowledge_docs(messages, "当前问题")

    assert result == [current_doc]


def test_extract_current_turn_knowledge_docs_ignores_non_knowledge_tools():
    """只有 retrieve_knowledge 的 artifact 才能成为回答参考来源。"""
    from app.services.rag_agent_service import _extract_current_turn_knowledge_docs

    knowledge_doc = Document(page_content="知识库来源")
    messages = [
        HumanMessage(content="查询"),
        ToolMessage(content="监控数据", tool_call_id="monitor-call", name="query_cpu", artifact=[Document(page_content="监控结果")]),
        ToolMessage(content="知识上下文", tool_call_id="knowledge-call", name="retrieve_knowledge", artifact=[knowledge_doc]),
    ]

    result = _extract_current_turn_knowledge_docs(messages, "查询")

    assert result == [knowledge_doc]


def test_extract_current_turn_knowledge_docs_returns_empty_when_question_not_found():
    """无法确认本轮问题边界时，宁可不展示来源，也不能引用历史资料。"""
    from app.services.rag_agent_service import _extract_current_turn_knowledge_docs

    old_doc = Document(page_content="旧来源")
    messages = [
        HumanMessage(content="历史问题"),
        ToolMessage(content="历史知识", tool_call_id="old-call", name="retrieve_knowledge", artifact=[old_doc]),
    ]

    result = _extract_current_turn_knowledge_docs(messages, "不存在的当前问题")

    assert result == []


async def test_query_stream_appends_source_footer_before_complete(mocker):
    """流式回答完成前，程序应在正文后统一输出一次真实来源清单。"""
    from app.services.rag_agent_service import RagAgentService

    source_doc = Document(
        page_content="CLOSE_WAIT 排查内容",
        metadata={"_file_name": "连接排查.md", "h1": "连接异常", "h2": "CLOSE_WAIT"},
    )

    class FakeAgent:
        """替代 LangGraph Agent：按真实流式顺序返回工具消息和模型文本。"""

        async def astream(self, **kwargs):
            yield (
                ToolMessage(
                    content="知识库上下文",
                    tool_call_id="knowledge-call",
                    name="retrieve_knowledge",
                    artifact=[source_doc],
                ),
                {"langgraph_node": "tools"},
            )
            yield (
                AIMessage(content=[{"type": "text", "text": "这是模型正文。"}]),
                {"langgraph_node": "model"},
            )

    # 不执行 __init__：避免构造 ChatQwen；本测试只验证 query_stream 的事件处理逻辑。
    service = object.__new__(RagAgentService)
    service.agent = FakeAgent()
    service.system_prompt = ""
    service._initialize_agent = mocker.AsyncMock()
    service._get_memory_context = mocker.AsyncMock(return_value="")
    mocker.patch(
        "app.services.rag_agent_service.daily_usage_counter.increment_and_get_reminder",
        return_value="",
    )

    events = [
        event
        async for event in service.query_stream("CLOSE_WAIT 如何排查", "test-session")
    ]

    content_events = [event for event in events if event["type"] == "content"]
    assert content_events[0]["data"] == "这是模型正文。"
    assert content_events[1]["node"] == "knowledge_sources"
    assert "连接排查.md > 连接异常 > CLOSE_WAIT" in content_events[1]["data"]
    assert events[-1] == {"type": "complete"}
