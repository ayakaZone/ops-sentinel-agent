"""AIOps Planner 来源累积与最终报告来源展示的回归测试。"""

import importlib

import pytest
from langchain_core.documents import Document


# 导入 AIOps 节点会经由知识检索工具加载 Milvus 管理对象，因此标记为 integration。
# 测试内部全部使用 Mock，不会实际调用 Milvus、DashScope 或 MCP 服务。
pytestmark = pytest.mark.integration


async def test_planner_saves_knowledge_source_records_to_state(mocker):
    """Planner 预检索到文档时，应把精简来源记录随 plan 一起写入 State。"""
    planner_module = importlib.import_module("app.agent.aiops.planner")

    docs = [
        Document(
            page_content="CLOSE_WAIT 的知识库内容",
            metadata={"_file_name": "连接排查.md", "h1": "连接异常", "h2": "CLOSE_WAIT"},
        )
    ]

    # Planner 内部通过 asyncio.to_thread 调用同步的 retrieve_knowledge.func。
    mocker.patch.object(
        planner_module.retrieve_knowledge,
        "func",
        return_value=("【参考资料 1】\n内容：CLOSE_WAIT 排查", docs),
    )

    fake_mcp_client = mocker.Mock()
    fake_mcp_client.get_tools = mocker.AsyncMock(return_value=[])
    mocker.patch(
        "app.agent.aiops.planner.get_mcp_client_with_retry",
        new=mocker.AsyncMock(return_value=fake_mcp_client),
    )

    class FakePlannerChain:
        async def ainvoke(self, _inputs):
            return planner_module.Plan(steps=["查询连接状态"])

    class FakePlannerPrompt:
        def __or__(self, _other):
            return FakePlannerChain()

    fake_llm = mocker.Mock()
    fake_llm.with_structured_output.return_value = mocker.Mock()
    mocker.patch("app.agent.aiops.planner.planner_prompt", FakePlannerPrompt())
    mocker.patch("app.agent.aiops.planner.ChatQwen", return_value=fake_llm)

    result = await planner_module.planner(
        {
            "input": "诊断 CLOSE_WAIT 告警",
            "plan": [],
            "past_steps": [],
            "knowledge_sources": [],
            "response": "",
        }
    )

    assert result["plan"] == ["查询连接状态"]
    assert result["knowledge_sources"] == [
        {
            "file_name": "连接排查.md",
            "headers": ["连接异常", "CLOSE_WAIT"],
        }
    ]


async def test_generate_response_appends_knowledge_source_footer(mocker):
    """最终 AIOps 报告应由程序统一追加来源，而不是要求模型生成来源。"""
    replanner_module = importlib.import_module("app.agent.aiops.replanner")

    class FakeResponseChain:
        async def ainvoke(self, _inputs):
            return replanner_module.Response(response="## 诊断结论\n连接存在 CLOSE_WAIT 堆积。")

    class FakeResponsePrompt:
        def __or__(self, _other):
            return FakeResponseChain()

    fake_llm = mocker.Mock()
    fake_llm.with_structured_output.return_value = mocker.Mock()
    mocker.patch("app.agent.aiops.replanner.response_prompt", FakeResponsePrompt())

    result = await replanner_module._generate_response(
        {
            "input": "诊断 CLOSE_WAIT 告警",
            "plan": [],
            "past_steps": [("查询连接状态", "发现 CLOSE_WAIT 连接")],
            # 故意重复，验证最终展示层仍会去重。
            "knowledge_sources": [
                {"file_name": "连接排查.md", "headers": ["连接异常", "CLOSE_WAIT"]},
                {"file_name": "连接排查.md", "headers": ["连接异常", "CLOSE_WAIT"]},
            ],
            "response": "",
        },
        fake_llm,
    )

    assert "## 诊断结论" in result["response"]
    assert "### 参考来源" in result["response"]
    assert result["response"].count("连接排查.md > 连接异常 > CLOSE_WAIT") == 1
