"""AIOps Human-in-the-loop 的离线测试，不调用真实模型、MCP 或基础设施。"""

import operator
from typing import Any, Annotated, TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from app.agent.aiops.approval import (
    execute_approved_action,
    handle_rejection,
    request_human_approval,
)
from app.agent.aiops.risk_policy import assess_tool_risk


class ApprovalTestState(TypedDict):
    """ApprovalTestState（审批图测试专用状态）。"""

    plan: list[str]
    past_steps: Annotated[list[tuple[str, str]], operator.add]
    pending_approval: dict[str, Any] | None
    approval_history: Annotated[list[dict[str, Any]], operator.add]


def _build_approval_test_graph():
    """_build_approval_test_graph（构造只包含审批分支的最小 LangGraph）。"""
    workflow = StateGraph(ApprovalTestState)
    # 测试图沿用正式图的命名规则：节点名和函数名一致。
    workflow.add_node("request_human_approval", request_human_approval)
    workflow.add_node("execute_approved_action", execute_approved_action)
    workflow.add_node("handle_rejection", handle_rejection)
    workflow.set_entry_point("request_human_approval")
    workflow.add_edge("execute_approved_action", END)
    workflow.add_edge("handle_rejection", END)
    return workflow.compile(checkpointer=MemorySaver())


def _pending_action() -> dict[str, Any]:
    """_pending_action（构造一份待审批的模拟重启操作）。"""
    return {
        "approval_id": "approval-test-001",
        "task": "模拟重启 order-service",
        "tool_name": "restart_mock_service",
        "arguments": {"service_name": "order-service", "environment": "production"},
        "risk_level": "high",
        "reason": "重启服务会影响业务可用性",
    }


def test_assess_tool_risk_allows_read_only_tool():
    """白名单中的日志查询工具应自动执行。"""
    result = assess_tool_risk("query_logs", {"query": "ERROR"})

    assert result["risk_level"] == "read_only"
    assert result["requires_approval"] is False


def test_assess_tool_risk_requires_approval_for_unknown_tool():
    """未分级的新工具默认走人工审批，避免绕过治理策略。"""
    result = assess_tool_risk("delete_production_resource", {"resource_id": "demo"})

    assert result["risk_level"] == "high"
    assert result["requires_approval"] is True


@pytest.mark.asyncio
async def test_approval_interrupt_then_execute_after_approved():
    """批准前图会暂停；批准后才会执行模拟重启工具。"""
    graph = _build_approval_test_graph()
    graph_config = {"configurable": {"thread_id": "approval-approved"}}
    initial_state: ApprovalTestState = {
        "plan": ["模拟重启 order-service"],
        "past_steps": [],
        "pending_approval": _pending_action(),
        "approval_history": [],
    }

    first_events = [
        event
        async for event in graph.astream(initial_state, graph_config, stream_mode="updates")
    ]

    # LangGraph 通过 __interrupt__ 返回审批单，并未执行模拟工具。
    interrupt_event = first_events[-1]
    assert "__interrupt__" in interrupt_event
    assert interrupt_event["__interrupt__"][0].value["tool_name"] == "restart_mock_service"

    resume_events = [
        event
        async for event in graph.astream(
            Command(resume={"decision": "approved", "comment": "允许执行演示"}),
            graph_config,
            stream_mode="updates",
        )
    ]
    final_state = await graph.aget_state(graph_config)

    assert "execute_approved_action" in resume_events[-1]
    assert "已模拟重启服务：order-service" in final_state.values["past_steps"][-1][1]
    assert final_state.values["approval_history"][-1]["decision"] == "approved"


@pytest.mark.asyncio
async def test_approval_rejection_does_not_execute_tool():
    """拒绝时不调用工具，只记录“未执行”的执行历史。"""
    graph = _build_approval_test_graph()
    graph_config = {"configurable": {"thread_id": "approval-rejected"}}
    initial_state: ApprovalTestState = {
        "plan": ["模拟重启 order-service"],
        "past_steps": [],
        "pending_approval": _pending_action(),
        "approval_history": [],
    }

    _ = [event async for event in graph.astream(initial_state, graph_config, stream_mode="updates")]
    resume_events = [
        event
        async for event in graph.astream(
            Command(resume={"decision": "rejected", "comment": "暂不处理"}),
            graph_config,
            stream_mode="updates",
        )
    ]
    final_state = await graph.aget_state(graph_config)

    assert "handle_rejection" in resume_events[-1]
    assert "未执行" in final_state.values["past_steps"][-1][1]
    assert final_state.values["approval_history"][-1]["decision"] == "rejected"
