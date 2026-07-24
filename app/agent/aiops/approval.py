"""AIOps Human-in-the-loop 人工审批节点。"""

from datetime import datetime
from typing import Any, Dict

from langgraph.types import Command, interrupt
from loguru import logger

from app.tools import AIOPS_LOCAL_AGENT_TOOLS
from .state import PlanExecuteState


NODE_REPLANNER = "replanner"
NODE_EXECUTE_APPROVED_ACTION = "execute_approved_action"
NODE_HANDLE_REJECTION = "handle_rejection"


def request_human_approval(state: PlanExecuteState) -> Command:
    """request_human_approval（请求人工审批）。"""

    # 从 State 获取 Executor 保存的待审批操作信息
    pending_action = state.get("pending_approval")

    # 如果没有待审批操作，则跳转到 replanner（重新规划）节点
    if not pending_action:
        logger.error("审批节点未找到待审批操作，返回重新规划节点")
        return Command(goto=NODE_REPLANNER)

    # 构造发送给前端的审批单信息
    approval_payload = {
        # 审批单唯一 ID，用于后续提交审批结果时校验
        "approval_id": pending_action["approval_id"],
        # 当前准备执行的计划步骤
        "task": pending_action["task"],
        # 待调用的工具名称
        "tool_name": pending_action["tool_name"],
        # 待调用工具的参数
        "arguments": pending_action["arguments"],
        # 工具风险等级
        "risk_level": pending_action["risk_level"],
        # 需要人工审批的原因
        "reason": pending_action["reason"],
    }

    # 调用 interrupt（暂停工作流），并将审批单返回给前端
    decision_data = interrupt(approval_payload)

    # 获取人工提交的批准或拒绝结果
    decision = decision_data.get("decision") if isinstance(decision_data, dict) else None

    # 获取人工填写的审批备注
    comment = decision_data.get("comment", "") if isinstance(decision_data, dict) else ""

    # 判断人工是否批准当前操作
    approved = decision == "approved"

    # 构造本次人工审批的历史记录
    approval_record = {
        # 复制审批单中的工具、参数和风险信息
        **approval_payload,
        # 保存最终审批结果
        "decision": "approved" if approved else "rejected",
        # 保存人工填写的备注
        "comment": comment,
        # 保存作出审批决定的时间
        "decided_at": datetime.now().isoformat(timespec="seconds"),
    }

    # 如果人工批准，则跳转到 execute_approved_action（执行已批准操作）节点
    if approved:
        logger.info("人工已批准高风险操作：{}", pending_action["tool_name"])
        return Command(
            goto=NODE_EXECUTE_APPROVED_ACTION,
            # 将本次审批记录追加到审批历史中
            update={"approval_history": [approval_record]},
        )

    # 如果人工拒绝，则跳转到 handle_rejection（处理拒绝）节点
    logger.info("人工拒绝高风险操作：{}", pending_action["tool_name"])
    return Command(
        goto=NODE_HANDLE_REJECTION,
        update={"approval_history": [approval_record]},
    )


async def execute_approved_action(state: PlanExecuteState) -> Dict[str, Any]:
    """execute_approved_action（执行已批准的高风险操作）。"""

    # 获取当前待审批操作
    pending_action = state.get("pending_approval")

    # 获取当前剩余计划步骤
    plan = state.get("plan", [])

    # 如果没有待执行的审批操作或计划步骤，则清空待审批状态
    if not pending_action or not plan:
        return {"pending_approval": None}

    # 获取人工已批准调用的工具名称
    tool_name = pending_action["tool_name"]

    # 获取人工已批准调用的工具参数
    tool_arguments = pending_action["arguments"]

    # 将 AIOps 本地工具列表转换为“工具名: 工具对象”的字典
    tool_map = {tool.name: tool for tool in AIOPS_LOCAL_AGENT_TOOLS}

    # 根据工具名称获取真正可调用的工具对象
    target_tool = tool_map.get(tool_name)

    # 如果工具不存在，则记录工具未执行的原因
    if target_tool is None:
        result = f"审批已通过，但未找到可执行工具 {tool_name}，操作未执行。"
    else:
        try:
            # 调用 ainvoke（异步执行工具）真正执行已批准的工具
            tool_result = await target_tool.ainvoke(tool_arguments)

            # 将工具执行结果转换为字符串
            result = str(tool_result)
        except Exception as error:
            logger.error("已批准操作执行失败: {}", error, exc_info=True)

            # 记录工具执行失败的原因
            result = f"审批已通过，但工具执行失败: {error}"

    # 获取当前计划步骤名称
    task = pending_action["task"]

    # 移除已处理的计划步骤，保存执行结果，清空待审批操作
    return {
        "plan": plan[1:],
        "past_steps": [(task, result)],
        "pending_approval": None,
    }


def handle_rejection(state: PlanExecuteState) -> Dict[str, Any]:
    """handle_rejection（处理人工拒绝）。"""

    # 获取当前待审批操作
    pending_action = state.get("pending_approval")

    # 获取当前剩余计划步骤
    plan = state.get("plan", [])

    # 如果没有待审批操作或计划步骤，则清空待审批状态
    if not pending_action or not plan:
        return {"pending_approval": None}

    # 获取被拒绝操作对应的计划步骤名称
    task = pending_action["task"]

    # 构造“人工拒绝，工具未执行”的步骤结果
    result = (
        f"高风险操作 {pending_action['tool_name']} 未执行："
        "人工审批已拒绝，系统已停止该操作。"
    )
    # 移除已处理的计划步骤，保存拒绝结果，清空待审批操作
    return {
        "plan": plan[1:],
        "past_steps": [(task, result)],
        "pending_approval": None,
    }
