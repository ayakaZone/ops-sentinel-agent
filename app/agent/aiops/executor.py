"""
Executor 节点：执行单个步骤
基于 LangGraph 官方教程实现
"""

from typing import Dict, Any
from uuid import uuid4
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.config import config
from app.tools import AIOPS_LOCAL_AGENT_TOOLS
from app.agent.mcp_client import get_mcp_client_with_retry
from .risk_policy import assess_tool_risk
from .state import PlanExecuteState


async def executor(state: PlanExecuteState) -> Command | Dict[str, Any]:
    """
    执行节点：执行计划中的下一个步骤
    
    使用 LangGraph 的 ToolNode 自动处理工具调用
    """
    logger.info("=== Executor：执行步骤 ===")

    plan = state.get("plan", [])

    # 如果计划为空，不执行
    if not plan:
        logger.info("计划为空，跳过执行")
        return Command(goto="replanner")

    # 取出第一个步骤
    task = plan[0]
    logger.info(f"当前任务: {task}")

    try:
        # 获取本地工具
        local_tools = list(AIOPS_LOCAL_AGENT_TOOLS)

        # 获取 MCP 工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 合并所有工具
        all_tools = local_tools + mcp_tools

        # 创建 LLM（绑定工具）
        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0
        )
        llm_with_tools = llm.bind_tools(all_tools)

        # 创建工具节点（自动执行工具调用）
        tool_node = ToolNode(all_tools)

        # 构建消息（只包含当前步骤，避免原始任务干扰）
        messages = [
            SystemMessage(content="""你是一个能力强大的助手，负责执行具体的任务步骤。

你可以使用各种工具来完成任务。对于每个步骤：
1. 理解步骤的目标
2. 选择合适的工具，如果已经指定了工具，则使用指定的工具
3. 调用工具获取信息
4. 返回执行结果

注意：
- 如果工具调用失败，请说明失败原因
- 不要编造数据，只返回实际获取的信息
- 执行结果要清晰、准确
- 专注于当前步骤，不要考虑其他任务"""),
            HumanMessage(content=f"请执行以下任务: {task}")
        ]

        # 第一步：LLM 决定是否调用工具
        llm_response = await llm_with_tools.ainvoke(messages)
        logger.info(f"LLM 响应类型: {type(llm_response)}")

        # 第二步：如果有工具调用，执行工具
        if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
            logger.info(f"检测到 {len(llm_response.tool_calls)} 个工具调用")

            # 遍历模型返回的工具调用列表
            for tool_call in llm_response.tool_calls:
                # 获取本次工具调用的工具名称与参数
                tool_name = tool_call.get("name", "")
                tool_arguments = tool_call.get("args", {})

                # 调用 assess_tool_risk（工具风险分级）判断是否需要人工审批
                assessment = assess_tool_risk(tool_name, tool_arguments)

                # 如果当前工具属于高风险操作，则构造待审批操作信息
                if assessment["requires_approval"]:
                    pending_approval = {
                        # 生成本次审批单的唯一 ID
                        "approval_id": str(uuid4()),
                        # 保存当前计划步骤
                        "task": task,
                        # 保存待调用的工具名称
                        "tool_name": tool_name,
                        # 保存调用该工具需要的参数
                        "arguments": tool_arguments,
                        # 保存模型本次工具调用的 ID
                        "tool_call_id": tool_call.get("id", ""),
                        # 保存工具风险等级
                        "risk_level": assessment["risk_level"],
                        # 保存需要人工审批的原因
                        "reason": assessment["reason"],
                    }
                    logger.warning(
                        "检测到需要人工审批的工具调用：{}，已暂停执行",
                        tool_name,
                    )
                    # 跳转到 request_human_approval（请求人工审批）节点
                    return Command(
                        goto="request_human_approval",
                        update={"pending_approval": pending_approval},
                    )
            
            # 使用 ToolNode 自动执行工具
            messages.append(llm_response)
            tool_messages = await tool_node.ainvoke({"messages": messages})
            
            # 第三步：将工具结果返回给 LLM 生成最终答案
            messages.extend(tool_messages["messages"])
            final_response = await llm_with_tools.ainvoke(messages)
            result = final_response.content if hasattr(final_response, 'content') else str(final_response)
        else:
            # 没有工具调用，直接使用 LLM 的输出
            logger.info("LLM 未调用工具，直接返回结果")
            result = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        logger.info(f"步骤执行完成，结果长度: {len(result)}")

        # 返回更新：移除已执行的步骤，添加执行历史
        return Command(
            goto="replanner",
            update={
                "plan": plan[1:],  # 移除第一个步骤
                "past_steps": [(task, result)],  # 使用 operator.add 追加
            },
        )

    except Exception as e:
        logger.error(f"执行步骤失败: {e}", exc_info=True)
        return Command(
            goto="replanner",
            update={
                "plan": plan[1:],
                "past_steps": [(task, f"执行失败: {str(e)}")],
            },
        )
