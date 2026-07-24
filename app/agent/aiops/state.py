"""
通用 Plan-Execute-Replan 状态定义
基于 LangGraph 官方教程实现
"""

from typing import Any, List, TypedDict, Annotated
import operator

from app.tools.knowledge_tool import SourceReference


class PlanExecuteState(TypedDict):
    """Plan-Execute-Replan 状态"""
    
    # 用户输入（任务描述）
    input: str
    
    # 执行计划（步骤列表）
    plan: List[str]
    
    # 已执行的步骤历史
    # 使用 operator.add 实现追加式更新（而非覆盖）
    past_steps: Annotated[List[tuple], operator.add]

    # Planner 预检索知识库时保存的来源。各节点返回的列表会追加合并，最终报告
    # 统一格式化展示；每项只保存文件名和标题层级，不保存完整文档正文。
    knowledge_sources: Annotated[List[SourceReference], operator.add]

    # 保存当前等待人工审批的工具调用信息
    pending_approval: dict[str, Any] | None

    # 保存每次人工批准或拒绝的审批记录
    # 使用 operator.add 将新记录追加到历史记录列表中
    approval_history: Annotated[List[dict[str, Any]], operator.add]
    
    # 最终响应/报告
    response: str
