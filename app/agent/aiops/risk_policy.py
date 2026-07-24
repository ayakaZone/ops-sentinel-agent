"""AIOps 工具风险分级策略。

这里把“工具能不能调用”和“调用前是否需要人确认”从 Executor 节点中拆出来，
后续接入真实的重启、扩缩容、变更配置等工具时，只需在本文件补充策略即可。
"""

from typing import Any, Literal, TypedDict


# 定义工具风险等级的可选值
RiskLevel = Literal["read_only", "high", "blocked"]


class ToolRiskAssessment(TypedDict):
    """一次工具调用的风险判定结果。"""

    risk_level: RiskLevel
    requires_approval: bool
    reason: str


# 定义可以自动执行的只读工具名称
READ_ONLY_TOOL_NAMES = {
    "retrieve_knowledge",
    "get_current_time",
    "query_prometheus_alerts",
    "query_logs",
    "query_metrics",
    "get_cpu_usage",
    "get_memory_usage",
    "get_disk_usage",
    "get_network_usage",
    "get_service_status",
}


# 定义高风险工具及其需要审批的原因
HIGH_RISK_TOOL_REASONS = {
    "restart_mock_service": "重启服务会中断现有连接并可能影响业务可用性",
}


def assess_tool_risk(tool_name: str, arguments: dict[str, Any]) -> ToolRiskAssessment:
    """assess_tool_risk（评估工具风险等级）。"""

    # 当前版本只按工具名称分级，暂不根据工具参数分级
    _ = arguments

    # 如果工具在只读白名单中，则允许 Agent 自动执行
    if tool_name in READ_ONLY_TOOL_NAMES:
        return {
            "risk_level": "read_only",
            "requires_approval": False,
            "reason": "该工具只读取知识库、监控或日志数据，不修改外部资源",
        }

    # 如果工具在高风险名单中，则要求人工审批
    if tool_name in HIGH_RISK_TOOL_REASONS:
        return {
            "risk_level": "high",
            "requires_approval": True,
            "reason": HIGH_RISK_TOOL_REASONS[tool_name],
        }

    # 如果工具未完成风险登记，则按高风险工具处理
    return {
        "risk_level": "high",
        "requires_approval": True,
        "reason": "该工具尚未完成风险分级，按保守策略需要人工确认",
    }
