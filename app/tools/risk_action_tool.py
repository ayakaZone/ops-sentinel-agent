"""仅用于验证 AIOps 人工审批链路的模拟高风险工具。"""

from langchain_core.tools import tool


@tool
def restart_mock_service(service_name: str, environment: str = "production") -> str:
    """restart_mock_service（模拟重启服务）。

    此工具**不会连接真实服务器，也不会实际重启任何服务**。它只返回一条模拟执行结果，
    用于验证 AIOps Agent 在遇到“重启服务”这类高风险操作时，是否会先暂停、等待人工审批，
    并且只在批准后执行。

    Args:
        service_name: 要模拟重启的服务名称，例如 ``order-service``。
        environment: 目标环境，例如 ``production`` 或 ``test``。
    """
    return (
        f"已模拟重启服务：{service_name}（环境：{environment}）。"
        "本次仅验证 Human-in-the-loop 审批链路，未对真实基础设施执行操作。"
    )
