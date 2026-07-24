"""
AIOps 请求和响应模型
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class AIOpsRequest(BaseModel):
    """AIOps 诊断请求"""
    
    session_id: Optional[str] = Field(
        default="default",
        description="会话ID，用于追踪诊断历史"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session-123"
            }
        }


class AIOpsApprovalRequest(BaseModel):
    """AIOps 高风险操作的人工审批请求。"""

    session_id: str = Field(description="与暂停工作流对应的会话 ID")
    approval_id: str = Field(description="approval_required 事件返回的审批单 ID")
    decision: Literal["approved", "rejected"] = Field(description="人工审批决定")
    comment: str = Field(default="", max_length=500, description="审批备注，可选")


class AlertInfo(BaseModel):
    """告警信息"""
    alertname: str
    severity: str
    instance: str
    duration: str
    description: Optional[str] = None


class DiagnosisResponse(BaseModel):
    """诊断响应（非流式）"""
    
    code: int = 200
    message: str = "success"
    data: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "message": "success",
                "data": {
                    "status": "completed",
                    "target_alert": {
                        "alertname": "HighCPUUsage",
                        "severity": "critical"
                    },
                    "diagnosis": {
                        "root_cause": "数据库连接池耗尽",
                        "recommendations": ["扩容数据库连接池", "优化SQL查询"]
                    }
                }
            }
        }
