"""用量查询接口

给前端展示「当前会话上下文窗口用量」+「今日调用次数」两类信息。
"""

from fastapi import APIRouter
from loguru import logger

from app.config import config
from app.services.rag_agent_service import rag_agent_service
from app.services.usage_tracker import daily_usage_counter

router = APIRouter()


def _build_daily_usage(category: str, limit: int) -> dict:
    """构建某个分类的今日调用次数用量信息"""
    used = daily_usage_counter.get_today_count(category)
    percent = round(used / limit * 100, 1) if limit else 0.0
    return {"used": used, "limit": limit, "percent": percent}


@router.get("/usage")
async def get_usage(session_id: str):
    """
    查询用量情况

    Args:
        session_id: 会话 ID，用于计算当前会话的上下文窗口 token 用量

    Returns:
        当前会话 token 用量 + 今日 chat/aiops 调用次数
    """
    try:
        data = {
            "session_token_usage": rag_agent_service.get_session_token_usage(session_id),
            "daily_calls": {
                "chat": _build_daily_usage("chat", config.daily_chat_limit),
                "aiops": _build_daily_usage("aiops", config.daily_aiops_limit),
            },
        }
        return {"code": 200, "message": "success", "data": data}

    except Exception as e:
        logger.error(f"查询用量失败: {e}")
        return {"code": 500, "message": "error", "data": None}
