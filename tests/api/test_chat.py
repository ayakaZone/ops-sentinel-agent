"""/api/chat 相关接口的集成测试

用 httpx.ASGITransport 直接把请求打进真实的 FastAPI app（走真实的路由匹配、
参数校验、响应包装），但 mock 掉 rag_agent_service.query 这个重量级依赖——
不然每跑一次测试都要真实调一次 LLM，慢且要花钱。

跟 test_knowledge_tool.py 同样的原因，import app.main 会触发 vector_store_manager
的模块级单例连接真实 Milvus，所以整个文件标记为 integration，CI 默认跳过。
"""

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.integration


@pytest.fixture
async def client():
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_chat_returns_answer_on_success(client, mocker):
    """正常问答：mock 掉 query，验证接口把答案正确包装进统一响应格式"""
    mocker.patch(
        "app.api.chat.rag_agent_service.query",
        return_value="CPU 使用率过高通常需要先检查 top 命令输出。",
    )

    response = await client.post(
        "/api/chat", json={"id": "test-session-1", "question": "CPU 使用率过高怎么办"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["data"]["success"] is True
    assert "CPU" in body["data"]["answer"]
    assert body["data"]["errorMessage"] is None


async def test_chat_wraps_exception_as_error_response(client, mocker):
    """query 抛异常时，接口不应该 500 崩掉，而是包装成统一的错误响应"""
    mocker.patch(
        "app.api.chat.rag_agent_service.query",
        side_effect=RuntimeError("DashScope 服务不可用"),
    )

    response = await client.post(
        "/api/chat", json={"id": "test-session-2", "question": "随便问点什么"}
    )

    assert response.status_code == 200  # 接口本身照样返回 200，错误信息包装在 body 里
    body = response.json()
    assert body["code"] == 500
    assert body["data"]["success"] is False
    assert "DashScope 服务不可用" in body["data"]["errorMessage"]


async def test_clear_session_success(client, mocker):
    """清空会话：验证请求参数正确传给 service，并按结果包装响应"""
    mock_clear = mocker.patch(
        "app.api.chat.rag_agent_service.clear_session", return_value=True
    )

    response = await client.post("/api/chat/clear", json={"session_id": "test-session-3"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    mock_clear.assert_called_once_with("test-session-3")


async def test_get_session_info_returns_history(client, mocker):
    """查询会话历史：验证 message_count 和返回的 history 列表对得上"""
    fake_history = [
        {"role": "human", "content": "你好"},
        {"role": "ai", "content": "你好，有什么可以帮你？"},
    ]
    mocker.patch(
        "app.api.chat.rag_agent_service.get_session_history", return_value=fake_history
    )

    response = await client.get("/api/chat/session/test-session-4")

    assert response.status_code == 200
    body = response.json()
    assert body["message_count"] == 2
    assert body["history"] == fake_history
