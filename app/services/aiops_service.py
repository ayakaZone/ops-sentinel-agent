"""
通用 Plan-Execute-Replan 服务
基于 LangGraph 官方教程实现
"""

from typing import AsyncGenerator, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from loguru import logger

from app.agent.aiops import (
    PlanExecuteState,
    execute_approved_action,
    executor,
    handle_rejection,
    planner,
    replanner,
    request_human_approval,
)
from app.config import config
from app.services.usage_tracker import daily_usage_counter


# 节点名称常量
NODE_PLANNER = "planner"
NODE_EXECUTOR = "executor"
NODE_REPLANNER = "replanner"
NODE_REQUEST_HUMAN_APPROVAL = "request_human_approval"
NODE_EXECUTE_APPROVED_ACTION = "execute_approved_action"
NODE_HANDLE_REJECTION = "handle_rejection"


class AIOpsService:
    """通用 Plan-Execute-Replan 服务"""

    def __init__(self):
        """初始化服务

        短期记忆（会话历史）持久化实例由应用启动时 configure_checkpointer() 注入，
        在此之前 checkpointer/graph 都是 None——正常运行时 FastAPI lifespan 会在
        收到第一个请求之前完成注入。
        """
        self.checkpointer = None
        self.graph = None
        logger.info("Plan-Execute-Replan Service 初始化完成")

    def configure_checkpointer(self, checkpointer):
        """
        注入持久化的短期记忆存储（由 main.py 的 lifespan 在应用启动时调用）

        Args:
            checkpointer: 短期记忆（会话历史）持久化实例，如 AsyncSqliteSaver
        """
        self.checkpointer = checkpointer
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建 Plan-Execute-Replan 工作流"""
        logger.info("构建工作流图...")

        # 创建状态图
        workflow = StateGraph(PlanExecuteState)

        # 添加节点
        workflow.add_node(NODE_PLANNER, planner)      # 制定计划
        workflow.add_node(NODE_EXECUTOR, executor)  # 执行步骤
        workflow.add_node(NODE_REPLANNER, replanner)  # 重新规划
        # 节点名与函数名保持完全一致，阅读 goto="request_human_approval" 时
        # 可以直接定位到 approval.py 中同名的 request_human_approval 函数。
        workflow.add_node(
            NODE_REQUEST_HUMAN_APPROVAL,
            request_human_approval,
        )
        workflow.add_node(NODE_EXECUTE_APPROVED_ACTION, execute_approved_action)
        workflow.add_node(NODE_HANDLE_REJECTION, handle_rejection)

        # 设置入口点
        workflow.set_entry_point(NODE_PLANNER)

        # 定义边
        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)     # planner -> executor
        # 审批通过后，从执行已批准操作节点跳转到 replanner 节点
        workflow.add_edge(NODE_EXECUTE_APPROVED_ACTION, NODE_REPLANNER)

        # 审批拒绝后，从处理拒绝节点跳转到 replanner 节点
        workflow.add_edge(NODE_HANDLE_REJECTION, NODE_REPLANNER)

        # replanner 的条件边
        def should_continue(state: PlanExecuteState) -> str:
            """判断是否继续执行"""
            # 如果已经生成了最终响应，结束
            if state.get("response"):
                logger.info("已生成最终响应，结束流程")
                return END

            # 如果还有计划步骤，继续执行
            plan = state.get("plan", [])
            if plan:
                logger.info(f"继续执行，剩余 {len(plan)} 个步骤")
                return NODE_EXECUTOR

            # 计划为空但没有响应，返回 replanner 生成响应
            logger.info("计划执行完毕，生成最终响应")
            return END

        workflow.add_conditional_edges(
            NODE_REPLANNER,
            should_continue,
            {
                NODE_EXECUTOR: NODE_EXECUTOR,
                END: END
            }
        )

        # 编译工作流
        compiled_graph = workflow.compile(checkpointer=self.checkpointer)

        logger.info("工作流图构建完成")
        return compiled_graph

    async def execute(
        self,
        user_input: str,
        session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行 Plan-Execute-Replan 流程

        Args:
            user_input: 用户的任务描述
            session_id: 会话ID

        Yields:
            Dict[str, Any]: 流式事件
        """
        logger.info(f"[会话 {session_id}] 开始执行任务: {user_input}")

        try:
            # 初始化状态
            initial_state: PlanExecuteState = {
                "input": user_input,
                "plan": [],
                "past_steps": [],
                "knowledge_sources": [],
                "pending_approval": None,
                "approval_history": [],
                "response": ""
            }

            async for event in self._run_graph(initial_state, session_id):
                yield event

        except Exception as e:
            logger.error(f"[会话 {session_id}] 任务执行失败: {e}", exc_info=True)
            yield {
                "type": "error",
                "stage": "error",
                "message": f"任务执行出错: {str(e)}"
            }

    async def resume_approval(
        self,
        session_id: str,
        approval_id: str,
        decision: Literal["approved", "rejected"],
        comment: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """resume_approval（提交人工审批结果并恢复工作流）。"""
        config_dict = self._get_graph_config(session_id)
        current_state = await self.graph.aget_state(config_dict)
        pending_action = current_state.values.get("pending_approval") if current_state.values else None

        if not pending_action:
            raise ValueError("当前会话没有等待审批的高风险操作")
        if pending_action.get("approval_id") != approval_id:
            raise ValueError("审批单 ID 与当前待审批操作不匹配")

        logger.info("[会话 {}] 收到人工审批结果：{}", session_id, decision)
        # 使用 Command(resume=...) 将人工审批结果传回暂停的 interrupt
        resume_command = Command(resume={"decision": decision, "comment": comment})
        async for event in self._run_graph(resume_command, session_id):
            yield event

    def _get_graph_config(self, session_id: str) -> Dict[str, Any]:
        """_get_graph_config（构造 LangGraph 会话配置）。"""
        return {"configurable": {"thread_id": session_id}}

    async def _run_graph(
        self,
        graph_input: PlanExecuteState | Command,
        session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """_run_graph（执行或恢复图，并把 LangGraph 事件转换为 SSE 事件）。"""
        config_dict = self._get_graph_config(session_id)

        async for event in self.graph.astream(
            input=graph_input,
            config=config_dict,
            stream_mode="updates",
        ):
            # 判断当前事件是否为 interrupt（工作流暂停）事件
            if "__interrupt__" in event:
                # 获取 interrupt 返回的审批单列表
                interrupt_items = event["__interrupt__"]

                # 获取第一张审批单
                first_interrupt = interrupt_items[0] if interrupt_items else None

                # 获取 interrupt 中保存的审批单内容
                approval_payload = getattr(first_interrupt, "value", {})

                # 返回 approval_required（需要人工审批）SSE 事件给前端
                yield {
                    "type": "approval_required",
                    "stage": "human_approval",
                    "message": "检测到高风险操作，等待人工审批",
                    "session_id": session_id,
                    "approval": approval_payload,
                }
                logger.info("[会话 {}] 工作流已暂停，等待人工审批", session_id)
                return

            for node_name, node_output in event.items():
                logger.info(f"节点 '{node_name}' 输出事件")
                if node_name == NODE_PLANNER:
                    yield self._format_planner_event(node_output)
                elif node_name == NODE_EXECUTOR:
                    yield self._format_executor_event(node_output)
                elif node_name == NODE_REPLANNER:
                    yield self._format_replanner_event(node_output)
                elif node_name == NODE_EXECUTE_APPROVED_ACTION:
                    yield self._format_approved_action_event(node_output)
                elif node_name == NODE_HANDLE_REJECTION:
                    yield self._format_rejection_event(node_output)

        final_state = await self.graph.aget_state(config_dict)
        final_response = final_state.values.get("response", "") if final_state.values else ""
        yield {
            "type": "complete",
            "stage": "complete",
            "message": "任务执行完成",
            "response": final_response,
        }
        logger.info(f"[会话 {session_id}] 任务执行完成")

    async def diagnose(
        self,
        session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        AIOps 诊断接口（兼容旧接口）

        Args:
            session_id: 会话ID

        Yields:
            Dict[str, Any]: 诊断过程的流式事件
        """
        # 使用固定的 AIOps 任务描述
        from textwrap import dedent
        aiops_task = dedent("""诊断当前系统是否存在告警，如果存在告警请详细分析告警原因并生成诊断报告，诊断报告输出格式要求：
                ```
                # 告警分析报告

                ---

                ## 📋 活跃告警清单

                | 告警名称 | 级别 | 目标服务 | 首次触发时间 | 最新触发时间 | 状态 |
                |---------|------|----------|-------------|-------------|------|
                | [告警1名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |
                | [告警2名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |

                ---

                ## 🔍 告警根因分析1 - [告警名称]

                ### 告警详情
                - **告警级别**: [级别]
                - **受影响服务**: [服务名]
                - **持续时间**: [X分钟]

                ### 症状描述
                [根据监控指标描述症状]

                ### 日志证据
                [引用查询到的关键日志]

                ### 根因结论
                [基于证据得出的根本原因]

                ---

                ## 🛠️ 处理方案执行1 - [告警名称]

                ### 已执行的排查步骤
                1. [步骤1]
                2. [步骤2]

                ### 处理建议
                [给出具体的处理建议]

                ### 预期效果
                [说明预期的效果]

                ---

                ## 🔍 告警根因分析2 - [告警名称]
                [如果有第2个告警，重复上述格式]

                ---

                ## 📊 结论

                ### 整体评估
                [总结所有告警的整体情况]

                ### 关键发现
                - [发现1]
                - [发现2]

                ### 后续建议
                1. [建议1]
                2. [建议2]

                ### 风险评估
                [评估当前风险等级和影响范围]
                ```

                **重要提醒**：
                - 最终输出必须是纯 Markdown 文本，不要包含 JSON 结构
                - 所有内容必须基于工具查询的真实数据，严禁编造
                - 如果某个步骤失败，在结论中如实说明，不要跳过""")

        async for event in self.execute(aiops_task, session_id):
            # 转换事件格式以兼容旧的 API
            if event.get("type") == "complete":
                report = event.get("response", "")

                # 软限流：超过今日 AIOps 调用次数阈值时，在报告末尾附加友好提示（不拦截请求）
                reminder = daily_usage_counter.increment_and_get_reminder("aiops", config.daily_aiops_limit)
                if reminder:
                    report += reminder

                # 将 response 包装为 diagnosis 格式
                yield {
                    "type": "complete",
                    "stage": "diagnosis_complete",
                    "message": "诊断流程完成",
                    "diagnosis": {
                        "status": "completed",
                        "report": report
                    }
                }
            else:
                yield event

    def _format_planner_event(self, state: Dict | None) -> Dict:
        """格式化 Planner 节点事件"""
        if not state:
            return {
                "type": "status",
                "stage": "planner",
                "message": "规划节点执行中"
            }

        plan = state.get("plan", [])

        return {
            "type": "plan",
            "stage": "plan_created",
            "message": f"执行计划已制定，共 {len(plan)} 个步骤",
            "plan": plan
        }

    def _format_executor_event(self, state: Dict | None) -> Dict:
        """格式化 Executor 节点事件"""
        if not state:
            return {
                "type": "status",
                "stage": "executor",
                "message": "执行节点运行中"
            }

        plan = state.get("plan", [])
        past_steps = state.get("past_steps", [])

        if past_steps:
            last_step, _ = past_steps[-1]
            return {
                "type": "step_complete",
                "stage": "step_executed",
                "message": f"步骤执行完成 ({len(past_steps)}/{len(past_steps) + len(plan)})",
                "current_step": last_step,
                "remaining_steps": len(plan)
            }
        else:
            return {
                "type": "status",
                "stage": "executor",
                "message": "开始执行步骤"
            }

    def _format_replanner_event(self, state: Dict | None) -> Dict:
        """格式化 Replanner 节点事件"""
        if not state:
            return {
                "type": "status",
                "stage": "replanner",
                "message": "评估节点运行中"
            }

        response = state.get("response", "")
        plan = state.get("plan", [])

        if response:
            # 已生成最终响应
            return {
                "type": "report",
                "stage": "final_report",
                "message": "最终报告已生成",
                "report": response
            }
        else:
            # 重新规划
            return {
                "type": "status",
                "stage": "replanner",
                "message": f"评估完成，{'继续执行剩余步骤' if plan else '准备生成最终响应'}",
                "remaining_steps": len(plan)
            }

    def _format_approved_action_event(self, state: Dict | None) -> Dict:
        """_format_approved_action_event（格式化已批准操作的执行事件）。"""
        past_steps = state.get("past_steps", []) if state else []
        step_name = past_steps[-1][0] if past_steps else "高风险操作"
        return {
            "type": "step_complete",
            "stage": "approved_action_executed",
            "message": "人工审批已通过，高风险操作已执行",
            "current_step": step_name,
        }

    def _format_rejection_event(self, state: Dict | None) -> Dict:
        """_format_rejection_event（格式化人工拒绝事件）。"""
        past_steps = state.get("past_steps", []) if state else []
        step_name = past_steps[-1][0] if past_steps else "高风险操作"
        return {
            "type": "step_rejected",
            "stage": "approval_rejected",
            "message": "人工已拒绝高风险操作，系统未执行该操作",
            "current_step": step_name,
        }


# 全局单例
aiops_service = AIOpsService()
