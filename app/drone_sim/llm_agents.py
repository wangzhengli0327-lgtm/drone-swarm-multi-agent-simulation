from __future__ import annotations

import json
from typing import Any

from .llm_provider import CompatibleAPIError, OpenAICompatibleClient


ALLOWED_AGENTS = {
    "ScenarioAgent",
    "SwarmStatusAgent",
    "CoverageAssessmentAgent",
    "TaskAllocationAgent",
    "PathPlanningAgent",
    "SafetyReviewAgent",
}


class LLMMeetingPanel:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    def deliberate(
        self,
        *,
        blackboard: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system_prompt = """你是一个抽象无人系统集群教学仿真的多专家会议引擎。
所有坐标仅是 60x60 教学网格，不是真实地点；不得生成真实航线、真实部署、武器使用、目标选择、攻击建议或规避现实防御的方法。
你必须让不同专家基于各自职责独立发言，允许相互反对，并只从给定候选 ID 中推荐一个方案。
只返回有效 JSON，不要 Markdown。"""
        schema_instruction = {
            "opinions": [
                {
                    "agent": "CoverageAssessmentAgent",
                    "focus": "覆盖收益",
                    "observation": "基于共享态势的观察",
                    "recommendation": "对候选方案的建议",
                    "evidence": "引用的输入指标",
                }
            ],
            "reviews": [
                {
                    "candidate_id": "A",
                    "agent": "SafetyReviewAgent",
                    "stance": "支持/质询/反对",
                    "comment": "对候选指标的具体评价",
                }
            ],
            "recommended_candidate_id": "A",
            "selection_reason": "综合覆盖、风险、距离和均衡度的选择理由",
        }
        user_prompt = (
            "请基于以下共享态势和候选方案完成一次多专家会商。"
            "opinions 必须恰好 6 条，分别覆盖场景、集群状态、覆盖、任务分配、路径和安全角色；"
            "reviews 最多 12 条，应体现真实分歧并引用候选指标。每个文本字段控制在 120 个汉字以内。\n"
            f"共享态势：{json.dumps(blackboard, ensure_ascii=False)}\n"
            f"候选方案：{json.dumps(candidates, ensure_ascii=False)}\n"
            f"返回结构示例：{json.dumps(schema_instruction, ensure_ascii=False)}"
        )
        result = self.client.chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        return self._validate(result, {item["candidate_id"] for item in candidates})

    @staticmethod
    def _validate(result: dict[str, Any], candidate_ids: set[str]) -> dict[str, Any]:
        opinions: list[dict[str, str]] = []
        for item in result.get("opinions", []):
            if not isinstance(item, dict) or item.get("agent") not in ALLOWED_AGENTS:
                continue
            opinions.append(
                {
                    "agent": str(item["agent"]),
                    "focus": str(item.get("focus", "领域研判"))[:80],
                    "observation": str(item.get("observation", ""))[:600],
                    "recommendation": str(item.get("recommendation", ""))[:600],
                    "evidence": str(item.get("evidence", "共享态势"))[:300],
                }
            )
        reviews: list[dict[str, str]] = []
        for item in result.get("reviews", []):
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("candidate_id", ""))
            agent = str(item.get("agent", ""))
            if candidate_id not in candidate_ids or agent not in ALLOWED_AGENTS:
                continue
            stance = str(item.get("stance", "质询"))
            if stance not in {"支持", "质询", "反对"}:
                stance = "质询"
            reviews.append(
                {
                    "candidate_id": candidate_id,
                    "agent": agent,
                    "stance": stance,
                    "comment": str(item.get("comment", ""))[:600],
                }
            )
        recommendation = str(result.get("recommended_candidate_id", ""))
        if recommendation not in candidate_ids:
            recommendation = ""
        return {
            "opinions": opinions,
            "reviews": reviews,
            "recommended_candidate_id": recommendation,
            "selection_reason": str(result.get("selection_reason", ""))[:1000],
        }


AgnesMeetingPanel = LLMMeetingPanel
AgnesAPIError = CompatibleAPIError

__all__ = ["LLMMeetingPanel", "AgnesMeetingPanel", "CompatibleAPIError", "AgnesAPIError"]
