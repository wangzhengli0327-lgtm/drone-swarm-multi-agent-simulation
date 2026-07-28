from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ExpertOpinion:
    agent: str
    focus: str
    observation: str
    recommendation: str
    evidence: str


@dataclass
class CandidatePlan:
    candidate_id: str
    owner_agent: str
    profile: str
    title: str
    assignments: list[dict[str, Any]]
    warnings: list[str]
    metrics: dict[str, float]
    utility: float = 0.0
    reviews: list[dict[str, str]] | None = None


@dataclass
class PlanningDecision:
    blackboard: dict[str, Any]
    opinions: list[ExpertOpinion]
    candidates: list[CandidatePlan]
    selected_candidate_id: str
    selected_profile: str
    selection_reason: str
    revision_required: bool
    llm_mode: str = "rule"
    llm_provider: str = "未配置"
    llm_status: str = "未调用"
    llm_recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "blackboard": self.blackboard,
            "opinions": [asdict(item) for item in self.opinions],
            "candidates": [asdict(item) for item in self.candidates],
            "selected_candidate_id": self.selected_candidate_id,
            "selected_profile": self.selected_profile,
            "selection_reason": self.selection_reason,
            "revision_required": self.revision_required,
            "llm_mode": self.llm_mode,
            "llm_provider": self.llm_provider,
            "llm_status": self.llm_status,
            "llm_recommendation": self.llm_recommendation,
        }


PROFILE_DEFINITIONS = [
    ("A", "CoverageAssessmentAgent", "coverage", "覆盖增益优先方案"),
    ("B", "SafetyReviewAgent", "safety", "风险约束优先方案"),
    ("C", "PathPlanningAgent", "time", "抵达时间优先方案"),
    ("D", "TaskAllocationAgent", "balanced", "任务均衡方案"),
]


def build_expert_opinions(blackboard: dict[str, Any]) -> list[ExpertOpinion]:
    coverage_gap = max(0.0, blackboard["target_coverage"] - blackboard["minimum_task_coverage"])
    return [
        ExpertOpinion(
            "ScenarioAgent",
            "场景约束",
            f"当前包含 {blackboard['task_count']} 个任务区和 {blackboard['risk_zone_count']} 个高风险区。",
            "候选方案必须让每个可服务任务区获得明确任务分配。",
            "任务区与风险区几何关系",
        ),
        ExpertOpinion(
            "SwarmStatusAgent",
            "集群可用性",
            f"可用 UAV {blackboard['active_uav_count']} 架，失效 {blackboard['failed_uav_count']} 架，低电量 {blackboard['low_battery_count']} 架。",
            "不得把失效或低电量节点纳入主任务候选方案。",
            "UAV 状态与剩余电量",
        ),
        ExpertOpinion(
            "CoverageAssessmentAgent",
            "覆盖收益",
            f"主指标最低任务区覆盖率为 {blackboard['minimum_task_coverage']:.1%}，距离目标仍差 {coverage_gap:.1%}。",
            "优先增加最低覆盖任务区的边际覆盖收益，避免重复堆叠同一落点。",
            "当前覆盖热力图与任务区覆盖明细",
        ),
        ExpertOpinion(
            "PathPlanningAgent",
            "抵达效率",
            f"本轮风险姿态为{blackboard['risk_posture_label']}，路径仍需比较安全避让与较短风险穿越。",
            "以平均抽象路径长度和风险代价共同审查候选方案。",
            "起点、候选服务点与风险区关系",
        ),
        ExpertOpinion(
            "SafetyReviewAgent",
            "风险预算",
            f"本轮最多允许 {blackboard['risk_budget']} 架 UAV 承担显著风险暴露。",
            "超过预算的方案必须修正；滚动覆盖模式下优先保留持续服务能力。",
            "动态风险姿态、累计损失与风险暴露预算",
        ),
        ExpertOpinion(
            "TaskAllocationAgent",
            "任务均衡",
            "只保证每区一架并不足以证明方案有效，还需比较各任务区预计服务覆盖。",
            "候选方案应降低任务区覆盖离散度，同时保留覆盖增益。",
            "候选落点的预计服务范围",
        ),
    ]


def score_candidate(metrics: dict[str, float], blackboard: dict[str, Any]) -> float:
    rolling_bonus = 12.0 if blackboard["coverage_mode"] == "rolling" else 0.0
    risk_penalty = {"保守": 13.0, "均衡": 8.0, "进取": 4.5}[blackboard["risk_posture_label"]]
    excess_risk = max(0.0, metrics["risk_exposure_count"] - blackboard["risk_budget"])
    severe_coverage_shortfall = max(0.0, 0.50 - metrics["minimum_projected_coverage"])
    return (
        metrics["minimum_projected_coverage"] * (48.0 + rolling_bonus)
        + metrics["average_projected_coverage"] * 22.0
        - metrics["coverage_imbalance"] * 18.0
        - metrics["risk_exposure_count"] * risk_penalty
        - excess_risk * 24.0
        - metrics["total_risk_score"] * 0.32
        - metrics["average_travel_distance"] * 0.16
        - severe_coverage_shortfall * 70.0
    )


def review_candidate(candidate: CandidatePlan, blackboard: dict[str, Any]) -> list[dict[str, str]]:
    metrics = candidate.metrics
    coverage_status = "支持" if metrics["minimum_projected_coverage"] >= blackboard["target_coverage"] * 0.78 else "质询"
    safety_status = "支持" if metrics["risk_exposure_count"] <= blackboard["risk_budget"] else "反对"
    balance_status = "支持" if metrics["coverage_imbalance"] <= 0.18 else "质询"
    return [
        {
            "agent": "CoverageAssessmentAgent",
            "stance": coverage_status,
            "comment": f"最低预计覆盖 {metrics['minimum_projected_coverage']:.1%}，平均预计覆盖 {metrics['average_projected_coverage']:.1%}。",
        },
        {
            "agent": "SafetyReviewAgent",
            "stance": safety_status,
            "comment": f"显著风险暴露 {metrics['risk_exposure_count']:.0f} 架，预算 {blackboard['risk_budget']} 架。",
        },
        {
            "agent": "PathPlanningAgent",
            "stance": "支持" if metrics["average_travel_distance"] <= blackboard["distance_reference"] else "质询",
            "comment": f"平均抽象移动距离 {metrics['average_travel_distance']:.1f} 格。",
        },
        {
            "agent": "TaskAllocationAgent",
            "stance": balance_status,
            "comment": f"任务区预计覆盖离散度 {metrics['coverage_imbalance']:.3f}。",
        },
    ]


def conduct_planning_meeting(
    blackboard: dict[str, Any],
    candidate_builder: Callable[[str], tuple[list[dict[str, Any]], list[str]]],
    candidate_evaluator: Callable[[list[dict[str, Any]]], dict[str, float]],
    *,
    llm_mode: str = "rule",
    llm_provider_name: str = "Agnes AI",
    llm_deliberator: Callable[..., dict[str, Any]] | None = None,
) -> PlanningDecision:
    opinions = build_expert_opinions(blackboard)
    candidates: list[CandidatePlan] = []
    for candidate_id, owner, profile, title in PROFILE_DEFINITIONS:
        assignments, warnings = candidate_builder(profile)
        metrics = candidate_evaluator(assignments)
        candidate = CandidatePlan(candidate_id, owner, profile, title, assignments, warnings, metrics)
        candidate.utility = score_candidate(metrics, blackboard)
        candidate.reviews = review_candidate(candidate, blackboard)
        candidates.append(candidate)

    llm_status = f"规则模式，未调用 {llm_provider_name}"
    llm_recommendation = ""
    llm_selection_reason = ""
    if llm_mode != "rule" and llm_deliberator is not None:
        advisor_candidates = [
            {
                "candidate_id": item.candidate_id,
                "owner_agent": item.owner_agent,
                "title": item.title,
                "metrics": item.metrics,
                "deterministic_utility": item.utility,
            }
            for item in candidates
        ]
        try:
            llm_result = llm_deliberator(blackboard=blackboard, candidates=advisor_candidates)
            llm_status = f"{llm_provider_name} 返回结构化会商结果"
            if len(llm_result.get("opinions", [])) >= 3:
                opinions = [ExpertOpinion(**item) for item in llm_result["opinions"]]
            for review in llm_result.get("reviews", []):
                candidate = next(
                    (item for item in candidates if item.candidate_id == review["candidate_id"]),
                    None,
                )
                if candidate is not None:
                    candidate.reviews = (candidate.reviews or []) + [
                        {
                            "agent": review["agent"],
                            "stance": f"模型-{review['stance']}",
                            "comment": review["comment"],
                        }
                    ]
            llm_recommendation = str(llm_result.get("recommended_candidate_id", ""))
            llm_selection_reason = str(llm_result.get("selection_reason", ""))
        except Exception as exc:
            llm_status = f"{llm_provider_name} 调用失败，已回退规则模式：{type(exc).__name__}: {str(exc)[:240]}"

    first_choice = max(candidates, key=lambda item: item.utility)
    revision_required = (
        first_choice.metrics["risk_exposure_count"] > blackboard["risk_budget"]
        or first_choice.metrics["minimum_projected_coverage"] < blackboard["target_coverage"] * 0.72
    )
    if revision_required:
        assignments, warnings = candidate_builder("consensus")
        metrics = candidate_evaluator(assignments)
        revised = CandidatePlan("E", "CoordinatorAgent", "consensus", "质询后共识修正版", assignments, warnings, metrics)
        revised.utility = score_candidate(metrics, blackboard) + 4.0
        revised.reviews = review_candidate(revised, blackboard)
        candidates.append(revised)

    selected = max(candidates, key=lambda item: item.utility)
    if llm_mode == "decision" and llm_recommendation:
        recommended = next((item for item in candidates if item.candidate_id == llm_recommendation), None)
        eligible = (
            recommended is not None
            and recommended.metrics["minimum_projected_coverage"] >= 0.45
            and recommended.metrics["risk_exposure_count"] <= blackboard["risk_budget"] + 1
        )
        if eligible:
            selected = recommended
            llm_status += "；模型推荐通过硬约束审查并参与最终选案"
        else:
            llm_status += "；模型推荐未通过覆盖/风险硬约束，保留规则选案"
    selection_reason = (
        f"候选 {selected.candidate_id} 被选中；综合评分 {selected.utility:.2f}，"
        f"最低预计覆盖 {selected.metrics['minimum_projected_coverage']:.1%}，"
        f"风险暴露 {selected.metrics['risk_exposure_count']:.0f} 架，平均移动距离 {selected.metrics['average_travel_distance']:.1f} 格。"
    )
    if llm_mode == "decision" and selected.candidate_id == llm_recommendation and llm_selection_reason:
        selection_reason += f" 模型选择理由：{llm_selection_reason}"
    return PlanningDecision(
        blackboard=blackboard,
        opinions=opinions,
        candidates=candidates,
        selected_candidate_id=selected.candidate_id,
        selected_profile=selected.profile,
        selection_reason=selection_reason,
        revision_required=revision_required,
        llm_mode=llm_mode,
        llm_provider=llm_provider_name,
        llm_status=llm_status,
        llm_recommendation=llm_recommendation,
    )
