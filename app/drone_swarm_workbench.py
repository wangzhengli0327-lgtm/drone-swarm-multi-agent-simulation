from __future__ import annotations

import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from drone_sim import (
    AgentMessage,
    CompatibleAPIError,
    DEMO_PRESETS,
    PlanningDecision,
    LLMMeetingPanel,
    OpenAICompatibleClient,
    OpenAICompatibleConfig,
    conduct_planning_meeting,
    make_rng,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_PATH = REPORT_DIR / "drone_swarm_simulation_report.md"
HANDOFF_PATH = REPORT_DIR / "drone_swarm_handoff.md"
TRACE_PATH = REPORT_DIR / "drone_swarm_agent_trace.md"

GRID_SIZE = 60
APP_SCHEMA_VERSION = "drone_dynamic_v17_multi_provider_llm"
SAFETY_STATEMENT = (
    "本系统仅用于抽象无人系统集群协同管控教学仿真，不连接真实设备，"
    "不生成真实飞行路线或军事行动方案。"
)
FORBIDDEN_NOTE = (
    "禁止扩展为真实无人机控制、真实航线规划、武器使用、目标选择、"
    "打击路径、突防规避、真实部署或任何军事行动方案。"
)

TASK_TYPE = "区域巡逻"
CONSTRAINT_OPTIONS = ["节点故障", "能量不足", "多任务冲突", "信息不确定性升高", "高风险区域节点失效"]
SPEED_OPTIONS = {"快速": 0.08, "标准": 0.22, "慢速演示": 0.5}
COVERAGE_MODE_OPTIONS = {
    "累计覆盖率": "cumulative",
    "滚动覆盖率": "rolling",
}
RISK_STRATEGY_OPTIONS = {
    "安全优先": "safe",
    "均衡权衡": "balanced",
    "覆盖优先": "coverage",
}
RISK_POSTURE_LABELS = {
    "conservative": "保守",
    "balanced": "均衡",
    "assertive": "进取",
}
LLM_MODE_OPTIONS = {
    "纯规则模式": "rule",
    "大模型辅助评审": "advisory",
    "大模型决策实验": "decision",
}
API_PROVIDER_PRESETS = {
    "Agnes AI（默认）": {
        "base_url": "https://apihub.agnes-ai.com/v1",
        "model": "agnes-2.0-flash",
        "requires_api_key": True,
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "",
        "requires_api_key": True,
    },
    "本地 Ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen3:8b",
        "requires_api_key": False,
    },
    "自定义 OpenAI 兼容 API": {
        "base_url": "",
        "model": "",
        "requires_api_key": True,
    },
}
PLATEAU_WINDOW = 14
PLATEAU_EPSILON = 0.003
MEETING_COOLDOWN = 12
STATE_TRIGGER_STEPS = 4

AGENT_NAMES = {
    "CoordinatorAgent": "总协调智能体",
    "ScenarioAgent": "场景研判智能体",
    "SwarmStatusAgent": "集群状态智能体",
    "CoverageAssessmentAgent": "覆盖评估智能体",
    "TaskAllocationAgent": "任务分配智能体",
    "PathPlanningAgent": "路径规划智能体",
    "SafetyReviewAgent": "安全审查智能体",
}

STATUS_LABELS = {
    "available": "可用",
    "assigned": "任务中",
    "relay": "中继",
    "low_battery": "低电量",
    "failed": "失效",
    "standby": "待命",
}

STATUS_COLORS = {
    "available": "#188038",
    "assigned": "#2563eb",
    "relay": "#7c3aed",
    "low_battery": "#d99b00",
    "failed": "#c7352b",
    "standby": "#667085",
}


def clamp(value: float, low: float = 0, high: float = GRID_SIZE - 1) -> float:
    return max(low, min(high, value))


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def in_circle(point: tuple[float, float], center: tuple[float, float], radius: float) -> bool:
    return distance(point, center) <= radius


def zone_label(zone: dict[str, Any]) -> str:
    return f"{zone['name']}({zone['center'][0]}, {zone['center'][1]}, r={zone['radius']})"


def build_circle_zone(name: str, center_row: int, center_col: int, radius: int, zone_type: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "name": name,
        "center": (center_row, center_col),
        "radius": radius,
        "type": zone_type,
    }
    if extra:
        payload.update(extra)
    return payload


def safe_point_near(center: tuple[float, float], restricted_zones: list[dict[str, Any]], offset: int = 0) -> tuple[float, float]:
    candidates: list[tuple[float, float]] = []
    for radius in range(0, 10):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) != radius:
                    continue
                point = (clamp(center[0] + dr + offset), clamp(center[1] + dc - offset))
                if all(not in_circle(point, zone["center"], zone["radius"] + 1.0) for zone in restricted_zones):
                    candidates.append(point)
        if candidates:
            return min(candidates, key=lambda item: distance(item, center))
    return (clamp(center[0]), clamp(center[1]))


def line_near_zone(start: tuple[float, float], end: tuple[float, float], zone: dict[str, Any]) -> bool:
    samples = np.linspace(0, 1, 28)
    for t in samples:
        point = (start[0] * (1 - t) + end[0] * t, start[1] * (1 - t) + end[1] * t)
        if in_circle(point, zone["center"], zone["radius"] + 1.2):
            return True
    return False


def detour_point(start: tuple[float, float], end: tuple[float, float], zone: dict[str, Any]) -> tuple[float, float]:
    center = zone["center"]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    normal = (-dy / length, dx / length)
    side_a = (clamp(center[0] + normal[0] * (zone["radius"] + 5)), clamp(center[1] + normal[1] * (zone["radius"] + 5)))
    side_b = (clamp(center[0] - normal[0] * (zone["radius"] + 5)), clamp(center[1] - normal[1] * (zone["radius"] + 5)))
    return side_a if distance(start, side_a) + distance(side_a, end) < distance(start, side_b) + distance(side_b, end) else side_b


def path_length(path: list[tuple[float, float]]) -> float:
    return sum(distance(left, right) for left, right in zip(path[:-1], path[1:]))


def smooth_segment(points: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    if count <= 1:
        return [points[-1]]
    if len(points) == 2:
        start, end = points
        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy) or 1
        control = (clamp(mid[0] - dy / length * 2.2), clamp(mid[1] + dx / length * 2.2))
        points = [start, control, end]

    dense: list[tuple[float, float]] = []
    anchors = points
    per_segment = max(3, math.ceil(count / (len(anchors) - 1)))
    for left, right in zip(anchors[:-1], anchors[1:]):
        mid = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        length = math.hypot(dx, dy) or 1
        control = (clamp(mid[0] - dy / length * 1.8), clamp(mid[1] + dx / length * 1.8))
        for raw_t in np.linspace(0, 1, per_segment, endpoint=False):
            t = float(raw_t)
            row = (1 - t) ** 2 * left[0] + 2 * (1 - t) * t * control[0] + t**2 * right[0]
            col = (1 - t) ** 2 * left[1] + 2 * (1 - t) * t * control[1] + t**2 * right[1]
            dense.append((clamp(row), clamp(col)))
    dense.append(anchors[-1])
    if len(dense) >= count:
        index = np.linspace(0, len(dense) - 1, count).round().astype(int)
        return [dense[int(i)] for i in index]
    return dense + [dense[-1]] * (count - len(dense))


def plan_smooth_path(
    start: tuple[float, float],
    end: tuple[float, float],
    avoid_zones: list[dict[str, Any]],
    steps: int,
    risk_posture: str = "balanced",
) -> list[tuple[float, float]]:
    anchors = [start]
    for zone in avoid_zones:
        if risk_posture != "assertive" and line_near_zone(start, end, zone):
            anchors.append(detour_point(start, end, zone))
    anchors.append(end)
    return smooth_segment(anchors, max(2, steps))


def estimate_arrival_steps(start: tuple[float, float], end: tuple[float, float], remaining: int) -> int:
    planned = int(math.ceil(distance(start, end) / 2.4)) + 4
    return max(2, min(max(2, remaining), min(32, planned)))


def path_hits_restricted(path: list[tuple[float, float]], restricted_zones: list[dict[str, Any]]) -> bool:
    for point in path:
        if any(in_circle(point, zone["center"], zone["radius"] + 0.5) for zone in restricted_zones):
            return True
    return False


def path_risk_score(path: list[tuple[float, float]], risk_zones: list[dict[str, Any]]) -> float:
    score = 0.0
    for point in path:
        for zone in risk_zones:
            if in_circle(point, zone["center"], zone["radius"]):
                score += 1.0
            elif in_circle(point, zone["center"], zone["radius"] + 3):
                score += 0.35
    return score


def point_risk_score(point: tuple[float, float], risk_zones: list[dict[str, Any]]) -> float:
    score = 0.0
    for zone in risk_zones:
        if in_circle(point, zone["center"], zone["radius"]):
            score += 10.0
        elif in_circle(point, zone["center"], zone["radius"] + 2):
            score += 1.5
    return score


def find_compliant_target(
    start: tuple[float, float],
    task: dict[str, Any],
    slot: int,
    risk_zones: list[dict[str, Any]],
    time_step: int,
    remaining: int,
    action_range: float,
    existing_service_points: list[tuple[tuple[float, float], float]],
    risk_posture: str,
) -> tuple[tuple[float, float], str, float]:
    base_angle = slot * 2.05 + time_step * 0.42
    candidates: list[tuple[float, float]] = []
    center = task["center"]
    if slot < 8:
        candidates.append((float(center[0]), float(center[1])))
    service_radii = [
        task["radius"] * 0.35,
        task["radius"] * 0.68,
        task["radius"] + min(action_range * 0.72, 6.0),
    ]
    for offset in [0, 0.8, -0.8, 1.6, -1.6, 2.4, -2.4, 3.14]:
        angle = base_angle + offset
        for service_radius in service_radii:
            target = (
                clamp(center[0] + math.sin(angle) * service_radius),
                clamp(center[1] + math.cos(angle) * service_radius),
            )
            if in_circle(target, center, task["radius"] + action_range * 0.8):
                candidates.append(target)

    safe_candidates = [target for target in candidates if point_risk_score(target, risk_zones) == 0]
    candidate_pool = candidates if risk_posture in {"balanced", "assertive"} else (safe_candidates if safe_candidates else candidates)
    scored: list[tuple[float, float, float, float, float]] = []
    arrival_steps = estimate_arrival_steps(start, center, remaining)
    spread_pattern = slot % 4
    desired_service_distance = 0.0 if spread_pattern == 0 else task["radius"] * (0.45 + 0.08 * (spread_pattern % 3))
    risk_unavoidable = not safe_candidates
    risk_weight_by_posture = {"conservative": 4.2, "balanced": 0.8, "assertive": 0.22}
    point_weight_by_posture = {"conservative": 55.0, "balanced": 0.5, "assertive": 0.12}
    service_weight_by_posture = {"conservative": 4.0, "balanced": 14.0, "assertive": 16.0}
    risk_weight = 0.35 if risk_unavoidable else risk_weight_by_posture[risk_posture]
    point_risk_weight = 1.0 if risk_unavoidable else point_weight_by_posture[risk_posture]
    for target in candidate_pool:
        path = plan_smooth_path(start, target, risk_zones, arrival_steps, risk_posture)
        risk_score = path_risk_score(path, risk_zones)
        point_risk = point_risk_score(target, risk_zones)
        current_service_rate = service_points_coverage_rate(task, existing_service_points)
        projected_service_rate = service_points_coverage_rate(
            task,
            existing_service_points + [(target, action_range)],
        )
        service_gain = max(0.0, projected_service_rate - current_service_rate)
        service_distance = distance(target, center)
        spread_score = abs(service_distance - desired_service_distance)
        score = (
            risk_score * risk_weight
            + point_risk * point_risk_weight
            - projected_service_rate * service_weight_by_posture[risk_posture]
            - service_gain * 8.0
            + spread_score * 0.18
            + path_length(path) * 0.015
        )
        scored.append((score, target[0], target[1], risk_score, point_risk))

    scored.sort(key=lambda item: item[0])
    _, row, col, risk_score, point_risk = scored[0]
    if point_risk > 0:
        return (row, col), "任务区服务点均受高风险区影响，当前落点需人工复核", risk_score
    if risk_score > 0:
        action = "接受较短的风险穿越" if risk_posture == "assertive" else "优先采用风险避让"
        return (row, col), f"落点位于任务区服务范围，{action}，保留风险代价", risk_score
    return (row, col), "落点位于任务区或其服务范围内，路径复核通过", risk_score


def make_uavs(uav_count: int, launch: tuple[int, int], action_range: int, low_count: int, failed_count: int) -> list[dict[str, Any]]:
    uavs: list[dict[str, Any]] = []
    for index in range(uav_count):
        status = "available"
        battery = max(52, 96 - index * 3)
        if index < failed_count:
            status = "failed"
            battery = 0
        elif index < failed_count + low_count:
            status = "low_battery"
            battery = 22
        uavs.append(
            {
                "id": f"UAV-{index + 1}",
                "status": status,
                "battery": battery,
                "range": action_range,
                "launch": launch,
                "position": tuple(float(v) for v in launch),
                "role": "待分配",
                "target": tuple(float(v) for v in launch),
                "path": [tuple(float(v) for v in launch)],
                "arrival_step": 0,
                "path_risk_score": 0.0,
                "replans": 0,
                "failure_reason": "",
            }
        )
    return uavs


def compute_dynamic_risk_posture(
    *,
    risk_strategy: str,
    coverage_mode: str,
    uavs: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    events: list[dict[str, Any]],
    coverage_rate: float,
    target_coverage: float,
    coverage_history: list[float],
    time_step: int,
    rolling_window: int,
) -> tuple[str, str, int]:
    score = {"safe": -1, "balanced": 0, "coverage": 1}[risk_strategy]
    reasons = [f"用户初始策略为 {next(label for label, value in RISK_STRATEGY_OPTIONS.items() if value == risk_strategy)}"]
    failed_count = sum(uav["status"] == "failed" for uav in uavs)
    active_count = sum(
        uav["status"] in {"available", "assigned", "relay"} and uav["battery"] > 18
        for uav in uavs
    )
    failed_ratio = failed_count / max(1, len(uavs))
    recent_risk_failures = sum(
        event["type"] == "高风险区域节点失效"
        and int(event["time_step"]) >= max(0, time_step - rolling_window)
        for event in events
    )
    severe_attrition = failed_ratio >= 0.35 or recent_risk_failures >= 2
    if severe_attrition:
        score -= 2
        reasons.append(f"失效比例 {failed_ratio:.0%}，近窗口高风险失效 {recent_risk_failures} 架，收紧风险暴露")
    elif failed_ratio >= 0.15 or recent_risk_failures >= 1:
        score -= 1
        reasons.append(f"已出现损失（失效比例 {failed_ratio:.0%}），降低连续投入强度")

    if active_count <= len(tasks):
        score -= 1
        reasons.append(f"可用 UAV 仅 {active_count} 架，需要优先保住各任务区基本覆盖")

    coverage_gap = max(0.0, target_coverage - coverage_rate)
    if time_step > 0 and coverage_gap >= 0.18 and not severe_attrition:
        score += 1
        reasons.append(f"主指标仍有 {coverage_gap:.0%} 缺口，允许适度提高覆盖投入")

    lookback = max(6, min(rolling_window, len(coverage_history) - 1))
    recent_gain = 0.0
    if lookback > 0 and len(coverage_history) > lookback:
        recent_gain = coverage_history[-1] - coverage_history[-lookback - 1]
    no_risk_losses = not any(event["type"] == "高风险区域节点失效" for event in events)
    if coverage_mode == "cumulative" and time_step >= max(20, rolling_window * 2) and no_risk_losses and recent_gain < 0.015:
        score += 1
        reasons.append(f"累计覆盖长时间无损失且近 {lookback} 步仅提升 {recent_gain:.1%}，可试探性扩大覆盖")
    elif coverage_mode == "rolling" and (recent_risk_failures > 0 or failed_ratio >= 0.15):
        score = min(score, 0)
        reasons.append("滚动覆盖需要持续留存可用节点，累计损失达到警戒线后禁止恢复为进取姿态")

    if score <= -1:
        posture = "conservative"
    elif score == 0:
        posture = "balanced"
    else:
        posture = "assertive"
    risk_budget = {
        "conservative": 0,
        "balanced": max(1, active_count // 5),
        "assertive": max(1, active_count // 3),
    }[posture]
    reasons.append(f"本轮风险暴露预算为 {risk_budget} 架")
    return posture, "；".join(reasons), risk_budget


def task_target(task: dict[str, Any], slot: int, restricted_zones: list[dict[str, Any]], time_step: int = 0) -> tuple[float, float]:
    center = task["center"]
    angle = slot * 2.05 + time_step * 0.42
    target = (
        center[0] + math.sin(angle) * task["radius"] * 0.72,
        center[1] + math.cos(angle) * task["radius"] * 0.72,
    )
    return safe_point_near((clamp(target[0]), clamp(target[1])), restricted_zones, slot % 3)


def allocate_tasks(
    uavs: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    zones: dict[str, list[dict[str, Any]]],
    constraints: list[str],
    time_step: int,
    remaining_steps: int,
    coverage_heatmap: np.ndarray,
    risk_posture: str,
    risk_budget: int,
    planning_profile: str = "balanced",
    search_variants: int = 2,
) -> tuple[list[dict[str, Any]], list[str]]:
    available = sorted(
        [uav for uav in uavs if uav["status"] in {"available", "assigned", "relay"} and uav["battery"] > 24],
        key=lambda item: item["battery"],
        reverse=True,
    )
    warnings: list[str] = []
    assignments: list[dict[str, Any]] = []

    if not available:
        return [], ["没有可参与主任务的虚拟 UAV，任务进入待人工复核状态。"]

    profile_settings = {
        "safety": {
            "posture": "conservative",
            "risk_budget": 0,
            "risk_weight": 2.2,
            "coverage_weight": 8.0,
            "projected_weight": 2.0,
            "gain_weight": 8.0,
            "travel_weight": 0.012,
        },
        "coverage": {
            "posture": "assertive",
            "risk_budget": max(risk_budget, max(1, len(available) // 3)),
            "risk_weight": 0.16,
            "coverage_weight": 11.0,
            "projected_weight": 4.0,
            "gain_weight": 17.0,
            "travel_weight": 0.008,
        },
        "time": {
            "posture": "assertive" if risk_posture == "assertive" else "balanced",
            "risk_budget": risk_budget,
            "risk_weight": 0.5,
            "coverage_weight": 8.5,
            "projected_weight": 2.5,
            "gain_weight": 9.0,
            "travel_weight": 0.18,
        },
        "balanced": {
            "posture": risk_posture,
            "risk_budget": risk_budget,
            "risk_weight": {"conservative": 1.6, "balanced": 0.65, "assertive": 0.18}[risk_posture],
            "coverage_weight": 10.0,
            "projected_weight": 3.0,
            "gain_weight": 10.0,
            "travel_weight": 0.01,
        },
        "consensus": {
            "posture": risk_posture,
            "risk_budget": risk_budget,
            "risk_weight": {"conservative": 1.9, "balanced": 0.8, "assertive": 0.24}[risk_posture],
            "coverage_weight": 11.0,
            "projected_weight": 3.5,
            "gain_weight": 14.0,
            "travel_weight": 0.035,
        },
    }
    settings = profile_settings[planning_profile]
    effective_posture = str(settings["posture"])
    effective_risk_budget = int(settings["risk_budget"])

    task_pressure = len(tasks) > len(available)
    if task_pressure or "多任务冲突" in constraints:
        warnings.append("任务数量或覆盖压力高于可用 UAV 冗余，需要分阶段搜索不同区域。")

    coverage_by_task = {item["任务区"]: item["覆盖率"] for item in task_coverage_details(coverage_heatmap, tasks)}
    assigned_by_task = {task["name"]: 0 for task in tasks}
    planned_service_points: dict[str, list[tuple[tuple[float, float], float]]] = {task["name"]: [] for task in tasks}
    risky_assignments = 0
    allocation_risk_weight = float(settings["risk_weight"])

    for index, uav in enumerate(available):
        unserved_tasks = [task for task in tasks if assigned_by_task[task["name"]] == 0]
        if unserved_tasks and len(available) >= len(tasks):
            candidate_tasks = unserved_tasks
        else:
            candidate_tasks = tasks
        ranked: list[tuple[float, dict[str, Any], tuple[float, float], str, float, float, bool]] = []
        for task_index, task in enumerate(candidate_tasks):
            base_slot = assigned_by_task[task["name"]] * 8
            current_service_rate = service_points_coverage_rate(task, planned_service_points[task["name"]])
            for variant in range(max(1, search_variants)):
                target, path_review, risk_score = find_compliant_target(
                    tuple(uav["position"]),
                    task,
                    base_slot + variant,
                    zones["risk"],
                    time_step,
                    max(2, remaining_steps),
                    float(uav["range"]),
                    planned_service_points[task["name"]],
                    effective_posture,
                )
                projected_service_rate = service_points_coverage_rate(
                    task,
                    planned_service_points[task["name"]] + [(target, float(uav["range"]))],
                )
                service_gain = max(0.0, projected_service_rate - current_service_rate)
                coverage_pressure = (
                    coverage_by_task.get(task["name"], 0.0) * 0.35
                    + current_service_rate
                    + assigned_by_task[task["name"]] * 0.03
                )
                travel_cost = distance(tuple(uav["position"]), target) * float(settings["travel_weight"])
                risky = risk_score > 3.0 or point_risk_score(target, zones["risk"]) > 0
                budget_penalty = 0.0
                if risky and risky_assignments >= effective_risk_budget and assigned_by_task[task["name"]] > 0:
                    budget_penalty = 35.0
                score = (
                    coverage_pressure * float(settings["coverage_weight"])
                    - projected_service_rate * float(settings["projected_weight"])
                    - service_gain * float(settings["gain_weight"])
                    + risk_score * allocation_risk_weight
                    + budget_penalty
                    + travel_cost
                )
                ranked.append((score, task, target, path_review, risk_score, projected_service_rate, risky))
        ranked.sort(key=lambda item: item[0])
        _, task, target, path_review, risk_score, projected_service_rate, risky = ranked[0]
        if risky:
            risky_assignments += 1
        assigned_by_task[task["name"]] += 1
        planned_service_points[task["name"]].append((target, float(uav["range"])))
        role = "区域巡逻"
        status = "assigned"
        uav["role"] = role
        uav["status"] = status
        uav["target"] = target
        assignments.append(
            {
                "time_step": time_step,
                "uav": uav["id"],
                "task_region": task["name"],
                "task_type": TASK_TYPE,
                "role": role,
                "target": target,
                "risk_score": round(risk_score, 2),
                "projected_service_coverage": round(projected_service_rate, 3),
                "risk_posture": RISK_POSTURE_LABELS[risk_posture],
                "planning_profile": planning_profile,
                "path_review": path_review,
                "reason": f"电量 {uav['battery']:.0f}% 且未失效，分配至 {task['name']} 做持续覆盖巡逻；预计服务覆盖 {projected_service_rate:.1%}；{path_review}。",
            }
        )

        if risk_score > 0:
            warnings.append(f"{uav['id']} 到 {task['name']} 的抽象轨迹存在高风险区代价；会议保留该风险并优先保证任务区覆盖。")

    for uav in uavs:
        if uav["status"] == "low_battery":
            uav["role"] = "返航/待命"
            uav["target"] = tuple(uav["position"])
        elif uav["status"] == "failed":
            uav["role"] = "失效"
            uav["target"] = tuple(uav["position"])
    if risky_assignments > effective_risk_budget:
        warnings.append(
            f"本轮有 {risky_assignments} 架 UAV 承担风险暴露，高于候选方案预算 {effective_risk_budget} 架；"
            "超出部分仅用于补齐尚未服务的任务区，需在下一次事件会议重新评估。"
        )
    return assignments, warnings


def apply_selected_candidate(uavs: list[dict[str, Any]], assignments: list[dict[str, Any]]) -> None:
    assignment_by_uav = {item["uav"]: item for item in assignments}
    for uav in uavs:
        if uav["status"] == "low_battery":
            uav["role"] = "返航/待命"
            uav["target"] = tuple(uav["position"])
            continue
        if uav["status"] == "failed":
            uav["role"] = "失效"
            uav["target"] = tuple(uav["position"])
            continue
        assignment = assignment_by_uav.get(uav["id"])
        if assignment is None:
            uav["role"] = "待命"
            uav["status"] = "standby"
            uav["target"] = tuple(uav["position"])
            continue
        uav["role"] = assignment["role"]
        uav["status"] = "assigned"
        uav["target"] = tuple(assignment["target"])


def evaluate_candidate_plan(
    assignments: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    uavs: list[dict[str, Any]],
    zones: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    uav_by_id = {uav["id"]: uav for uav in uavs}
    service_points: dict[str, list[tuple[tuple[float, float], float]]] = {task["name"]: [] for task in tasks}
    travel_distances: list[float] = []
    total_risk_score = 0.0
    risk_exposure_count = 0
    for assignment in assignments:
        uav = uav_by_id[assignment["uav"]]
        target = tuple(assignment["target"])
        service_points[assignment["task_region"]].append((target, float(uav["range"])))
        travel_distances.append(distance(tuple(uav["position"]), target))
        risk_score = float(assignment["risk_score"])
        total_risk_score += risk_score
        if risk_score > 3.0 or point_risk_score(target, zones["risk"]) > 0:
            risk_exposure_count += 1

    projected_rates = [service_points_coverage_rate(task, service_points[task["name"]]) for task in tasks]
    return {
        "minimum_projected_coverage": min(projected_rates, default=0.0),
        "average_projected_coverage": float(np.mean(projected_rates)) if projected_rates else 0.0,
        "coverage_imbalance": float(np.std(projected_rates)) if projected_rates else 0.0,
        "risk_exposure_count": float(risk_exposure_count),
        "total_risk_score": total_risk_score,
        "average_travel_distance": float(np.mean(travel_distances)) if travel_distances else 0.0,
        "unserved_task_count": float(sum(rate <= 0 for rate in projected_rates)),
    }


def build_meeting(
    meeting_id: int,
    time_step: int,
    trigger: str,
    uavs: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    zones: dict[str, list[dict[str, Any]]],
    warnings: list[str],
    coverage_rate: float,
    target_coverage: float,
    improvement: float,
    risk_strategy_label: str,
    risk_posture: str,
    risk_reason: str,
    risk_budget: int,
    meeting_type: str,
    trigger_evidence: str,
    planning_decision: PlanningDecision,
) -> tuple[list[AgentMessage], dict[str, Any]]:
    active = [uav["id"] for uav in uavs if uav["status"] in {"available", "assigned", "relay"}]
    low = [uav["id"] for uav in uavs if uav["status"] == "low_battery"]
    failed = [uav["id"] for uav in uavs if uav["status"] == "failed"]
    assignment_text = "；".join(f"{item['uav']}->{item['task_region']}({item['role']})" for item in assignments[:8]) or "暂无可执行分配"
    warnings_text = "；".join(warnings) if warnings else "无明显硬性冲突，但仍需复核覆盖连续性。"
    coverage_gap = max(0.0, target_coverage - coverage_rate)
    posture_label = RISK_POSTURE_LABELS[risk_posture]
    phase_messages: list[tuple[str, str, str, str, str, str]] = [
        (
            "共享态势",
            "CoordinatorAgent",
            "主持",
            f"第 {meeting_id} 场{meeting_type}，t={time_step}，触发原因：{trigger}。我已把任务区、UAV 状态、覆盖热力、风险预算和剩余时间写入共享态势黑板。",
            trigger_evidence,
            "请各领域专家基于同一份黑板分别给出意见，不预设最终方案。",
        )
    ]
    phase_messages.append(
        (
            "模型运行",
            "CoordinatorAgent",
            "运行时记录",
            f"本轮智能体模式：{planning_decision.llm_mode}。{planning_decision.llm_status}",
            f"提供商：{planning_decision.llm_provider}；模型推荐候选：{planning_decision.llm_recommendation or '无'}。API Key 不写入会议记录。",
            "继续使用已通过结构校验的专家意见；调用失败时使用规则专家回退。",
        )
    )
    for opinion in planning_decision.opinions:
        phase_messages.append(
            (
                "专家研判",
                opinion.agent,
                opinion.focus,
                f"我的独立观察是：{opinion.observation} 我的建议是：{opinion.recommendation}",
                opinion.evidence,
                "将该意见转化为候选方案约束或评分项。",
            )
        )

    for candidate in planning_decision.candidates:
        metrics = candidate.metrics
        assignment_preview = "；".join(
            f"{item['uav']}→{item['task_region']}" for item in candidate.assignments[:6]
        ) or "无可执行分配"
        phase_messages.append(
            (
                "候选提案" if candidate.candidate_id != "E" else "质询后修正",
                candidate.owner_agent,
                f"候选 {candidate.candidate_id}",
                f"我提交{candidate.title}：{assignment_preview}。预计最低覆盖 {metrics['minimum_projected_coverage']:.1%}，"
                f"平均覆盖 {metrics['average_projected_coverage']:.1%}，风险暴露 {metrics['risk_exposure_count']:.0f} 架，"
                f"平均移动距离 {metrics['average_travel_distance']:.1f} 格，综合评分 {candidate.utility:.2f}。",
                "该方案由任务分配器在 UAV 副本上实际推演，不会提前修改真实执行状态。",
                "请覆盖、安全、路径和任务分配专家逐项质询。",
            )
        )
        for review in candidate.reviews or []:
            phase_messages.append(
                (
                    "交叉质询",
                    review["agent"],
                    f"{review['stance']}候选 {candidate.candidate_id}",
                    review["comment"],
                    f"候选 {candidate.candidate_id} 的结构化评分指标。",
                    "将本意见计入候选方案选择与是否生成共识修正版。",
                )
            )

    selected = next(
        item for item in planning_decision.candidates if item.candidate_id == planning_decision.selected_candidate_id
    )
    phase_messages.extend(
        [
            (
                "安全复审",
                "SafetyReviewAgent",
                "最终边界审查",
                f"最终候选 {selected.candidate_id} 的风险暴露为 {selected.metrics['risk_exposure_count']:.0f} 架，"
                f"本轮预算为 {risk_budget} 架。执行保留意见：{warnings_text}",
                f"动态风险姿态为{posture_label}；{risk_reason}",
                "仅允许抽象教学仿真执行，不得解释为真实航线或行动方案。",
            ),
            (
                "形成共识",
                "CoordinatorAgent",
                "选择并下达抽象方案",
                f"会议选择候选 {selected.candidate_id}（{selected.title}）。{planning_decision.selection_reason} "
                f"真实执行分配为：{assignment_text}",
                "候选方案由相同态势分别推演，并经过覆盖、安全、路径和均衡四类审查。",
                "现在才把所选 profile 写入真实 UAV，进入下一段仿真。",
            ),
        ]
    )
    messages = [
        AgentMessage(time_step, meeting_id, phase, agent, stance, speech, evidence, next_action)
        for phase, agent, stance, speech, evidence, next_action in phase_messages
    ]
    minute = {
        "会议编号": meeting_id,
        "时间步": time_step,
        "会议类型": meeting_type,
        "触发原因": trigger,
        "触发证据": trigger_evidence,
        "当前覆盖率": f"{coverage_rate:.1%}",
        "目标覆盖率": f"{target_coverage:.1%}",
        "本轮覆盖提升": f"{improvement:.1%}",
        "用户风险策略": risk_strategy_label,
        "动态风险姿态": posture_label,
        "风险姿态依据": risk_reason,
        "风险暴露预算": risk_budget,
        "候选方案数量": len(planning_decision.candidates),
        "入选候选": planning_decision.selected_candidate_id,
        "入选策略": planning_decision.selected_profile,
        "智能体模式": planning_decision.llm_mode,
        "模型提供商": planning_decision.llm_provider,
        "模型状态": planning_decision.llm_status,
        "模型推荐": planning_decision.llm_recommendation or "无",
        "核心争议": f"覆盖缺口 {coverage_gap:.1%}；风险预算 {risk_budget} 架；可用节点 {active or ['无']}；低电量 {low or ['无']}；失效 {failed or ['无']}。",
        "修正结论": planning_decision.selection_reason,
        "保留风险": warnings_text,
    }
    return messages, minute


def simulate(
    *,
    random_seed: int,
    uav_count: int,
    launch: tuple[int, int],
    action_range: int,
    sim_steps: int,
    target_coverage: float,
    tasks: list[dict[str, Any]],
    zones: dict[str, list[dict[str, Any]]],
    constraints: list[str],
    low_count: int,
    failed_count: int,
    risk_failure_probability: float,
    coverage_mode: str,
    rolling_window: int,
    risk_strategy: str,
    llm_mode: str = "rule",
    llm_deliberator: Any | None = None,
    llm_provider_name: str = "Agnes AI（默认）",
) -> dict[str, Any]:
    rng = make_rng(random_seed)
    uavs = make_uavs(uav_count, launch, action_range, low_count, failed_count)
    meetings: list[AgentMessage] = []
    meeting_minutes: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    assignments_log: list[dict[str, Any]] = []
    risk_posture_log: list[dict[str, Any]] = []
    planning_rounds: list[dict[str, Any]] = []
    histories: dict[str, list[tuple[float, float]]] = {uav["id"]: [tuple(float(v) for v in launch)] for uav in uavs}
    meeting_id = 0
    last_meeting_coverage = 0.0
    primary_coverage_history: list[float] = [0.0]
    target_reached_step: int | None = None
    stop_reason = "达到最大仿真时间步"
    current_risk_posture: str | None = None
    last_meeting_step = -MEETING_COOLDOWN
    state_counters = {
        "coverage_stagnation": 0,
        "task_imbalance": 0,
        "rolling_decline": 0,
        "metric_divergence": 0,
        "posture_change": 0,
    }

    def current_primary_heatmap(time_step: int | None = None) -> np.ndarray:
        if time_step is None:
            time_step = max(0, len(next(iter(histories.values()), [])) - 1)
        return build_mode_coverage_heatmap(
            histories,
            uavs,
            coverage_mode=coverage_mode,
            until_step=time_step,
            rolling_window=rolling_window,
        )

    def current_primary_coverage() -> float:
        heatmap = current_primary_heatmap()
        return task_coverage_rate(heatmap, tasks)

    def active_uav_count() -> int:
        return sum(1 for uav in uavs if uav["status"] in {"available", "assigned", "relay"} and uav["battery"] > 18)

    def active_uavs_arrived(time_step: int) -> bool:
        return all(
            uav["status"] not in {"available", "assigned", "relay"} or int(uav.get("arrival_step", 0)) <= time_step
            for uav in uavs
        )

    def replan(
        time_step: int,
        trigger: str,
        meeting_type: str = "事件会议",
        trigger_evidence: str = "UAV 状态或风险事件发生变化，需要重新审查当前方案。",
    ) -> None:
        nonlocal meeting_id, last_meeting_coverage, current_risk_posture, last_meeting_step
        meeting_id += 1
        current_heatmap = current_primary_heatmap(time_step)
        coverage_rate = task_coverage_rate(current_heatmap, tasks)
        improvement = coverage_rate - last_meeting_coverage
        last_meeting_coverage = coverage_rate
        risk_posture, risk_reason, risk_budget = compute_dynamic_risk_posture(
            risk_strategy=risk_strategy,
            coverage_mode=coverage_mode,
            uavs=uavs,
            tasks=tasks,
            events=events,
            coverage_rate=coverage_rate,
            target_coverage=target_coverage,
            coverage_history=primary_coverage_history,
            time_step=time_step,
            rolling_window=rolling_window,
        )
        risk_strategy_label = next(label for label, value in RISK_STRATEGY_OPTIONS.items() if value == risk_strategy)
        previous_posture = current_risk_posture
        current_risk_posture = risk_posture
        last_meeting_step = time_step
        risk_posture_log.append(
            {
                "会议编号": meeting_id,
                "时间步": time_step,
                "触发原因": trigger,
                "用户风险策略": risk_strategy_label,
                "动态风险姿态": RISK_POSTURE_LABELS[risk_posture],
                "上轮风险姿态": RISK_POSTURE_LABELS.get(previous_posture, "初始"),
                "风险暴露预算": risk_budget,
                "会议类型": meeting_type,
                "调整依据": risk_reason,
            }
        )

        remaining = max(2, sim_steps - time_step + 1)
        current_task_details = task_coverage_details(current_heatmap, tasks)
        active_nodes = [
            uav for uav in uavs
            if uav["status"] in {"available", "assigned", "relay"} and uav["battery"] > 24
        ]
        nearest_task_distances = [
            min((distance(tuple(uav["position"]), task["center"]) for task in tasks), default=0.0)
            for uav in active_nodes
        ]
        blackboard = {
            "meeting_id": meeting_id,
            "time_step": time_step,
            "trigger": trigger,
            "task_count": len(tasks),
            "risk_zone_count": len(zones["risk"]),
            "active_uav_count": len(active_nodes),
            "failed_uav_count": sum(uav["status"] == "failed" for uav in uavs),
            "low_battery_count": sum(uav["status"] == "low_battery" for uav in uavs),
            "minimum_task_coverage": min((item["覆盖率"] for item in current_task_details), default=0.0),
            "target_coverage": target_coverage,
            "coverage_mode": coverage_mode,
            "risk_posture_label": RISK_POSTURE_LABELS[risk_posture],
            "risk_budget": risk_budget,
            "distance_reference": float(np.mean(nearest_task_distances)) if nearest_task_distances else 0.0,
        }

        def candidate_builder(profile: str) -> tuple[list[dict[str, Any]], list[str]]:
            return allocate_tasks(
                deepcopy(uavs),
                tasks,
                zones,
                constraints,
                time_step,
                remaining,
                current_heatmap,
                risk_posture,
                risk_budget,
                profile,
                1,
            )

        def candidate_evaluator(candidate_assignments: list[dict[str, Any]]) -> dict[str, float]:
            return evaluate_candidate_plan(candidate_assignments, tasks, uavs, zones)

        planning_decision = conduct_planning_meeting(
            blackboard,
            candidate_builder,
            candidate_evaluator,
            llm_mode=llm_mode,
            llm_provider_name=llm_provider_name,
            llm_deliberator=llm_deliberator,
        )
        planning_round = planning_decision.to_dict()
        planning_round["meeting_id"] = meeting_id
        planning_round["time_step"] = time_step
        planning_rounds.append(planning_round)

        selected_candidate = next(
            candidate
            for candidate in planning_decision.candidates
            if candidate.candidate_id == planning_decision.selected_candidate_id
        )
        assignments = deepcopy(selected_candidate.assignments)
        warnings = list(selected_candidate.warnings)
        apply_selected_candidate(uavs, assignments)
        assignments_log.extend(assignments)
        meeting_messages, minute = build_meeting(
            meeting_id,
            time_step,
            trigger,
            uavs,
            assignments,
            tasks,
            zones,
            warnings,
            coverage_rate,
            target_coverage,
            improvement,
            risk_strategy_label,
            risk_posture,
            risk_reason,
            risk_budget,
            meeting_type,
            trigger_evidence,
            planning_decision,
        )
        meetings.extend(meeting_messages)
        meeting_minutes.append(minute)
        execution_posture = {
            "safety": "conservative",
            "coverage": "assertive",
            "time": "assertive" if risk_posture == "assertive" else "balanced",
            "balanced": risk_posture,
            "consensus": risk_posture,
        }[planning_decision.selected_profile]
        for uav in uavs:
            start = tuple(uav["position"])
            if uav["status"] in {"failed", "low_battery"}:
                target = start
            else:
                target = tuple(uav["target"])
                uav["replans"] += 1
                uav["risk_posture"] = risk_posture
            arrival_steps = estimate_arrival_steps(start, target, remaining)
            uav["path"] = plan_smooth_path(start, target, zones["risk"], arrival_steps + 1, execution_posture)
            uav["arrival_step"] = time_step + len(uav["path"]) - 1
            uav["path_risk_score"] = round(path_risk_score(uav["path"], zones["risk"]), 2)

    replan(
        0,
        "初始任务规划",
        "初始规划会议",
        "用户完成场景设置，需要形成首轮任务分配、路径与风险共识。",
    )

    actual_steps = 0
    for t in range(1, sim_steps + 1):
        actual_steps = t
        need_replan = False
        trigger_reasons: list[str] = []
        pending_meeting_type = "事件会议"
        trigger_evidence = "UAV 状态或风险事件发生变化，需要立即重新审查当前方案。"
        for uav in uavs:
            previous = tuple(uav["position"])
            if uav["status"] != "failed":
                path_index = min(t - max([item["time_step"] for item in assignments_log if item["uav"] == uav["id"]] or [0]), len(uav["path"]) - 1)
                uav["position"] = tuple(uav["path"][path_index])
                moved = distance(previous, tuple(uav["position"]))
                if uav["status"] != "low_battery":
                    uav["battery"] = max(0, uav["battery"] - 0.35 - moved * 0.08)
                    if uav["battery"] < 18:
                        uav["status"] = "low_battery"
                        uav["role"] = "返航/待命"
                        events.append({"time_step": t, "type": "低电量", "uav": uav["id"], "detail": "电量低于阈值，触发重规划。"})
                        need_replan = True
                        trigger_reasons.append(f"{uav['id']} 低电量")

                for zone in zones["risk"]:
                    if uav["status"] not in {"failed", "low_battery"} and in_circle(tuple(uav["position"]), zone["center"], zone["radius"]):
                        if rng.random() < risk_failure_probability:
                            uav["status"] = "failed"
                            uav["role"] = "失效"
                            uav["battery"] = 0
                            uav["failure_reason"] = f"在 {zone['name']} 触发高风险区域节点失效事件"
                            events.append(
                                {
                                    "time_step": t,
                                    "type": "高风险区域节点失效",
                                    "uav": uav["id"],
                                    "detail": uav["failure_reason"],
                                }
                            )
                            need_replan = True
                            trigger_reasons.append(f"{uav['id']} 节点失效")
                            break
            histories[uav["id"]].append(tuple(uav["position"]))

        current_heatmap = current_primary_heatmap(t)
        coverage_now = task_coverage_rate(current_heatmap, tasks)
        primary_coverage_history.append(coverage_now)
        if target_reached_step is None and coverage_now >= target_coverage:
            target_reached_step = t

        state_review_allowed = (
            not need_replan
            and t - last_meeting_step >= MEETING_COOLDOWN
            and active_uavs_arrived(t)
        )
        if state_review_allowed:
            task_details = task_coverage_details(current_heatmap, tasks)
            task_rates = [item["覆盖率"] for item in task_details]
            min_task_rate = min(task_rates, default=0.0)
            max_task_rate = max(task_rates, default=0.0)
            review_window = min(PLATEAU_WINDOW, len(primary_coverage_history) - 1)
            recent_gain = (
                primary_coverage_history[-1] - primary_coverage_history[-review_window - 1]
                if review_window > 0
                else 0.0
            )
            recent_peak = max(primary_coverage_history[-max(2, rolling_window):])
            rolling_drop = recent_peak - coverage_now if coverage_mode == "rolling" else 0.0

            counterpart_mode = "rolling" if coverage_mode == "cumulative" else "cumulative"
            counterpart_heatmap = build_mode_coverage_heatmap(
                histories,
                uavs,
                coverage_mode=counterpart_mode,
                until_step=t,
                rolling_window=rolling_window,
            )
            counterpart_rates = [item["覆盖率"] for item in task_coverage_details(counterpart_heatmap, tasks)]
            if coverage_mode == "cumulative":
                cumulative_min = min_task_rate
                rolling_min = min(counterpart_rates, default=0.0)
            else:
                cumulative_min = min(counterpart_rates, default=0.0)
                rolling_min = min_task_rate

            candidate_posture, candidate_reason, _ = compute_dynamic_risk_posture(
                risk_strategy=risk_strategy,
                coverage_mode=coverage_mode,
                uavs=uavs,
                tasks=tasks,
                events=events,
                coverage_rate=coverage_now,
                target_coverage=target_coverage,
                coverage_history=primary_coverage_history,
                time_step=t,
                rolling_window=rolling_window,
            )
            active_count = active_uav_count()
            conditions = {
                "coverage_stagnation": review_window >= PLATEAU_WINDOW and recent_gain < 0.012 and coverage_now < 0.995,
                "task_imbalance": max_task_rate - min_task_rate >= 0.18 and min_task_rate < target_coverage - 0.08,
                "rolling_decline": rolling_drop >= 0.10,
                "metric_divergence": cumulative_min >= target_coverage and rolling_min < target_coverage - 0.12,
                "posture_change": current_risk_posture is not None and candidate_posture != current_risk_posture,
                "resource_pressure": active_count <= len(tasks) and min_task_rate < target_coverage,
            }
            for key, condition in conditions.items():
                state_counters[key] = state_counters.get(key, 0) + 1 if condition else 0

            trigger_candidates = [
                (
                    "posture_change",
                    "动态风险姿态需要调整",
                    f"风险姿态拟由 {RISK_POSTURE_LABELS.get(current_risk_posture, '初始')} 调整为 {RISK_POSTURE_LABELS[candidate_posture]}；{candidate_reason}",
                    "策略评审会议",
                ),
                (
                    "metric_divergence",
                    "累计覆盖与滚动覆盖明显背离",
                    f"累计最低覆盖率 {cumulative_min:.1%}，滚动最低覆盖率 {rolling_min:.1%}，说明历史覆盖较高但当前持续覆盖不足。",
                    "策略评审会议",
                ),
                (
                    "resource_pressure",
                    "可用资源无法稳定支撑全部任务区",
                    f"可用 UAV {active_count} 架，任务区 {len(tasks)} 个，当前最低任务区覆盖率 {min_task_rate:.1%}。",
                    "状态修正会议",
                ),
                (
                    "task_imbalance",
                    "任务区覆盖分布持续失衡",
                    f"任务区最高覆盖率 {max_task_rate:.1%}、最低覆盖率 {min_task_rate:.1%}，差值持续达到 {max_task_rate - min_task_rate:.1%}。",
                    "状态修正会议",
                ),
                (
                    "rolling_decline",
                    "滚动覆盖率持续下降",
                    f"滚动主指标较近窗口峰值下降 {rolling_drop:.1%}，需要重新检查持续覆盖能力。",
                    "状态修正会议",
                ),
                (
                    "coverage_stagnation",
                    "主覆盖指标提升停滞",
                    f"最近 {review_window} 步主指标仅提升 {recent_gain:.1%}，当前为 {coverage_now:.1%}。",
                    "状态修正会议",
                ),
            ]
            for key, reason, evidence, meeting_type in trigger_candidates:
                if state_counters.get(key, 0) >= STATE_TRIGGER_STEPS:
                    need_replan = True
                    trigger_reasons.append(reason)
                    trigger_evidence = evidence
                    pending_meeting_type = meeting_type
                    break

        if need_replan and t < sim_steps:
            replan(
                t,
                "、".join(sorted(set(trigger_reasons))),
                pending_meeting_type,
                trigger_evidence,
            )
            for key in state_counters:
                state_counters[key] = 0

    stop_reason = f"达到最大仿真时间步 {sim_steps}，已按用户设置完整推进"

    cumulative_coverage = build_coverage_heatmap(histories, uavs)
    rolling_coverage = build_mode_coverage_heatmap(
        histories,
        uavs,
        coverage_mode="rolling",
        until_step=actual_steps,
        rolling_window=rolling_window,
    )
    primary_coverage = rolling_coverage if coverage_mode == "rolling" else cumulative_coverage
    cumulative_by_step = [build_coverage_heatmap(histories, uavs, until_step=step) for step in range(actual_steps + 1)]
    rolling_by_step = [
        build_mode_coverage_heatmap(histories, uavs, coverage_mode="rolling", until_step=step, rolling_window=rolling_window)
        for step in range(actual_steps + 1)
    ]
    metrics_timeline = build_metrics_timeline(cumulative_by_step, rolling_by_step, tasks, uavs, events, actual_steps, coverage_mode)
    final_task_coverage = task_coverage_details(primary_coverage, tasks)
    cumulative_task_coverage = task_coverage_details(cumulative_coverage, tasks)
    rolling_task_coverage = task_coverage_details(rolling_coverage, tasks)
    decision_changes = build_decision_changes(assignments_log, meetings)
    risk_heatmap = build_risk_heatmap(zones)
    return {
        "schema_version": APP_SCHEMA_VERSION,
        "random_seed": random_seed,
        "uavs": uavs,
        "tasks": tasks,
        "zones": zones,
        "constraints": constraints,
        "histories": histories,
        "meetings": meetings,
        "events": events,
        "assignments": assignments_log,
        "meeting_minutes": meeting_minutes,
        "planning_rounds": planning_rounds,
        "risk_postures": risk_posture_log,
        "coverage": primary_coverage,
        "cumulative_coverage": cumulative_coverage,
        "rolling_coverage": rolling_coverage,
        "coverage_by_step": cumulative_by_step,
        "cumulative_by_step": cumulative_by_step,
        "rolling_by_step": rolling_by_step,
        "task_coverage": final_task_coverage,
        "cumulative_task_coverage": cumulative_task_coverage,
        "rolling_task_coverage": rolling_task_coverage,
        "metrics_timeline": metrics_timeline,
        "decision_changes": decision_changes,
        "risk_heatmap": risk_heatmap,
        "sim_steps": actual_steps,
        "max_steps": sim_steps,
        "target_coverage": target_coverage,
        "target_reached_step": target_reached_step,
        "coverage_mode": coverage_mode,
        "rolling_window": rolling_window,
        "risk_strategy": risk_strategy,
        "llm_mode": llm_mode,
        "llm_provider": llm_provider_name,
        "stop_reason": stop_reason,
        "launch": launch,
        "action_range": action_range,
        "risk_failure_probability": risk_failure_probability,
    }


def build_coverage_heatmap(
    histories: dict[str, list[tuple[float, float]]],
    uavs: list[dict[str, Any]],
    until_step: int | None = None,
    start_step: int = 0,
    current_only: bool = False,
) -> np.ndarray:
    heat = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    ranges = {uav["id"]: int(uav["range"]) for uav in uavs}
    for uav_id, points in histories.items():
        radius = ranges.get(uav_id, 4)
        selected_points = points
        if until_step is not None:
            end_index = min(until_step + 1, len(points))
            selected_points = points[max(0, start_step):end_index]
        if current_only and selected_points:
            selected_points = [selected_points[-1]]
        for row, col in selected_points:
            r0, r1 = max(0, int(row - radius)), min(GRID_SIZE - 1, int(row + radius))
            c0, c1 = max(0, int(col - radius)), min(GRID_SIZE - 1, int(col + radius))
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    d = distance((row, col), (rr, cc))
                    if d <= radius:
                        heat[rr, cc] += max(0.15, 1 - d / (radius + 0.1))
    return heat


def build_mode_coverage_heatmap(
    histories: dict[str, list[tuple[float, float]]],
    uavs: list[dict[str, Any]],
    *,
    coverage_mode: str,
    until_step: int | None = None,
    rolling_window: int = 12,
) -> np.ndarray:
    if coverage_mode == "rolling" and until_step is not None:
        start_step = max(0, until_step - rolling_window + 1)
        return build_coverage_heatmap(histories, uavs, until_step=until_step, start_step=start_step)
    return build_coverage_heatmap(histories, uavs, until_step=until_step)


def task_coverage_details(heatmap: np.ndarray, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for task in tasks:
        task_cells = 0
        covered_cells = 0
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if in_circle((row, col), task["center"], task["radius"]):
                    task_cells += 1
                    if heatmap[row, col] > 0:
                        covered_cells += 1
        rate = covered_cells / task_cells if task_cells else 0.0
        details.append(
            {
                "任务区": task["name"],
                "中心": f"({task['center'][0]}, {task['center'][1]})",
                "半径": task["radius"],
                "覆盖率": round(rate, 3),
                "覆盖网格": covered_cells,
                "总网格": task_cells,
            }
        )
    return details


def task_coverage_rate(heatmap: np.ndarray, tasks: list[dict[str, Any]]) -> float:
    details = task_coverage_details(heatmap, tasks)
    if not details:
        return 0.0
    return min(item["覆盖率"] for item in details)


def service_points_coverage_rate(task: dict[str, Any], service_points: list[tuple[tuple[float, float], float]]) -> float:
    if not service_points:
        return 0.0
    task_cells = 0
    covered_cells = 0
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            point = (row, col)
            if in_circle(point, task["center"], task["radius"]):
                task_cells += 1
                if any(distance(point, target) <= radius for target, radius in service_points):
                    covered_cells += 1
    return covered_cells / task_cells if task_cells else 0.0


def build_metrics_timeline(
    cumulative_by_step: list[np.ndarray],
    rolling_by_step: list[np.ndarray],
    tasks: list[dict[str, Any]],
    uavs: list[dict[str, Any]],
    events: list[dict[str, Any]],
    sim_steps: int,
    coverage_mode: str,
) -> pd.DataFrame:
    rows = []
    total_uavs = len(uavs)
    for step in range(sim_steps + 1):
        event_count = sum(1 for event in events if int(event["time_step"]) <= step)
        failed_count = sum(1 for event in events if event["type"] == "高风险区域节点失效" and int(event["time_step"]) <= step)
        low_count = sum(1 for event in events if event["type"] == "低电量" and int(event["time_step"]) <= step)
        available_estimate = max(0, total_uavs - failed_count - low_count)
        cumulative_details = task_coverage_details(cumulative_by_step[step], tasks)
        rolling_details = task_coverage_details(rolling_by_step[step], tasks)
        cumulative_min = min((item["覆盖率"] for item in cumulative_details), default=0.0)
        cumulative_avg = sum((item["覆盖率"] for item in cumulative_details), 0.0) / len(cumulative_details) if cumulative_details else 0.0
        rolling_min = min((item["覆盖率"] for item in rolling_details), default=0.0)
        rolling_avg = sum((item["覆盖率"] for item in rolling_details), 0.0) / len(rolling_details) if rolling_details else 0.0
        primary_min = rolling_min if coverage_mode == "rolling" else cumulative_min
        primary_avg = rolling_avg if coverage_mode == "rolling" else cumulative_avg
        rows.append(
            {
                "时间步": step,
                "主指标最低覆盖率": round(primary_min, 3),
                "主指标平均覆盖率": round(primary_avg, 3),
                "累计最低覆盖率": round(cumulative_min, 3),
                "累计平均覆盖率": round(cumulative_avg, 3),
                "滚动最低覆盖率": round(rolling_min, 3),
                "滚动平均覆盖率": round(rolling_avg, 3),
                "可用 UAV 估计": available_estimate,
                "累计风险事件": event_count,
            }
        )
    return pd.DataFrame(rows)


def build_decision_changes(assignments_log: list[dict[str, Any]], meetings: list[AgentMessage]) -> pd.DataFrame:
    if not assignments_log:
        return pd.DataFrame(columns=["会议时间步", "触发/阶段", "调整对象", "决策变化", "原因"])
    rows: list[dict[str, str]] = []
    previous: dict[str, str] = {}
    meeting_steps = sorted({message.time_step for message in meetings})
    for step in meeting_steps:
        current_items = [item for item in assignments_log if item["time_step"] == step]
        for item in current_items:
            before = previous.get(item["uav"], "未分配")
            after = f"{item['task_region']} / {item['role']}"
            if before != after:
                rows.append(
                    {
                        "会议时间步": step,
                        "触发/阶段": "提案-质询-修正-复审-共识",
                        "调整对象": item["uav"],
                        "决策变化": f"{before} -> {after}",
                        "原因": item["reason"],
                    }
                )
            previous[item["uav"]] = after
    return pd.DataFrame(rows) if rows else pd.DataFrame([{"会议时间步": "-", "触发/阶段": "无变化", "调整对象": "-", "决策变化": "任务分配保持稳定", "原因": "未触发改派"}])


def build_risk_heatmap(zones: dict[str, list[dict[str, Any]]]) -> np.ndarray:
    heat = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            point = (row, col)
            for zone in zones["risk"]:
                if in_circle(point, zone["center"], zone["radius"]):
                    heat[row, col] = max(heat[row, col], 0.7)
            for zone in zones["restricted"]:
                if in_circle(point, zone["center"], zone["radius"]):
                    heat[row, col] = max(heat[row, col], 1.0)
    return heat


def zone_shapes(zones: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    styles = {
        "task": ("rgba(34,197,94,0.10)", "#22c55e"),
        "restricted": ("rgba(220,38,38,0.18)", "#dc2626"),
        "risk": ("rgba(234,88,12,0.16)", "#ea580c"),
    }
    shapes: list[dict[str, Any]] = []
    for kind, zone_list in zones.items():
        if kind not in styles:
            continue
        fill, line = styles[kind]
        for zone in zone_list:
            row, col = zone["center"]
            radius = zone["radius"]
            shapes.append(
                {
                    "type": "circle",
                    "xref": "x",
                    "yref": "y",
                    "x0": col - radius,
                    "x1": col + radius,
                    "y0": row - radius,
                    "y1": row + radius,
                    "fillcolor": fill,
                    "line": {"color": line, "width": 2},
                }
            )
    return shapes


def add_zone_labels(fig: go.Figure, zones: dict[str, list[dict[str, Any]]]) -> None:
    for kind, zone_list in zones.items():
        if kind not in {"task", "risk", "restricted"}:
            continue
        for zone in zone_list:
            row, col = zone["center"]
            label = zone["name"]
            if kind == "task":
                label += " / 区域巡逻"
            fig.add_annotation(x=col, y=row, text=label, showarrow=False, font=dict(size=10, color="#111827"), bgcolor="rgba(255,255,255,0.72)")


def build_path_figure(result: dict[str, Any], title: str = "最终位置与抽象平滑路径图") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(shapes=zone_shapes({"task": result["tasks"], **result["zones"]}))
    add_zone_labels(fig, {"task": result["tasks"], **result["zones"]})

    for uav in result["uavs"]:
        history = result["histories"][uav["id"]]
        rows = [point[0] for point in history]
        cols = [point[1] for point in history]
        color = STATUS_COLORS.get(uav["status"], "#2563eb")
        fig.add_trace(
            go.Scatter(
                x=cols,
                y=rows,
                mode="lines",
                line=dict(color=color, width=3),
                name=f"{uav['id']} 路径",
                hovertemplate=f"{uav['id']}<br>抽象路径点<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[cols[-1]],
                y=[rows[-1]],
                mode="markers+text",
                marker=dict(size=14, color=color, line=dict(color="#ffffff", width=2)),
                text=[uav["id"]],
                textposition="top center",
                name=f"{uav['id']} 最终位置",
                hovertemplate=(
                    f"{uav['id']}<br>状态：{STATUS_LABELS.get(uav['status'], uav['status'])}"
                    f"<br>角色：{uav['role']}<br>电量：{uav['battery']:.1f}%<extra></extra>"
                ),
            )
        )

    launch = result["launch"]
    fig.add_trace(
        go.Scatter(
            x=[launch[1]],
            y=[launch[0]],
            mode="markers+text",
            marker=dict(size=18, color="#111827", symbol="star"),
            text=["起飞点"],
            textposition="bottom center",
            name="共同起飞点",
        )
    )
    fig.update_layout(
        title=title,
        height=760,
        margin=dict(l=25, r=25, t=55, b=25),
        xaxis=dict(title="抽象列", range=[-1, GRID_SIZE], dtick=5, showgrid=True),
        yaxis=dict(title="抽象行", range=[GRID_SIZE, -1], dtick=5, showgrid=True, scaleanchor="x", scaleratio=1),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def build_snapshot_figure(result: dict[str, Any], time_step: int) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(shapes=zone_shapes({"task": result["tasks"], **result["zones"]}))
    add_zone_labels(fig, {"task": result["tasks"], **result["zones"]})
    for uav in result["uavs"]:
        history = result["histories"][uav["id"]]
        point = history[min(time_step, len(history) - 1)]
        color = STATUS_COLORS.get(uav["status"], "#2563eb")
        fig.add_trace(
            go.Scatter(
                x=[point[1]],
                y=[point[0]],
                mode="markers+text",
                marker=dict(size=max(10, result["action_range"] * 2.1), color=color, opacity=0.78),
                text=[uav["id"]],
                textposition="middle center",
                name=uav["id"],
            )
        )
    fig.update_layout(
        title=f"时间轴回放：t={time_step}",
        height=700,
        margin=dict(l=25, r=25, t=55, b=25),
        xaxis=dict(range=[-1, GRID_SIZE], dtick=5),
        yaxis=dict(range=[GRID_SIZE, -1], dtick=5, scaleanchor="x", scaleratio=1),
        plot_bgcolor="#f8fafc",
    )
    return fig


def build_heatmap_figure(data: np.ndarray, title: str, colorscale: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=data, colorscale=colorscale, showscale=True, hovertemplate="列 %{x}, 行 %{y}<br>强度 %{z:.2f}<extra></extra>"))
    fig.update_layout(
        title=title,
        height=700,
        margin=dict(l=25, r=25, t=55, b=25),
        xaxis=dict(title="抽象列", dtick=5),
        yaxis=dict(title="抽象行", range=[GRID_SIZE, -1], dtick=5, scaleanchor="x", scaleratio=1),
        plot_bgcolor="#ffffff",
    )
    return fig


def build_metrics_figure(result: dict[str, Any]) -> go.Figure:
    df = result["metrics_timeline"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["时间步"], y=df["主指标最低覆盖率"], mode="lines+markers", name="主指标最低覆盖率", line=dict(width=4)))
    fig.add_trace(go.Scatter(x=df["时间步"], y=df["累计最低覆盖率"], mode="lines", name="累计最低覆盖率"))
    fig.add_trace(go.Scatter(x=df["时间步"], y=df["滚动最低覆盖率"], mode="lines", name="滚动最低覆盖率"))
    fig.add_trace(go.Scatter(x=df["时间步"], y=df["主指标平均覆盖率"], mode="lines", name="主指标平均覆盖率", line=dict(dash="dot")))
    fig.add_trace(
        go.Scatter(
            x=df["时间步"],
            y=df["可用 UAV 估计"] / max(1, df["可用 UAV 估计"].max()),
            mode="lines+markers",
            name="可用 UAV 比例",
        )
    )
    fig.update_layout(
        title="任务完成度时间线",
        height=420,
        yaxis=dict(title="归一化指标", range=[0, 1.05]),
        xaxis=dict(title="抽象时间步"),
        margin=dict(l=30, r=20, t=55, b=30),
        plot_bgcolor="#ffffff",
    )
    return fig


def uav_summary_df(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for uav in result["uavs"]:
        history = result["histories"][uav["id"]]
        rows.append(
            {
                "UAV": uav["id"],
                "状态": STATUS_LABELS.get(uav["status"], uav["status"]),
                "任务角色": uav["role"],
                "起飞点": f"({result['launch'][0]}, {result['launch'][1]})",
                "最终抽象位置": f"({history[-1][0]:.1f}, {history[-1][1]:.1f})",
                "剩余电量": f"{uav['battery']:.1f}%",
                "重规划次数": uav["replans"],
                "说明": uav["failure_reason"] or "按会商结果完成抽象路径规划",
            }
        )
    return pd.DataFrame(rows)


def event_df(result: dict[str, Any]) -> pd.DataFrame:
    if not result["events"]:
        return pd.DataFrame([{"时间步": "-", "事件": "无突发事件", "UAV": "-", "说明": "仿真期间未触发失效或低电量重规划。"}])
    return pd.DataFrame(
        [
            {"时间步": item["time_step"], "事件": item["type"], "UAV": item["uav"], "说明": item["detail"]}
            for item in result["events"]
        ]
    )


def meeting_df(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "会议编号": msg.meeting_id,
                "时间步": msg.time_step,
                "阶段": msg.phase,
                "Agent": msg.agent,
                "中文译名": AGENT_NAMES[msg.agent],
                "立场": msg.stance,
                "发言": msg.speech,
                "证据": msg.evidence,
                "下一步": msg.next_action,
            }
            for msg in result["meetings"]
        ]
    )


def meeting_minutes_df(result: dict[str, Any]) -> pd.DataFrame:
    if not result.get("meeting_minutes"):
        return pd.DataFrame(columns=["会议编号", "时间步", "会议类型", "触发原因", "触发证据", "当前覆盖率", "目标覆盖率", "本轮覆盖提升", "核心争议", "修正结论", "保留风险"])
    return pd.DataFrame(result["meeting_minutes"])


def decision_changes_df(result: dict[str, Any]) -> pd.DataFrame:
    return result["decision_changes"]


def task_coverage_df(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(result["task_coverage"])


def planning_candidates_df(result: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for planning_round in result.get("planning_rounds", []):
        selected_id = planning_round["selected_candidate_id"]
        for candidate in planning_round["candidates"]:
            metrics = candidate["metrics"]
            rows.append(
                {
                    "会议编号": planning_round["meeting_id"],
                    "时间步": planning_round["time_step"],
                    "智能体模式": planning_round.get("llm_mode", "rule"),
                    "模型提供商": planning_round.get("llm_provider", "未配置"),
                    "模型推荐": planning_round.get("llm_recommendation") or "无",
                    "模型状态": planning_round.get("llm_status", "未调用"),
                    "候选": candidate["candidate_id"],
                    "提出专家": AGENT_NAMES.get(candidate["owner_agent"], candidate["owner_agent"]),
                    "方案": candidate["title"],
                    "策略 profile": candidate["profile"],
                    "最低预计覆盖": f"{metrics['minimum_projected_coverage']:.1%}",
                    "平均预计覆盖": f"{metrics['average_projected_coverage']:.1%}",
                    "覆盖离散度": metrics["coverage_imbalance"],
                    "风险暴露 UAV": int(metrics["risk_exposure_count"]),
                    "总风险评分": metrics["total_risk_score"],
                    "平均移动距离": metrics["average_travel_distance"],
                    "综合评分": candidate["utility"],
                    "会议选择": "入选" if candidate["candidate_id"] == selected_id else "未入选",
                }
            )
    return pd.DataFrame(rows)


def planning_blackboard_df(result: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for planning_round in result.get("planning_rounds", []):
        board = planning_round["blackboard"]
        rows.append(
            {
                "会议编号": planning_round["meeting_id"],
                "时间步": planning_round["time_step"],
                "触发原因": board["trigger"],
                "可用 UAV": board["active_uav_count"],
                "失效 UAV": board["failed_uav_count"],
                "低电量 UAV": board["low_battery_count"],
                "任务区": board["task_count"],
                "高风险区": board["risk_zone_count"],
                "当前最低覆盖": f"{board['minimum_task_coverage']:.1%}",
                "目标覆盖": f"{board['target_coverage']:.1%}",
                "主指标": board["coverage_mode"],
                "风险姿态": board["risk_posture_label"],
                "风险预算": board["risk_budget"],
            }
        )
    return pd.DataFrame(rows)


def planning_reviews_df(result: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for planning_round in result.get("planning_rounds", []):
        for candidate in planning_round["candidates"]:
            for review in candidate.get("reviews") or []:
                rows.append(
                    {
                        "会议编号": planning_round["meeting_id"],
                        "候选": candidate["candidate_id"],
                        "审查专家": AGENT_NAMES.get(review["agent"], review["agent"]),
                        "立场": review["stance"],
                        "质询或评价": review["comment"],
                    }
                )
    return pd.DataFrame(rows)


def as_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "暂无数据"
    columns = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("\n", " ") for column in df.columns) + " |")
    return "\n".join(lines)


def write_markdown_outputs(result: dict[str, Any]) -> tuple[str, str, str]:
    write_errors: list[str] = []
    try:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        write_errors.append(str(exc))

    uav_df = uav_summary_df(result)
    events = event_df(result)
    meetings = meeting_df(result)
    minutes = meeting_minutes_df(result)
    metrics = result["metrics_timeline"]
    changes = decision_changes_df(result)
    task_coverage = task_coverage_df(result)
    planning_candidates = planning_candidates_df(result)
    planning_reviews = planning_reviews_df(result)
    risk_postures = pd.DataFrame(result.get("risk_postures", []))
    coverage_compare = pd.DataFrame(result["cumulative_task_coverage"]).merge(
        pd.DataFrame(result["rolling_task_coverage"]),
        on=["任务区", "中心", "半径", "总网格"],
        suffixes=("（累计）", "（滚动）"),
    )
    risk_probability = result["risk_failure_probability"] * 100
    mode_label = "滚动覆盖率" if result["coverage_mode"] == "rolling" else "累计覆盖率"
    risk_strategy_label = next(
        (label for label, value in RISK_STRATEGY_OPTIONS.items() if value == result.get("risk_strategy")),
        "均衡权衡",
    )
    target_reached = result.get("target_reached_step")
    target_reached_text = f"t={target_reached}" if target_reached is not None else "未达标"

    report = f"""# 无人机集群动态多智能体协同管控仿真报告

## 1. 仿真定位

{SAFETY_STATEMENT}

本报告展示 60x60 抽象网格中的动态任务会商、平滑路径、风险事件和重规划过程。所有坐标均为教学网格坐标，不是经纬度、航线或飞控指令。

## 2. 用户设置摘要

- 抽象空间：60x60
- 仿真随机种子：{result['random_seed']}
- 共同起飞点：({result['launch'][0]}, {result['launch'][1]})
- 最大仿真时长：{result['max_steps']} 个抽象时间步
- 实际仿真时长：{result['sim_steps']} 个抽象时间步
- 目标覆盖率：{result['target_coverage']:.0%}
- 主覆盖指标：{mode_label}
- 智能体会议模式：{result.get('llm_mode', 'rule')}
- 模型 API 提供商：{result.get('llm_provider', '未配置')}
- 用户风险策略：{risk_strategy_label}（仿真中会按实时状态调整动态风险姿态）
- 滚动覆盖窗口：{result['rolling_window']} 个时间步
- 达标时间：{target_reached_text}
- 结束原因：{result['stop_reason']}
- UAV 作用范围：{result['action_range']} 格
- 高风险区节点失效概率：每时间步 {risk_probability:.1f}%
- 限制条件：{', '.join(result['constraints']) if result['constraints'] else '无'}

## 3. 多智能体角色

- CoordinatorAgent（总协调智能体）：主持会商并形成共识。
- ScenarioAgent（场景研判智能体）：解释任务区和高风险区。
- SwarmStatusAgent（集群状态智能体）：检查电量、状态、失效节点和可参与节点。
- TaskAllocationAgent（任务分配智能体）：给出每架 UAV 的抽象任务角色。
- PathPlanningAgent（路径规划智能体）：生成抽象平滑路径、抵达时间和高风险代价评估。
- SafetyReviewAgent（安全审查智能体）：审查高风险区、失效节点、低电量和越界表达。

## 4. 每架 UAV 的最终建议

{as_markdown_table(uav_df)}

## 5. 事件时间线

{as_markdown_table(events)}

## 6. 主指标各任务区覆盖率

{as_markdown_table(task_coverage)}

## 7. 累计覆盖率与滚动覆盖率对照

{as_markdown_table(coverage_compare)}

## 8. AutoGen 式多智能体会议记录

{as_markdown_table(meetings)}

## 9. 候选方案评分与会议选择

{as_markdown_table(planning_candidates)}

## 10. 专家交叉质询

{as_markdown_table(planning_reviews)}

## 11. 每场会议纪要

{as_markdown_table(minutes)}

## 12. 任务完成度时间线

{as_markdown_table(metrics)}

## 13. 会议决策差异表

{as_markdown_table(changes)}

## 14. 动态风险姿态记录

{as_markdown_table(risk_postures)}

## 15. 人在回路与安全边界

{FORBIDDEN_NOTE}

本仿真中的“路径”仅为抽象平滑曲线，用于说明多智能体如何在约束变化后重新规划，不可解释为真实飞行路线、部署建议或军事行动方案。
"""

    handoff = f"""# 无人机集群动态协同演示摘要

## 场景

- 60x60 抽象网格
- 仿真随机种子：{result['random_seed']}
- 共同起飞点：({result['launch'][0]}, {result['launch'][1]})
- 最大仿真时长：{result['max_steps']} 个抽象时间步
- 实际仿真时长：{result['sim_steps']} 个抽象时间步
- 目标覆盖率：{result['target_coverage']:.0%}
- 主覆盖指标：{mode_label}
- 达标时间：{target_reached_text}
- 结束原因：{result['stop_reason']}
- 安全声明：{SAFETY_STATEMENT}

## 会商结论

系统通过 7 个智能体进行 AutoGen 式会议，在初始规划、突发事件、持续状态异常和策略姿态变化时重新规划。

## UAV 最终建议

{as_markdown_table(uav_df)}

## 主指标各任务区覆盖率

{as_markdown_table(task_coverage)}

## 累计/滚动覆盖率对照

{as_markdown_table(coverage_compare)}

## 事件

{as_markdown_table(events)}

## 会议决策差异

{as_markdown_table(changes)}

## 会议纪要

{as_markdown_table(minutes)}

## 禁止事项

{FORBIDDEN_NOTE}
"""

    trace = "# 无人机集群多智能体会议 Trace\n\n" + SAFETY_STATEMENT + "\n\n" + as_markdown_table(meetings)

    for path, content in [(REPORT_PATH, report), (HANDOFF_PATH, handoff), (TRACE_PATH, trace)]:
        try:
            path.write_text(content, encoding="utf-8")
        except PermissionError as exc:
            write_errors.append(f"{path}: {exc}")
    st.session_state["drone_swarm_report_write_errors"] = write_errors
    return report, handoff, trace


def apply_style() -> None:
    st.markdown(
        """
<style>
.block-container { padding-top: 1.2rem; }
.safety-box {
  border: 1px solid #e5e7eb;
  border-left: 6px solid #dc2626;
  background: #fff7f6;
  border-radius: 8px;
  padding: 0.85rem 1rem;
  margin-bottom: 1rem;
}
.flow-strip {
  display: flex;
  flex-wrap: wrap;
  gap: .45rem;
  align-items: center;
  border: 1px solid #dbe3ec;
  background: #f8fafc;
  border-radius: 8px;
  padding: .7rem .8rem;
  margin: .5rem 0 .9rem 0;
}
.flow-strip span {
  background: #ffffff;
  border: 1px solid #d0d7de;
  border-radius: 6px;
  padding: .28rem .5rem;
  font-size: .88rem;
}
.meeting-town {
  border: 1px solid #d7dfeb;
  border-radius: 8px;
  padding: .7rem;
  margin: .55rem 0 1rem 0;
  background:
    linear-gradient(135deg, rgba(239,246,255,.82), rgba(248,250,252,.96)),
    repeating-linear-gradient(90deg, rgba(148,163,184,.12) 0, rgba(148,163,184,.12) 1px, transparent 1px, transparent 36px);
}
.meeting-town-title {
  font-weight: 750;
  color: #0f172a;
  margin-bottom: .45rem;
}
.agent-card {
  display: grid;
  grid-template-columns: 3rem 1fr;
  gap: .75rem;
  border: 1px solid #dbe3ec;
  border-radius: 8px;
  padding: .78rem .9rem;
  background: rgba(255,255,255,.96);
  box-shadow: 0 8px 24px rgba(15, 23, 42, .06);
  margin-bottom: .65rem;
}
.agent-avatar {
  width: 2.55rem;
  height: 2.55rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #ffffff;
  font-weight: 800;
  border: 2px solid rgba(255,255,255,.9);
  box-shadow: 0 6px 14px rgba(15,23,42,.16);
}
.agent-headline {
  font-weight: 760;
  color: #111827;
  margin-bottom: .2rem;
}
.agent-badge {
  display: inline-block;
  border: 1px solid #d0d7de;
  background: #f8fafc;
  border-radius: 999px;
  padding: .08rem .48rem;
  margin-left: .35rem;
  color: #475467;
  font-size: .78rem;
}
.agent-speech {
  color: #1f2937;
  line-height: 1.55;
  margin: .2rem 0 .35rem 0;
}
.small-muted { color: #667085; font-size: .88rem; }
</style>
""",
        unsafe_allow_html=True,
    )


def zone_controls(
    prefix: str,
    label: str,
    count: int,
    default_centers: list[tuple[int, int]],
    default_radius: int,
    zone_type: str,
    max_radius: int = 12,
) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []
    for index in range(count):
        with st.sidebar.expander(f"{label} {index + 1}", expanded=index == 0):
            default_row, default_col = default_centers[index % len(default_centers)]
            row = st.slider(f"{label}{index + 1} 中心行", 0, GRID_SIZE - 1, default_row, key=f"{prefix}_r_{index}")
            col = st.slider(f"{label}{index + 1} 中心列", 0, GRID_SIZE - 1, default_col, key=f"{prefix}_c_{index}")
            radius = st.slider(f"{label}{index + 1} 半径", 2, max_radius, min(default_radius, max_radius), key=f"{prefix}_rad_{index}")
            zones.append(build_circle_zone(f"{label}{index + 1}", row, col, radius, zone_type))
    return zones


def task_controls(task_count: int) -> list[dict[str, Any]]:
    defaults = [(18, 42), (42, 42), (30, 22), (45, 18), (18, 18)]
    tasks: list[dict[str, Any]] = []
    for index in range(task_count):
        with st.sidebar.expander(f"任务区域 {index + 1}", expanded=index < 2):
            row = st.slider(f"任务{index + 1} 中心行", 0, GRID_SIZE - 1, defaults[index % len(defaults)][0], key=f"task_r_{index}")
            col = st.slider(f"任务{index + 1} 中心列", 0, GRID_SIZE - 1, defaults[index % len(defaults)][1], key=f"task_c_{index}")
            radius = st.slider(f"任务{index + 1} 半径", 3, 12, 6, key=f"task_rad_{index}")
            st.caption("任务类型：区域巡逻。目标是让 UAV 作用范围持续覆盖该任务区。")
            tasks.append(build_circle_zone(f"任务区 {chr(65 + index)}", row, col, radius, "task", {"task_type": TASK_TYPE}))
    return tasks


def render_meeting(result: dict[str, Any], delay: float) -> None:
    agent_initials = {
        "CoordinatorAgent": ("总", "#2563eb"),
        "ScenarioAgent": ("景", "#0891b2"),
        "SwarmStatusAgent": ("群", "#7c3aed"),
        "CoverageAssessmentAgent": ("覆", "#16a34a"),
        "TaskAllocationAgent": ("任", "#ca8a04"),
        "PathPlanningAgent": ("路", "#ea580c"),
        "SafetyReviewAgent": ("审", "#dc2626"),
    }
    grouped: dict[int, list[AgentMessage]] = {}
    for message in result["meetings"]:
        grouped.setdefault(message.meeting_id, []).append(message)
    for meeting_id, messages in grouped.items():
        time_step = messages[0].time_step if messages else "-"
        st.markdown(f"<div class='meeting-town'><div class='meeting-town-title'>第 {meeting_id} 场专家会议｜AI 小镇规划室｜t={time_step}</div><div class='small-muted'>每张专家卡片代表一个领域专家的质询、回应、修正或复审发言。</div></div>", unsafe_allow_html=True)
        for message in messages:
            initial, color = agent_initials.get(message.agent, ("专", "#475467"))
            st.markdown(
                f"""
<div class="agent-card">
  <div class="agent-avatar" style="background:{color}">{initial}</div>
  <div>
    <div class="agent-headline">{AGENT_NAMES[message.agent]}<span class="agent-badge">{message.phase}</span><span class="agent-badge">立场：{message.stance}</span></div>
    <div class="small-muted">{message.agent}</div>
    <div class="agent-speech">{message.speech}</div>
    <div class="small-muted">证据：{message.evidence}</div>
    <div class="small-muted">下一步：{message.next_action}</div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            time.sleep(delay)


def apply_demo_preset(preset_name: str) -> None:
    if preset_name == "自定义":
        return
    base_state = {
        "launch_row": 5,
        "launch_col": 5,
        "constraints": ["高风险区域节点失效"],
        "low_count": 0,
        "failed_count": 0,
        "task_r_0": 18,
        "task_c_0": 42,
        "task_rad_0": 6,
        "task_r_1": 42,
        "task_c_1": 42,
        "task_rad_1": 6,
        "task_r_2": 30,
        "task_c_2": 22,
        "task_rad_2": 6,
    }
    base_state.update(DEMO_PRESETS[preset_name])
    for key, value in base_state.items():
        st.session_state[key] = value
    st.session_state["scenario_notice"] = f"已应用演示预设：{preset_name}"


def apply_selected_demo_preset() -> None:
    apply_demo_preset(st.session_state.get("demo_preset", "自定义"))


def apply_selected_api_provider() -> None:
    provider_name = st.session_state.get("api_provider", "Agnes AI（默认）")
    preset = API_PROVIDER_PRESETS[provider_name]
    st.session_state["model_base_url"] = preset["base_url"]
    st.session_state["model_name"] = preset["model"]
    st.session_state["provider_requires_key"] = preset["requires_api_key"]
    st.session_state["model_api_key"] = ""
    st.session_state["provider_notice"] = f"已切换到 {provider_name}；为避免密钥发往错误服务，密码框已清空。"


def main() -> None:
    st.set_page_config(page_title="无人机集群动态多智能体协同仿真台", page_icon="MA", layout="wide")
    apply_style()

    st.title("无人机集群动态多智能体协同仿真台")
    st.markdown(f"<div class='safety-box'><b>安全边界：</b>{SAFETY_STATEMENT}<br/><b>禁止事项：</b>{FORBIDDEN_NOTE}</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("用户手动设置")
        st.caption("所有坐标均为 60x60 抽象网格坐标。")
        st.subheader("智能体运行时")
        llm_mode_label = st.radio(
            "会议推理模式",
            list(LLM_MODE_OPTIONS.keys()),
            key="llm_mode_label",
        )
        requested_llm_mode = LLM_MODE_OPTIONS[llm_mode_label]
        api_provider = st.selectbox(
            "API 提供商",
            list(API_PROVIDER_PRESETS.keys()),
            key="api_provider",
            on_change=apply_selected_api_provider,
        )
        if provider_notice := st.session_state.pop("provider_notice", None):
            st.info(provider_notice)
        provider_preset = API_PROVIDER_PRESETS[api_provider]
        with st.expander("模型 API 配置", expanded=requested_llm_mode != "rule"):
            key_default = {} if "model_api_key" in st.session_state else {"value": os.environ.get("AGNES_API_KEY", "") if api_provider == "Agnes AI（默认）" else ""}
            model_api_key = st.text_input(
                "API Key",
                type="password",
                key="model_api_key",
                **key_default,
            )
            base_default = {} if "model_base_url" in st.session_state else {"value": provider_preset["base_url"]}
            model_base_url = st.text_input(
                "Base URL",
                key="model_base_url",
                **base_default,
            )
            model_default = {} if "model_name" in st.session_state else {"value": provider_preset["model"]}
            model_name = st.text_input(
                "模型",
                key="model_name",
                **model_default,
            )
            requires_default = {} if "provider_requires_key" in st.session_state else {"value": bool(provider_preset["requires_api_key"])}
            provider_requires_key = st.checkbox("此接口需要 API Key", key="provider_requires_key", **requires_default)
            model_timeout = int(st.number_input("API 超时（秒）", 10, 120, 45, key="model_timeout"))
            config_ready = bool(model_base_url.strip() and model_name.strip()) and (
                bool(model_api_key.strip()) or not provider_requires_key
            )
            if st.button("测试模型连接", disabled=not config_ready, use_container_width=True):
                try:
                    test_client = OpenAICompatibleClient(
                        OpenAICompatibleConfig(
                            model_api_key,
                            model_base_url,
                            model_name,
                            model_timeout,
                            provider_requires_key,
                        )
                    )
                    test_result = test_client.chat_json(
                        system_prompt="只返回有效 JSON。",
                        user_prompt='请返回 {"status":"ok","message":"model connection ready"}',
                    )
                    st.success(f"连接成功：{test_result}")
                except CompatibleAPIError as exc:
                    st.error(str(exc))

        llm_mode = requested_llm_mode
        llm_deliberator = None
        if requested_llm_mode != "rule":
            if config_ready:
                llm_panel = LLMMeetingPanel(
                    OpenAICompatibleClient(
                        OpenAICompatibleConfig(
                            model_api_key,
                            model_base_url,
                            model_name,
                            model_timeout,
                            provider_requires_key,
                        )
                    )
                )
                llm_deliberator = llm_panel.deliberate
                st.caption(f"{api_provider} 每场会议调用一次，生成结构化专家意见、交叉质询和候选推荐。")
            else:
                llm_mode = "rule"
                st.warning("模型 API 配置不完整，本次运行自动使用纯规则模式。")

        preset_name = st.selectbox(
            "演示场景预设",
            list(DEMO_PRESETS.keys()),
            key="demo_preset",
            on_change=apply_selected_demo_preset,
        )
        st.caption("选择预设后会立即恢复该演示的任务区、风险区和运行策略；继续修改任意参数即可作为自定义场景运行。")
        if notice := st.session_state.pop("scenario_notice", None):
            st.success(notice)

        seed_default = {} if "random_seed" in st.session_state else {"value": 20250711}
        random_seed = int(st.number_input("仿真随机种子", min_value=0, max_value=99999999, step=1, key="random_seed", **seed_default))
        st.caption("相同场景与随机种子会生成相同的失效事件，便于上台复现。")
        uav_count = st.slider("UAV 数量", 4, 18, 8, key="uav_count")
        launch_row = st.slider("共同起飞点：行", 0, GRID_SIZE - 1, 5, key="launch_row")
        launch_col = st.slider("共同起飞点：列", 0, GRID_SIZE - 1, 5, key="launch_col")
        action_range = st.slider("UAV 作用范围", 2, 10, 5, key="action_range")
        sim_steps = st.slider("最大仿真时长（抽象时间步）", 30, 180, 120, key="sim_steps")
        target_coverage_pct = st.slider("目标覆盖率", 50, 100, 90, key="target_coverage_pct")
        target_coverage = target_coverage_pct / 100
        coverage_mode_label = st.radio("主覆盖指标", list(COVERAGE_MODE_OPTIONS.keys()), index=0, horizontal=True, key="coverage_mode_label")
        coverage_mode = COVERAGE_MODE_OPTIONS[coverage_mode_label]
        rolling_window = st.slider("滚动覆盖窗口（时间步）", 4, 40, 14, key="rolling_window")
        risk_strategy_label = st.radio("风险决策策略", list(RISK_STRATEGY_OPTIONS.keys()), index=1, horizontal=True, key="risk_strategy_label")
        risk_strategy = RISK_STRATEGY_OPTIONS[risk_strategy_label]
        st.caption("策略是初始倾向；仿真会根据覆盖缺口、损失、剩余 UAV 和覆盖停滞动态调整风险姿态。")
        constraints = st.multiselect("限制条件", CONSTRAINT_OPTIONS, default=["高风险区域节点失效"], key="constraints")
        low_count = st.slider("初始低电量 UAV 数量", 0, min(5, uav_count), 1 if "能量不足" in constraints else 0, key="low_count")
        failed_count = st.slider("初始失效 UAV 数量", 0, min(4, uav_count), 1 if "节点故障" in constraints else 0, key="failed_count")
        risk_failure_pct = st.slider("高风险区节点失效概率/时间步", 0, 80, 18, key="risk_failure_pct")
        risk_failure_probability = risk_failure_pct / 100
        speed_label = st.radio("会议展示速度", list(SPEED_OPTIONS.keys()), index=1, horizontal=True, key="speed_label")

        st.markdown("---")
        task_count = st.slider("任务区域数量", 1, 5, 3, key="task_count")
        tasks = task_controls(task_count)

        risk_count = st.slider("高风险失效区数量", 0, 4, 1 if "高风险区域节点失效" in constraints else 0, key="risk_count")
        risk = zone_controls("risk", "高风险区", risk_count, [(36, 34), (24, 46), (44, 22), (30, 30)], 14, "risk", max_radius=24)

        start = st.button("开始动态多智能体会商仿真", type="primary", use_container_width=True)

    zones = {"restricted": [], "risk": risk}

    st.markdown(
        """
<div class="flow-strip">
  <span>用户设置态势</span><b>→</b><span>初始会议</span><b>→</b><span>抽象平滑路径</span><b>→</b>
  <span>时间步推进</span><b>→</b><span>事件/状态/策略触发</span><b>→</b><span>重新会商</span><b>→</b><span>最终位置与路径</span>
</div>
""",
        unsafe_allow_html=True,
    )

    cached_result = st.session_state.get("drone_dynamic_result")
    should_simulate = (
        start
        or cached_result is None
        or cached_result.get("schema_version") != APP_SCHEMA_VERSION
    )
    if should_simulate:
        result = simulate(
            random_seed=random_seed,
            uav_count=uav_count,
            launch=(launch_row, launch_col),
            action_range=action_range,
            sim_steps=sim_steps,
            target_coverage=target_coverage,
            tasks=tasks,
            zones=zones,
            constraints=constraints,
            low_count=low_count,
            failed_count=failed_count,
            risk_failure_probability=risk_failure_probability,
            coverage_mode=coverage_mode,
            rolling_window=rolling_window,
            risk_strategy=risk_strategy,
            llm_mode=llm_mode,
            llm_deliberator=llm_deliberator,
            llm_provider_name=api_provider,
        )
        st.session_state["drone_dynamic_result"] = result
    else:
        result = cached_result

    mode_label = "滚动覆盖率" if result["coverage_mode"] == "rolling" else "累计覆盖率"
    target_reached = result.get("target_reached_step")
    target_status = f"t={target_reached}" if target_reached is not None else "未达标"
    final_min_coverage = min((item["覆盖率"] for item in result["task_coverage"]), default=0.0)
    metrics = st.columns(8)
    metrics[0].metric("抽象空间", f"{GRID_SIZE}x{GRID_SIZE}")
    metrics[1].metric("UAV 数量", len(result["uavs"]))
    metrics[2].metric("任务区域", len(result["tasks"]))
    metrics[3].metric("事件数量", len(result["events"]))
    metrics[4].metric("主指标", mode_label)
    metrics[5].metric("目标覆盖率", f"{result['target_coverage']:.0%}")
    metrics[6].metric("达标时间", target_status)
    metrics[7].metric("主指标最低覆盖率", f"{final_min_coverage:.1%}")
    latest_posture = result.get("risk_postures", [{}])[-1] if result.get("risk_postures") else {}
    st.caption(
        f"当前风险决策：用户策略 {latest_posture.get('用户风险策略', '未记录')} → "
        f"动态姿态 {latest_posture.get('动态风险姿态', '未记录')}；"
        f"风险暴露预算 {latest_posture.get('风险暴露预算', 0)} 架。"
    )
    st.info(f"仿真结束原因：{result['stop_reason']}；实际运行 {result['sim_steps']} / 最大 {result['max_steps']} 个抽象时间步；目标覆盖率作为达标线，不作为立即停止条件。")
    st.caption(
        f"当前智能体会议模式：{result.get('llm_mode', 'rule')}；提供商：{result.get('llm_provider', '未配置')}。"
        "API Key 不会写入仿真结果或 Markdown 报告。"
    )

    if st.session_state.get("linked_time_step", 0) > result["sim_steps"]:
        st.session_state["linked_time_step"] = result["sim_steps"]
    timeline_t = st.slider(
        "联动仿真时间轴",
        0,
        result["sim_steps"],
        min(result["sim_steps"], result["sim_steps"] // 2),
        key="linked_time_step",
    )
    completed_minutes = [item for item in result.get("meeting_minutes", []) if int(item["时间步"]) <= timeline_t]
    if completed_minutes:
        latest_minute = completed_minutes[-1]
        st.caption(
            f"t={timeline_t} 最近会议：第 {latest_minute['会议编号']} 场 {latest_minute.get('会议类型', '会议')}，"
            f"触发原因：{latest_minute['触发原因']}，动态风险姿态：{latest_minute.get('动态风险姿态', '未记录')}。"
        )

    tabs = st.tabs(["最终路径图", "时间轴回放", "当前覆盖热力", "累计覆盖热力", "完成度时间线", "风险热力图", "专家争辩记录", "会议纪要", "决策差异与报告"])
    with tabs[0]:
        st.plotly_chart(build_path_figure(result), use_container_width=True)
        st.caption("曲线是抽象平滑路径，不是经纬度航线或飞控指令。")
    with tabs[1]:
        st.plotly_chart(build_snapshot_figure(result, timeline_t), use_container_width=True)
        st.caption("位置图与当前覆盖热力图共用页面上方的联动时间轴。")
    with tabs[2]:
        current_heat = build_coverage_heatmap(result["histories"], result["uavs"], until_step=timeline_t, current_only=True)
        st.plotly_chart(build_heatmap_figure(current_heat, f"当前时间步覆盖热力图：t={timeline_t}", "Greens"), use_container_width=True)
        st.caption("当前覆盖热力只显示选定时间步附近 UAV 作用范围，用于观察覆盖缺口和重规划后的恢复。")
    with tabs[3]:
        st.plotly_chart(build_heatmap_figure(result["cumulative_coverage"], "累计覆盖热力图", "Greens"), use_container_width=True)
        st.caption("累计覆盖热力展示从 t=0 到仿真结束所有 UAV 作用范围覆盖过的区域。")
        if result["coverage_mode"] == "rolling":
            st.caption(f"当前主指标为滚动覆盖率，滚动窗口 {result['rolling_window']} 步；详细数值保留在完成度时间线和报告表中。")
    with tabs[4]:
        st.plotly_chart(build_metrics_figure(result), use_container_width=True)
        st.dataframe(result["metrics_timeline"], use_container_width=True, hide_index=True)
    with tabs[5]:
        st.plotly_chart(build_heatmap_figure(result["risk_heatmap"], "限制与风险热力图", "YlOrRd"), use_container_width=True)
    with tabs[6]:
        st.subheader("专家共享态势黑板")
        st.dataframe(planning_blackboard_df(result), use_container_width=True, hide_index=True)
        st.subheader("会议实际评估的候选方案")
        st.dataframe(
            planning_candidates_df(result),
            use_container_width=True,
            hide_index=True,
            column_config={
                "覆盖离散度": st.column_config.NumberColumn(format="%.3f"),
                "总风险评分": st.column_config.NumberColumn(format="%.2f"),
                "平均移动距离": st.column_config.NumberColumn(format="%.1f"),
                "综合评分": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption("候选方案由相同共享态势分别推演；标记为“入选”的 profile 才会写入真实 UAV。")
        st.subheader("专家交叉质询")
        st.dataframe(planning_reviews_df(result), use_container_width=True, hide_index=True)
        st.subheader("完整专家会议记录")
        render_meeting(result, SPEED_OPTIONS[speed_label] if start else 0.0)
    with tabs[7]:
        st.dataframe(meeting_minutes_df(result), use_container_width=True, hide_index=True)
    with tabs[8]:
        st.subheader("动态风险姿态记录")
        st.dataframe(pd.DataFrame(result.get("risk_postures", [])), use_container_width=True, hide_index=True)
        st.subheader("会议决策差异表")
        st.dataframe(decision_changes_df(result), use_container_width=True, hide_index=True)
        st.subheader("主指标各任务区覆盖率")
        st.dataframe(task_coverage_df(result), use_container_width=True, hide_index=True)
        st.subheader("累计覆盖率 / 滚动覆盖率对照")
        st.dataframe(
            pd.DataFrame(result["cumulative_task_coverage"]).merge(
                pd.DataFrame(result["rolling_task_coverage"]),
                on=["任务区", "中心", "半径", "总网格"],
                suffixes=("（累计）", "（滚动）"),
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("每架 UAV 的最终建议")
        st.dataframe(uav_summary_df(result), use_container_width=True, hide_index=True)
        st.subheader("事件时间线")
        st.dataframe(event_df(result), use_container_width=True, hide_index=True)
        report, handoff, trace = write_markdown_outputs(result)
        errors = st.session_state.get("drone_swarm_report_write_errors", [])
        if errors:
            st.warning("Markdown 已在内存中生成，但当前进程可能没有权限覆盖 outputs/reports。可使用下载按钮保存。")
            with st.expander("写入失败详情"):
                for error in errors:
                    st.code(error)
        else:
            st.success("Markdown 输出已覆盖生成到 outputs/reports。")
        st.download_button("下载完整仿真报告", report, file_name=REPORT_PATH.name, mime="text/markdown")
        st.download_button("下载 PPT 摘要", handoff, file_name=HANDOFF_PATH.name, mime="text/markdown")
        st.download_button("下载会议 Trace", trace, file_name=TRACE_PATH.name, mime="text/markdown")

    st.caption(SAFETY_STATEMENT)


if __name__ == "__main__":
    main()
