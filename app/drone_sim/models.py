from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    random_seed: int
    uav_count: int
    launch: tuple[int, int]
    action_range: int
    sim_steps: int
    target_coverage: float
    coverage_mode: str
    rolling_window: int
    risk_strategy: str
    risk_failure_probability: float


@dataclass
class AgentMessage:
    time_step: int
    meeting_id: int
    phase: str
    agent: str
    stance: str
    speech: str
    evidence: str
    next_action: str
