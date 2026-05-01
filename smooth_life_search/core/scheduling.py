"""Optional runtime scheduling for SmoothLife parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


SMOOTHLIFE_PER_STEP_FIELDS = frozenset(
    {
        "dt",
        "diffusion",
        "objective_coupling",
        "objective_gamma",
        "support_ema_alpha",
        "evaluations_per_step",
    }
)

ZOOM_BOUNDARY_FIELDS = frozenset(
    {
        "birth_low",
        "birth_high",
        "death_low",
        "death_high",
        "alpha_n",
        "alpha_m",
        "inner_radius",
        "outer_radius",
        "anti_alias_radius",
    }
)

RUN_CONSTANT_FIELDS = frozenset(
    {
        "grid_shape",
        "field_floor",
        "field_ceiling",
        "initial_field_center",
        "initial_field_noise",
        "snapshot_interval",
        "time_mode",
        "run_mode",
        "maximize",
        "preset",
        "store_all_snapshots",
    }
)

KERNEL_PARAMETER_FIELDS = frozenset({"inner_radius", "outer_radius", "anti_alias_radius"})

SCHEDULE_MODES = frozenset(
    {
        "constant",
        "zoom_linear",
        "budget_sigmoid",
        "plateau_reactive",
        "decision_gap_reactive",
    }
)


@dataclass(slots=True)
class RuntimeSignals:
    """Live signals available to runtime schedules."""

    step_index: int
    zoom_index: int
    evaluations: int
    remaining_budget: int | None
    total_budget: int | None
    explored_fraction: float
    current_box_widths: tuple[float, float]
    best_value: float
    local_best_value: float
    global_improvement: float
    stage_improvement: float
    basin_count: int = 0
    top_basin_score_gap: float = 0.0
    max_zoom_cycles: int = 1

    def budget_fraction(self) -> float:
        if self.total_budget is None or self.total_budget <= 0:
            return 0.0
        used = max(float(self.total_budget - max(self.remaining_budget or 0, 0)), 0.0)
        return min(1.0, used / float(self.total_budget))

    def zoom_fraction(self) -> float:
        if self.max_zoom_cycles <= 1:
            return 0.0
        return min(1.0, max(0.0, float(self.zoom_index) / float(self.max_zoom_cycles - 1)))


def _coerce_like(reference: Any, value: float) -> Any:
    if isinstance(reference, bool):
        return bool(value)
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(round(value))
    return float(value)


def _summary_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value


@dataclass(slots=True)
class ScheduleFieldStats:
    """Tracking summary for one scheduled field during a trial."""

    field_name: str
    schedule_mode: str
    resolution_count: int = 0
    update_count: int = 0
    start_value: Any | None = None
    end_value: Any | None = None
    min_value: Any | None = None
    max_value: Any | None = None
    _last_value: Any | None = field(default=None, init=False, repr=False)

    def observe(self, value: Any) -> None:
        resolved = _summary_value(value)
        self.resolution_count += 1
        if self.start_value is None:
            self.start_value = resolved
            self.min_value = resolved
            self.max_value = resolved
        if self._last_value is not None and resolved != self._last_value:
            self.update_count += 1
        self.end_value = resolved
        self._last_value = resolved
        if isinstance(resolved, (int, float)) and not isinstance(resolved, bool):
            if self.min_value is None or float(resolved) < float(self.min_value):
                self.min_value = resolved
            if self.max_value is None or float(resolved) > float(self.max_value):
                self.max_value = resolved

    def to_payload(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "schedule_mode": self.schedule_mode,
            "resolution_count": int(self.resolution_count),
            "update_count": int(self.update_count),
            "start_value": self.start_value,
            "end_value": self.end_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }


@dataclass(slots=True)
class FieldSchedule:
    """Schedule for one mutable field."""

    field_name: str
    mode: str
    base_value: Any
    low_value: Any | None = None
    high_value: Any | None = None
    plateau_threshold: float = 1e-6
    zoom_fraction_threshold: float = 0.6
    activation_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in SCHEDULE_MODES:
            raise ValueError(f"unknown schedule mode: {self.mode}")
        if not 0.0 <= self.activation_fraction < 1.0:
            raise ValueError("activation_fraction must be in [0, 1)")

    def evaluate(self, signals: RuntimeSignals) -> Any:
        if self.mode == "constant":
            return self.base_value
        if self.mode == "zoom_linear":
            start = self.base_value if self.low_value is None else self.low_value
            end = self.high_value if self.high_value is not None else self.base_value
            fraction = signals.zoom_fraction()
            if self.activation_fraction > 0.0:
                if fraction <= self.activation_fraction:
                    fraction = 0.0
                else:
                    fraction = (fraction - self.activation_fraction) / (1.0 - self.activation_fraction)
            return _coerce_like(self.base_value, float(start) + (float(end) - float(start)) * fraction)
        if self.mode == "budget_sigmoid":
            start = self.base_value if self.low_value is None else self.low_value
            end = self.high_value if self.high_value is not None else self.base_value
            fraction = signals.budget_fraction()
            sigmoid = 1.0 / (1.0 + math.exp(-8.0 * (fraction - 0.5)))
            return _coerce_like(self.base_value, float(start) + (float(end) - float(start)) * sigmoid)
        if self.mode == "plateau_reactive":
            improvement = max(float(signals.global_improvement), float(signals.stage_improvement))
            if improvement <= self.plateau_threshold and self.high_value is not None:
                return _coerce_like(self.base_value, float(self.high_value))
            return self.base_value
        if self.mode == "decision_gap_reactive":
            if signals.basin_count <= 1:
                return self.base_value
            if signals.top_basin_score_gap <= self.plateau_threshold and self.high_value is not None:
                return _coerce_like(self.base_value, float(self.high_value))
            if self.low_value is not None:
                return _coerce_like(self.base_value, float(self.low_value))
            return self.base_value
        raise RuntimeError(f"unsupported schedule mode: {self.mode}")

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "FieldSchedule":
        return cls(**payload)


@dataclass(slots=True)
class SchedulePolicy:
    """Policy that resolves field schedules against live runtime signals."""

    schedules: tuple[FieldSchedule, ...]
    family: str | None = None
    schedule_kind: str = "constant"
    _field_stats: dict[str, ScheduleFieldStats] = field(default_factory=dict, init=False, repr=False)

    def reset_tracking(self) -> None:
        self._field_stats = {
            schedule.field_name: ScheduleFieldStats(
                field_name=schedule.field_name,
                schedule_mode=schedule.mode,
            )
            for schedule in self.schedules
        }

    def resolve(self, signals: RuntimeSignals, allowed_fields: set[str] | frozenset[str]) -> dict[str, Any]:
        if not self._field_stats:
            self.reset_tracking()
        allowed = set(allowed_fields)
        overrides: dict[str, Any] = {}
        for schedule in self.schedules:
            if schedule.field_name not in allowed:
                continue
            value = schedule.evaluate(signals)
            overrides[schedule.field_name] = value
            self._field_stats[schedule.field_name].observe(value)
        return overrides

    def summary(self) -> dict[str, Any]:
        field_summaries = [stats.to_payload() for _, stats in sorted(self._field_stats.items())]
        return {
            "family": self.family,
            "schedule_kind": self.schedule_kind,
            "scheduled_fields": [summary["field_name"] for summary in field_summaries],
            "field_summaries": field_summaries,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "schedule_kind": self.schedule_kind,
            "schedules": [schedule.to_payload() for schedule in self.schedules],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "SchedulePolicy | None":
        if payload is None:
            return None
        policy = cls(
            family=payload.get("family"),
            schedule_kind=str(payload.get("schedule_kind", "constant")),
            schedules=tuple(FieldSchedule.from_payload(item) for item in payload.get("schedules", [])),
        )
        policy.reset_tracking()
        return policy


def build_kernel_shrink_policy(
    inner_radius: float,
    outer_radius: float,
    *,
    end_scale: float = 0.6,
    anti_alias_radius: float | None = None,
) -> "SchedulePolicy":
    """Return a ``SchedulePolicy`` that linearly shrinks the SmoothLife kernel radii over zoom progress.

    At zoom_fraction=0 the radii keep their initial values; at zoom_fraction=1 they are
    scaled by ``end_scale``. ``end_scale`` should be in ``(0, 1]``; values < 1 favour
    exploitation by concentrating CA dynamics on fine-scale structure as the search
    narrows. If ``anti_alias_radius`` is provided, it is shrunk on the same schedule.
    """

    if not 0.0 < end_scale <= 1.0:
        raise ValueError("end_scale must be in (0, 1]")
    if inner_radius <= 0.0 or outer_radius <= inner_radius:
        raise ValueError("require 0 < inner_radius < outer_radius")
    schedules: list[FieldSchedule] = [
        FieldSchedule(
            field_name="inner_radius",
            mode="zoom_linear",
            base_value=float(inner_radius),
            low_value=float(inner_radius),
            high_value=float(inner_radius) * float(end_scale),
        ),
        FieldSchedule(
            field_name="outer_radius",
            mode="zoom_linear",
            base_value=float(outer_radius),
            low_value=float(outer_radius),
            high_value=float(outer_radius) * float(end_scale),
        ),
    ]
    if anti_alias_radius is not None:
        schedules.append(
            FieldSchedule(
                field_name="anti_alias_radius",
                mode="zoom_linear",
                base_value=float(anti_alias_radius),
                low_value=float(anti_alias_radius),
                high_value=max(0.5, float(anti_alias_radius) * float(end_scale)),
            )
        )
    return SchedulePolicy(
        schedules=tuple(schedules),
        family="kernel_geometry",
        schedule_kind="zoom_linear",
    )


def build_gamma_ramp_policy(
    *,
    start: float = 1.0,
    end: float = 1.25,
    activation_zoom_fraction: float = 0.3,
) -> "SchedulePolicy":
    """Return a delayed zoom-linear objective-gamma ramp policy.

    With the default ``activation_zoom_fraction=0.3`` and a typical
    ``max_zoom_cycles=5``, ``objective_gamma`` begins ramping at zoom_index 2
    and reaches ``end`` at zoom_index 4 — giving the sharpening 60% of the
    run to take effect, instead of only the final zoom.
    """

    if not 0.0 < start <= 3.0:
        raise ValueError("start must be in (0, 3]")
    if not 0.0 < end <= 3.0:
        raise ValueError("end must be in (0, 3]")
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    if not 0.0 <= activation_zoom_fraction < 1.0:
        raise ValueError("activation_zoom_fraction must be in [0, 1)")
    return SchedulePolicy(
        schedules=(
            FieldSchedule(
                field_name="objective_gamma",
                mode="zoom_linear",
                base_value=float(start),
                low_value=float(start),
                high_value=float(end),
                activation_fraction=float(activation_zoom_fraction),
            ),
        ),
        family="objective_gamma",
        schedule_kind="guarded_zoom_linear",
    )


def build_ema_alpha_ramp_policy(
    *,
    start: float = 0.30,
    end: float = 0.05,
    activation_zoom_fraction: float = 0.0,
) -> "SchedulePolicy":
    """Return a zoom-linear ``support_ema_alpha`` ramp policy.

    In the EMA update ``ema = (1-α) * ema + α * support``, a large α tracks the
    current support quickly (responsive, little smoothing) and a small α
    preserves history (heavy smoothing).
    """

    if not 0.0 <= start <= 1.0:
        raise ValueError("start must be in [0, 1]")
    if not 0.0 <= end <= 1.0:
        raise ValueError("end must be in [0, 1]")
    if not 0.0 <= activation_zoom_fraction < 1.0:
        raise ValueError("activation_zoom_fraction must be in [0, 1)")
    return SchedulePolicy(
        schedules=(
            FieldSchedule(
                field_name="support_ema_alpha",
                mode="zoom_linear",
                base_value=float(start),
                low_value=float(start),
                high_value=float(end),
                activation_fraction=float(activation_zoom_fraction),
            ),
        ),
        family="support_ema_alpha",
        schedule_kind="zoom_linear",
    )


def combine_schedule_policies(*policies: "SchedulePolicy | None") -> "SchedulePolicy | None":
    """Merge policy schedules while preserving their existing field order."""

    schedules: list[FieldSchedule] = []
    families: list[str] = []
    kinds: list[str] = []
    for policy in policies:
        if policy is None:
            continue
        schedules.extend(policy.schedules)
        if policy.family:
            families.append(policy.family)
        if policy.schedule_kind:
            kinds.append(policy.schedule_kind)
    if not schedules:
        return None
    return SchedulePolicy(
        schedules=tuple(schedules),
        family="+".join(families) if families else None,
        schedule_kind="+".join(kinds) if kinds else "combined",
    )


__all__ = [
    "FieldSchedule",
    "KERNEL_PARAMETER_FIELDS",
    "RUN_CONSTANT_FIELDS",
    "RuntimeSignals",
    "SMOOTHLIFE_PER_STEP_FIELDS",
    "SCHEDULE_MODES",
    "ScheduleFieldStats",
    "SchedulePolicy",
    "ZOOM_BOUNDARY_FIELDS",
    "build_ema_alpha_ramp_policy",
    "build_gamma_ramp_policy",
    "build_kernel_shrink_policy",
    "combine_schedule_policies",
]
