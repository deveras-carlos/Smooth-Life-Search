"""Known SmoothLife parameter presets."""

from __future__ import annotations

from dataclasses import replace

from .config import SmoothLifeConfig


PRESET_BUILDERS: dict[str, SmoothLifeConfig] = {
    "paper_glider": SmoothLifeConfig(
        birth_low=0.278,
        birth_high=0.365,
        death_low=0.267,
        death_high=0.445,
        alpha_n=0.028,
        alpha_m=0.147,
        inner_radius=7.0,
        outer_radius=21.0,
        dt=0.25,
        diffusion=0.10,
        objective_coupling=0.0,
        run_mode="simulation",
    ),
    "search": SmoothLifeConfig(
        birth_low=0.278,
        birth_high=0.365,
        death_low=0.267,
        death_high=0.445,
        alpha_n=0.028,
        alpha_m=0.147,
        inner_radius=7.0,
        outer_radius=21.0,
        dt=0.25,
        diffusion=0.10,
        objective_coupling=0.30,
        run_mode="search",
    ),
    "search_exploit": SmoothLifeConfig(
        birth_low=0.278,
        birth_high=0.365,
        death_low=0.267,
        death_high=0.445,
        alpha_n=0.028,
        alpha_m=0.147,
        inner_radius=7.0,
        outer_radius=21.0,
        dt=0.25,
        diffusion=0.10,
        objective_coupling=0.30,
        objective_gamma=1.0,
        support_ema_alpha=0.30,
        subpixel_best_point=True,
        subpixel_confirm=True,
        exploitation_score_late_stage=True,
        run_mode="search",
    ),
}


def apply_preset(config: SmoothLifeConfig) -> SmoothLifeConfig:
    """Resolve a preset onto a config instance."""

    if config.preset is None:
        return config
    if config.preset not in PRESET_BUILDERS:
        raise ValueError(f"unknown SmoothLife preset: {config.preset}")
    preset = PRESET_BUILDERS[config.preset]
    defaults = SmoothLifeConfig()
    resolved = replace(
        preset,
        grid_shape=config.grid_shape,
        anti_alias_radius=config.anti_alias_radius,
        dt=config.dt,
        diffusion=config.diffusion,
        objective_coupling=config.objective_coupling,
        objective_gamma=config.objective_gamma,
        field_floor=config.field_floor,
        field_ceiling=config.field_ceiling,
        initial_field_center=config.initial_field_center,
        initial_field_noise=config.initial_field_noise,
        evaluations_per_step=config.evaluations_per_step,
        snapshot_interval=config.snapshot_interval,
        time_mode=config.time_mode,
        run_mode=config.run_mode,
        maximize=config.maximize,
        preset=config.preset,
        store_all_snapshots=config.store_all_snapshots,
        subpixel_best_point=config.subpixel_best_point,
    )
    if config.preset == "search_exploit":
        if config.support_ema_alpha == defaults.support_ema_alpha:
            resolved.support_ema_alpha = preset.support_ema_alpha
        else:
            resolved.support_ema_alpha = config.support_ema_alpha
        if config.subpixel_confirm == defaults.subpixel_confirm:
            resolved.subpixel_confirm = preset.subpixel_confirm
        else:
            resolved.subpixel_confirm = config.subpixel_confirm
        if config.exploitation_score_late_stage == defaults.exploitation_score_late_stage:
            resolved.exploitation_score_late_stage = preset.exploitation_score_late_stage
        else:
            resolved.exploitation_score_late_stage = config.exploitation_score_late_stage
    else:
        resolved.support_ema_alpha = config.support_ema_alpha
        resolved.subpixel_confirm = config.subpixel_confirm
        resolved.exploitation_score_late_stage = config.exploitation_score_late_stage
    return resolved
