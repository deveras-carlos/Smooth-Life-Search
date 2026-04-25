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
        subpixel_confirm_candidates=3,
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
    for field_name in (
        "support_ema_alpha",
        "subpixel_confirm",
        "subpixel_confirm_candidates",
        "exploitation_score_late_stage",
    ):
        config_value = getattr(config, field_name)
        preset_value = getattr(preset, field_name)
        default_value = getattr(defaults, field_name)
        if config_value == default_value and preset_value != default_value:
            setattr(resolved, field_name, preset_value)
        else:
            setattr(resolved, field_name, config_value)
    return resolved
