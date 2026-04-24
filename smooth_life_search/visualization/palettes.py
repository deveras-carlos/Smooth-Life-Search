"""Color palettes used by dashboard frame rendering."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

PALETTES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "signed": (
        (9, 23, 57),
        (55, 94, 151),
        (208, 224, 246),
        (246, 212, 168),
        (176, 79, 44),
    ),
    "objective": (
        (12, 9, 32),
        (71, 18, 99),
        (164, 44, 122),
        (239, 115, 62),
        (252, 232, 126),
    ),
    "support": (
        (5, 11, 24),
        (29, 59, 96),
        (84, 141, 154),
        (217, 179, 95),
        (255, 240, 182),
    ),
    "vitality": (
        (7, 18, 26),
        (26, 88, 83),
        (88, 171, 145),
        (205, 239, 198),
    ),
}


@lru_cache(maxsize=None)
def palette_table(name: str) -> np.ndarray:
    """Return a 256-row RGB lookup table for a named palette."""

    stops = PALETTES[name]
    positions = np.linspace(0.0, 1.0, len(stops))
    ramp = np.linspace(0.0, 1.0, 256)
    palette = np.empty((256, 3), dtype=np.uint8)
    for channel in range(3):
        palette[:, channel] = np.round(
            np.interp(ramp, positions, [color[channel] for color in stops])
        ).astype(np.uint8)
    return palette
