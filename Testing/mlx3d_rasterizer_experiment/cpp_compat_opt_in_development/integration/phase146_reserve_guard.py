"""Phase146 explicitly opt-in scene-transition reserve-reset prototype.

This module does not modify mlx_gs or the qualified Fast source. The guard
uses only host-known scene count, prepared camera size and existing reserve.
It is NOT production-qualified; a fresh 65-contract/dispatch/lifecycle gate
is required before any integration.
"""
from phase142_combined import CombinedRenderer

MIN_INHERITED_CAPACITY = 1 << 22
MAX_SMALL_GAUSSIANS = 4096
MAX_SMALL_AREA = 640 * 360


def eligible_transition_reset(*, supported, after_update, gaussians, width, height, capacity):
    """A conservative candidate for the Phase144 large->small transition."""
    return (
        supported is True
        and after_update is True
        and isinstance(gaussians, int) and 0 < gaussians <= MAX_SMALL_GAUSSIANS
        and isinstance(width, int) and isinstance(height, int)
        and 0 < width and 0 < height and width * height <= MAX_SMALL_AREA
        and isinstance(capacity, int) and capacity >= MIN_INHERITED_CAPACITY
    )


class ExperimentalReserveResetRenderer(CombinedRenderer):
    """Limited transition guard; no automatic active sizing or API/default changes."""

    def __init__(self, gaussians, **kwargs):
        self._transition_pending = False
        self.last_reserve_reset = None
        super().__init__(gaussians, **kwargs)

    def update(self, gaussians):
        had_previous = self._renderer is not None
        super().update(gaussians)  # inherited validation/replacement is atomic
        self._transition_pending = had_previous
        self.last_reserve_reset = None

    def _render_prepared(self, camera, background):
        frame = self._renderer
        before = int(frame._capacity)
        should_reset = eligible_transition_reset(
            supported=self.supported,
            after_update=self._transition_pending,
            gaussians=int(frame.n),
            width=int(camera.width),
            height=int(camera.height),
            capacity=before,
        )
        # Clear one-shot eligibility on this first valid render. The native
        # renderer owns subsequent capacity allocation and reserve shrinking.
        self._transition_pending = False
        if should_reset:
            frame._capacity = 0
        self.last_reserve_reset = {
            "applied": bool(should_reset), "capacity_before": before,
            "capacity_after_policy": int(frame._capacity),
            "width": int(camera.width), "height": int(camera.height),
            "gaussians": int(frame.n),
        }
        return super()._render_prepared(camera, background)
