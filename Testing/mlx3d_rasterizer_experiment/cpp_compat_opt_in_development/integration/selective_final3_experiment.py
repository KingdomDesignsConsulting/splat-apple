"""Phase 86 experimental, workload-selective final3 dispatch.

Opt-in *scoped* hook, not thread-safe. Production default remains unchanged.
Selection depends only on sort-slot capacity and tile domain (no GPU sync).
"""
from contextlib import contextmanager
from radix_final3_experiment import backend,final_pass_plan,stable_final3_argsort

MIN_SLOTS = 8_000_000
MIN_TILES = 8_000

def select_final3(slots, tiles, *, min_slots=MIN_SLOTS, min_tiles=MIN_TILES):
    """Conservative evidence-driven dispatch: large Full-HD-class workloads."""
    if not (isinstance(slots,int) and isinstance(tiles,int)):
        raise TypeError("slots and tiles must be integer scalars")
    if slots < min_slots or tiles < min_tiles:
        return False
    return final_pass_plan(tiles)["eligible"]

@contextmanager
def selective_final3_override(*, min_slots=MIN_SLOTS, min_tiles=MIN_TILES):
    """Use final3 indices_only for eligible sorts; otherwise exact radix4.

    This modifies one process-local Python module function temporarily. It is
    not safe for concurrent renders; use only for serial experimental tests.
    """
    module, _ = backend()
    previous = module._radix_tile_argsort
    def dispatch(keys, tiles):
        if select_final3(int(keys.size),int(tiles),
                         min_slots=min_slots,min_tiles=min_tiles):
            return stable_final3_argsort(keys,tiles,variant="indices_only")
        return previous(keys,tiles)
    module._radix_tile_argsort = dispatch
    try:
        yield
    finally:
        module._radix_tile_argsort = previous
