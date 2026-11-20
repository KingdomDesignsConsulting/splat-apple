#!/usr/bin/env python3
"""Run Phase 143 with two exact order/reverse-order process pairs per workload."""
import sys
import run_phase140_143 as runner

def ordered_jobs(phase):
    if phase != "143":
        raise ValueError("Use the shared runner directly for Phase 140 or smoke")
    jobs = original_jobs(phase)
    result = jobs[:1]  # original contracts and native boundary worker
    for workload in runner.PROFILES:
        for block in range(4):
            order = runner.MODES if block < 2 else runner.MODES[2:] + runner.MODES[:2]
            if block % 2:
                order = order[::-1]
            group = {job["mode"]:job for job in jobs
                     if job["kind"] == "render" and job["workload"] == workload and job["block"] == block}
            result.extend(group[mode] for mode in order)
    return result

original_jobs = runner.jobs_for
if __name__ == "__main__":
    runner.jobs_for = ordered_jobs
    sys.exit(runner.main())
