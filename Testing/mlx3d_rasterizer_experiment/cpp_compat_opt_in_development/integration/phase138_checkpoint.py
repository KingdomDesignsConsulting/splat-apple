"""Phase138 CPU-only helpers: stable GPU entry and fail-closed fresh-process checkpoints.

Only a fully attested, checksum-matched completed worker checkpoint may be
reused. Old Phase138 artifact directories lacking these records are NOT resumable
for qualification: their original per-worker GPU admission cannot be proven.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time

from qualification_common import digest, gpu_gate

SCHEMA = 2
MAX_GPU_PERCENT = 30
MIN_FREE_MEMORY_PERCENT = 20


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def hash_inventory(packages):
    return hashlib.sha256(json.dumps(packages, sort_keys=True).encode()).hexdigest()


def write_once(path, value):
    """Exclusive immutable checkpoint creation; no replacement or silent overwrite."""
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())


def stable_gpu_gate(report, *, reader=None, sleeper=None, attempts=12,
                    cooldown_seconds=10, confirm_seconds=2, settle_seconds=0):
    """Require two successive <=30% independent samples before a GPU worker.

    A busy reading resets the consecutive count. The 2nd accepted sample is
    the actual launch gate. ioreg observes the whole device, not a process.
    """
    if reader is None:
        reader = gpu_gate
    if sleeper is None:
        sleeper = time.sleep
    if not (attempts >= 2 and cooldown_seconds >= 0 and confirm_seconds > 0
            and settle_seconds >= 0):
        raise ValueError('Invalid stable GPU gate configuration')
    if settle_seconds:
        print(f'GPU_SETTLE {settle_seconds:g}s after prior worker', flush=True)
        sleeper(settle_seconds)
    samples = []
    accepted = []
    for attempt in range(1, attempts + 1):
        try:
            sample = reader()
        except RuntimeError as exc:
            if not str(exc).startswith('GPU test blocked:'):
                raise
            samples.append({'attempt':attempt, 'status':'BUSY', 'message':str(exc),
                            'utc':utc_now()})
            accepted.clear()
            print(f'GPU_GATE_BUSY {attempt} of {attempts}: {exc}', flush=True)
        else:
            # Double-check the reader's stated maximum rather than trusting its name.
            utilization = sample.get('utilization_pct')
            if (isinstance(utilization, bool) or not isinstance(utilization, int)
                    or utilization < 0 or utilization > MAX_GPU_PERCENT):
                raise RuntimeError(f'GPU reader did not establish <=30%: {utilization!r}')
            record = {'attempt':attempt, 'status':'PASS',
                      'utc':sample['utc'], 'utilization_pct':utilization}
            samples.append(record)
            accepted.append(record)
            if len(accepted) >= 2:
                report['gpu_gate_attempts'].append(samples)
                print('GPU_GATE_STABLE', {'previous':accepted[-2],
                                          'launch':accepted[-1]}, flush=True)
                return {'utilization_pct':utilization, 'utc':sample['utc'],
                        'confirmations':accepted[-2:], 'attempts':samples}
        if attempt < attempts:
            sleeper(confirm_seconds if accepted else cooldown_seconds)
    report['gpu_gate_attempts'].append(samples)
    raise RuntimeError(f'GPU remained unstable across {attempts} readings; '
                       'no worker was launched')


def manifest_for(*, invariants, packages, script_sha, helper_sha,
                 adaptive_sha, matrix, python, platform, attested):
    return {'phase':138, 'checkpoint_schema':SCHEMA,
            'invariants':invariants, 'inventory_sha256':hash_inventory(packages),
            'experiment_sha256':script_sha, 'helper_sha256':helper_sha,
            'adaptive_sha256':adaptive_sha,
            'matrix':[list(item) for item in matrix],
            'python':python, 'platform':platform,
            'attested_m4':bool(attested), 'created_utc':utc_now()}


def validate_manifest(source_directory, expected):
    source = Path(source_directory).resolve(strict=True)
    if not source.is_dir():
        raise ValueError(f'Resume source must be an artifact directory: {source}')
    marker = source/'phase138_manifest.json'
    if not marker.is_file():
        raise RuntimeError('This artifact directory predates Phase138 checkpoints. '
                           'It lacks per-worker GPU admission provenance and '
                           'cannot be reused as a qualification PASS; preserve it.')
    actual = json.loads(marker.read_text())
    for field in ('phase', 'checkpoint_schema', 'invariants', 'inventory_sha256',
                  'experiment_sha256', 'helper_sha256', 'adaptive_sha256',
                  'matrix', 'python', 'platform', 'attested_m4'):
        if actual.get(field) != expected.get(field):
            raise RuntimeError(f'Resume manifest mismatch for {field}; '
                               'the source matrix/environment/code has changed')
    lock = source/'phase138_active.lock'
    if not lock.is_file():
        raise RuntimeError('Missing source run lock; unable to prove writer is inactive')
    with lock.open('r') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Source Phase138 run is still active; '
                               'do not resume from a live artifact directory') from exc
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return source


def completed_checkpoint(source, label, *, mode, workload, block, invariants):
    """Return (record, worker-report) or None. Reject rather than cherry-pick bad evidence."""
    source = Path(source)
    checkpoint = source/(label+'.checkpoint.json')
    if not checkpoint.exists():
        return None
    entry = json.loads(checkpoint.read_text())
    if (entry.get('schema') != SCHEMA or entry.get('status') != 'PASS'
            or entry.get('label') != label or entry.get('mode') != mode
            or entry.get('workload') != workload or entry.get('block') != block
            or entry.get('free_memory_pct', -1) < MIN_FREE_MEMORY_PERCENT
            or entry.get('returncode') != 0
            or entry.get('invariants') != invariants):
        raise RuntimeError(f'Invalid completed-worker provenance: {checkpoint}')
    gate = entry.get('gpu_gate', {})
    confirmations = gate.get('confirmations', [])
    recorded_attempts = gate.get('attempts', [])
    if (len(confirmations) != 2 or len(recorded_attempts) < 2
            or recorded_attempts[-2:] != confirmations
            or not all(row.get('status') == 'PASS' and
                       isinstance(row.get('utilization_pct'), int) and
                       0 <= row['utilization_pct'] <= MAX_GPU_PERCENT for row in confirmations)
            or gate.get('utilization_pct') != confirmations[-1]['utilization_pct']):
        raise RuntimeError(f'Missing or invalid dual GPU admission gate: {checkpoint}')
    path = source/(label+'.json')
    log = source/(label+'.log')
    if (not path.is_file() or not log.is_file()
            or digest(path) != entry.get('json_sha256')
            or digest(log) != entry.get('log_sha256')):
        raise RuntimeError(f'Worker file integrity mismatch: {label}')
    report = json.loads(path.read_text())
    if (report.get('phase') != 138 or report.get('status') != 'PASS'
            or report.get('mode') != mode or report.get('workload') != workload
            or report.get('block') != block or report.get('invariants') != invariants
            or not report.get('frames') or not report.get('timeline')):
        raise RuntimeError(f'Worker report does not match checkpoint: {label}')
    return entry, report


def copy_verified_checkpoint(source, destination, label, entry):
    """Copy original evidence, then store a new immutable provenance record."""
    source, destination = Path(source), Path(destination)
    for suffix in ('.json','.log'):
        src = source/(label+suffix)
        dst = destination/(label+suffix)
        with src.open('rb') as stream, dst.open('xb') as target:
            shutil.copyfileobj(stream, target)
    copied = dict(entry)
    copied['reused_from'] = {
        'source_artifact_directory':str(source.resolve()),
        'source_checkpoint_sha256':digest(source/(label+'.checkpoint.json')),
        'copied_utc':utc_now(),
    }
    if (digest(destination/(label+'.json')) != entry['json_sha256'] or
            digest(destination/(label+'.log')) != entry['log_sha256']):
        raise RuntimeError(f'Copied worker evidence mismatch: {label}')
    write_once(destination/(label+'.checkpoint.json'), copied)
    return copied


def save_checkpoint(directory, label, entry, invariants):
    """Only call after returncode 0 and a complete successful child report."""
    record = {'schema':SCHEMA, 'label':label, 'status':'PASS',
              'mode':entry['mode'], 'workload':entry['workload'],
              'block':entry['block'], 'returncode':0,
              'free_memory_pct':entry['free_memory_pct'],
              'gpu_gate':entry['gpu_gate'], 'invariants':invariants,
              'json_sha256':digest(Path(directory)/(label+'.json')),
              'log_sha256':digest(Path(directory)/(label+'.log')),
              'completed_utc':utc_now()}
    write_once(Path(directory)/(label+'.checkpoint.json'), record)
    return record
