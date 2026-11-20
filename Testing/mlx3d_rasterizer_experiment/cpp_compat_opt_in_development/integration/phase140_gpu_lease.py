"""OS-wide GPU lease and immutable worker evidence for phases 140–143."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

from phase138_checkpoint import utc_now
from qualification_common import command, digest

LOCK_PATH = Path.home() / "Library/Caches/SplatApple/gpu-experiment.lock"
LEGACY = re.compile(r"^(?:(?:run|bench|test)_phase\d+.*|"
                    r"run_overnight_renderer|run_renderer_qualification|"
                    r"run_cpp_compat_api_integrity_gate|test_runtime_adapter|"
                    r"test_persistent_cpp_compat_api)\.py$")

class AdmissionBusy(RuntimeError):
    pass

def legacy_pids(process_table, current_pid):
    processes = {}
    for line in process_table.splitlines():
        columns = line.strip().split(maxsplit=2)
        if len(columns) == 3 and columns[0].isdigit() and columns[1].isdigit():
            processes[int(columns[0])] = (int(columns[1]), columns[2])
    ancestors = set()
    pid = current_pid
    while pid in processes and pid not in ancestors:
        ancestors.add(pid)
        pid = processes[pid][0]
    found = []
    for pid, (_, executable) in processes.items():
        if pid in ancestors:
            continue
        try:
            argv = shlex.split(executable)
        except ValueError:
            argv = executable.split()
        if argv and ("python" in Path(argv[0]).name.lower() or Path(argv[0]).name == "conda"):
            if any(LEGACY.fullmatch(Path(arg).name) for arg in argv[1:]):
                found.append(pid)
    return found

def check_legacy():
    rc, output, _ = command(["ps", "-axo", "pid=,ppid=,command="], timeout=10)
    if rc:
        raise AdmissionBusy("GPU_LOCK_BUSY: process inventory unavailable")
    running = legacy_pids(output, os.getpid())
    if running:
        raise AdmissionBusy(f"GPU_LOCK_BUSY: legacy experiment PID(s) {running}")

@contextmanager
def lease(events, path=LOCK_PATH, *, check=check_legacy):
    check()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AdmissionBusy(f"GPU_LOCK_BUSY: {path}") from exc
        events.append(dict(event="acquired", pid=os.getpid(), utc=utc_now(), path=str(path)))
        try:
            check()
            yield stream.fileno()
        finally:
            # Closing rather than LOCK_UN preserves the lease while any
            # inherited descriptor in a surviving child is still open.
            stream.close()
            events.append(dict(event="owner_closed", pid=os.getpid(), utc=utc_now(), path=str(path)))

def require_inherited_lease(env=os.environ, path=LOCK_PATH):
    fd = int(env.get("SPLAT_LEASE_FD", "-1"))
    if fd < 0 or int(env.get("SPLAT_LEASE_PID", "-1")) != os.getppid():
        raise RuntimeError("GPU worker requires its admitting parent and inherited lease")
    opened, expected = os.fstat(fd), path.stat()
    if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError("GPU worker lease inode differs")
    # Separate open description must see a conflicting lease.
    with path.open("rb") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return fd
    raise RuntimeError("GPU worker inherited no active lease")

def atomic_json(path, value):
    """Publish only fully written files, never replace an existing checkpoint."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    try:
        with os.fdopen(fd, "w") as data:
            json.dump(value, data, indent=2, sort_keys=True)
            data.write("\n")
            data.flush()
            os.fsync(data.fileno())
        os.link(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(tmp)

def checkpoint(directory, label, job, manifest, gate, free, original=None):
    target = directory / (label + ".checkpoint.json")
    report = directory / (label + ".json")
    log = directory / (label + ".log")
    row = dict(job=job, manifest=manifest, gate=gate, free_memory_pct=free,
               returncode=0, completed_utc=utc_now(), original=original or str(directory),
               report_sha=digest(report), log_sha=digest(log))
    atomic_json(target, row)
    return row

def verify(directory, label, job, manifest):
    marker = directory / (label + ".checkpoint.json")
    if not marker.exists():
        return None
    row = json.loads(marker.read_text())
    if row["job"] != job or row["manifest"] != manifest or row["returncode"] != 0:
        raise ValueError(f"Checkpoint provenance mismatch: {label}")
    gate = row["gate"]
    confirmations = gate["confirmations"]
    if (len(confirmations) != 2 or gate["attempts"][-2:] != confirmations
            or any(r["status"] != "PASS" or type(r["utilization_pct"]) is not int
                   or not 0 <= r["utilization_pct"] <= 30 for r in confirmations)
            or gate["utilization_pct"] != confirmations[-1]["utilization_pct"]
            or not 20 <= row["free_memory_pct"] <= 100):
        raise ValueError(f"Invalid admission checkpoint: {label}")
    for suffix, key in [(".json", "report_sha"), (".log", "log_sha")]:
        if digest(directory / (label + suffix)) != row[key]:
            raise ValueError(f"Evidence hash mismatch: {label}{suffix}")
    result = json.loads((directory / (label + ".json")).read_text())
    if result["status"] != "PASS" or result["job"] != job or not result["output_checksums"]:
        raise ValueError(f"Worker output evidence is incomplete: {label}")
    return row, result

def run_child(argv, log, fd, env, cwd):
    argv = list(map(str, argv))
    print("+ " + shlex.join(argv), flush=True)
    child_env = dict(env, SPLAT_LEASE_FD=str(fd), SPLAT_LEASE_PID=str(os.getpid()))
    with log.open("x") as out:
        print("COMMAND:", shlex.join(argv), file=out, flush=True)
        return subprocess.run(argv, cwd=cwd, env=child_env, pass_fds=(fd,),
                              stdout=out, stderr=subprocess.STDOUT, timeout=900).returncode
