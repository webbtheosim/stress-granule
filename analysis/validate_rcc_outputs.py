#!/usr/bin/env python3
"""Validate that RCC_ANALYSIS produced the files required downstream.

Pipeline completeness checker. For each requested temperature and system
category, confirms that the per-window RCC output directory
``TEMP_{temp}/ANALYSIS_{group}/`` contains the expected count of every
required file family (stress, tracked-cluster, cluster, domain-contact, and
density-profile files), flags zero-byte files, and optionally scans SLURM
logs under ``{root}/logs`` for failure signatures. Expected counts derive
from the module constants (40 windows/system, 28 domain pairs, etc.) times
the number of systems in each group (1 for SG, 10 for DSM/NDSM).

Role: a QA gate, not a figure script. Run after the RCC stage to verify
completeness before launching downstream analysis.

Key inputs (CLI flags):
    --root         analysis repository root (default ".")
    --source-base  source data root used to detect which groups exist
    --temps        temperature labels to check (default 285..315)
    --groups       SG | DSM | NDSM | ALL (default ALL)
    --since-epoch  only count files newer than this mtime (freshness check)
    --scan-logs    also scan {root}/logs/*.out,*.log for failures

Key outputs: human-readable PASS/FAIL lines to stdout/stderr; process exit
code 0 on success, 1 on any validation error.

Exact CLI invocation:
    python validate_rcc_outputs.py [--root .] [--source-base BASE] \
        [--temps 285 ... 315] [--groups ALL] [--since-epoch EPOCH] [--scan-logs]
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Set


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_TEMPS = ("285", "290", "295", "300", "305", "310", "315")
DEFAULT_SOURCE_BASE = os.path.join(_REPO_ROOT, "PYTHON_ANALYSIS")
WINDOWS_PER_SYSTEM = 40
DOMAIN_PAIRS_PER_SYSTEM = 28
SELF_DOMAIN_PAIRS_PER_SYSTEM = 7
CLUSTER_FILES_PER_SYSTEM = 11

REQUIRED_PREFIXES = {
    "Stress_Segmented": WINDOWS_PER_SYSTEM,
    "Stress_Tensor_Tracked": WINDOWS_PER_SYSTEM,
    "Tracked_Cluster": WINDOWS_PER_SYSTEM,
    "Max_Continuous_Cluster": WINDOWS_PER_SYSTEM,
    "Cluster_": CLUSTER_FILES_PER_SYSTEM * WINDOWS_PER_SYSTEM,
    "Domain_Contacts_TotalCombinedFiltered": DOMAIN_PAIRS_PER_SYSTEM * WINDOWS_PER_SYSTEM,
    "Domain_Contacts_TotalInter": DOMAIN_PAIRS_PER_SYSTEM * WINDOWS_PER_SYSTEM,
    "Domain_Contacts_TotalIntraFiltered": SELF_DOMAIN_PAIRS_PER_SYSTEM * WINDOWS_PER_SYSTEM,
    "Domain_Contacts_TotalFull": DOMAIN_PAIRS_PER_SYSTEM * WINDOWS_PER_SYSTEM,
}

FAIL_PATTERNS = re.compile(
    r"Traceback|failed with error|segmented NPZ not found|Disk quota exceeded|"
    r"No such file|Job dependency problem|QOSMaxJobsPerUserLimit|ERROR\b|"
    r"Exception|Killed|CANCELLED|OUT_OF_MEMORY",
    re.IGNORECASE,
)


def expected_groups(source_base: Path, temp: str, requested: Set[str]) -> List[str]:
    """Return the requested system groups that actually exist for a temperature.

    Filters the requested {SG, DSM, NDSM} set down to those whose source
    directory ``source_base/TEMP_{temp}/{group}`` is present on disk.
    """
    groups = []
    for group in ("SG", "DSM", "NDSM"):
        if group in requested and (source_base / f"TEMP_{temp}" / group).is_dir():
            groups.append(group)
    return groups


def count_prefix(files, prefix: str, since_epoch: Optional[float]) -> int:
    """Count directory entries whose name starts with ``prefix``.

    If ``since_epoch`` is given, only files modified at or after that mtime
    are counted (freshness filter); unreadable entries are skipped.
    """
    count = 0
    for entry in files:
        if not entry.name.startswith(prefix):
            continue
        try:
            stat_result = entry.stat()
        except OSError:
            continue
        if since_epoch is not None and stat_result.st_mtime < since_epoch:
            continue
        count += 1
    return count


def validate_group(root: Path, temp: str, group: str, since_epoch: Optional[float]) -> List[str]:
    """Check one (temperature, group) analysis directory for completeness.

    Verifies the directory exists, flags any zero-byte files, and compares
    observed-vs-expected counts for every required file prefix (including
    density profiles and the legacy tmin=0 stress artifact), scaling
    expectations by the number of systems in the group. Returns a list of
    human-readable error strings (empty if the group passes).
    """
    analysis_dir = root / f"TEMP_{temp}" / f"ANALYSIS_{group}"
    errors = []  # type: List[str]
    systems = 1 if group == "SG" else 10

    if not analysis_dir.is_dir():
        return [f"TEMP_{temp} {group}: missing {analysis_dir}"]

    files = [entry for entry in os.scandir(analysis_dir) if entry.is_file()]
    zero_files = []
    for entry in files:
        try:
            if entry.stat().st_size == 0:
                zero_files.append(entry.name)
        except OSError:
            continue
    if zero_files:
        errors.append(
            f"TEMP_{temp} {group}: {len(zero_files)} zero-byte files, examples={zero_files[:5]}"
        )

    for prefix, per_system_expected in REQUIRED_PREFIXES.items():
        expected = per_system_expected * systems
        observed = count_prefix(files, prefix, since_epoch)
        if observed != expected:
            freshness = " fresh" if since_epoch is not None else ""
            errors.append(
                f"TEMP_{temp} {group}: {prefix}{freshness} count {observed} != expected {expected}"
            )

    density_profiles_per_system = 11 if group == "SG" else 12
    expected_density = density_profiles_per_system * WINDOWS_PER_SYSTEM * systems
    observed_density = count_prefix(files, "DensityProfile_", since_epoch)
    if observed_density != expected_density:
        freshness = " fresh" if since_epoch is not None else ""
        errors.append(
            f"TEMP_{temp} {group}: DensityProfile_{freshness} count "
            f"{observed_density} != expected {expected_density}"
        )

    # Legacy tmin=0 stress is still used as a compatibility artifact.
    legacy_stress = [
        entry
        for entry in files
        if entry.name.startswith("Stress_Tensor_") and not entry.name.startswith("Stress_Tensor_Tracked")
        and (since_epoch is None or _fresh_enough(entry, since_epoch))
    ]
    if len(legacy_stress) != systems:
        freshness = " fresh" if since_epoch is not None else ""
        errors.append(
            f"TEMP_{temp} {group}: legacy Stress_Tensor{freshness} count "
            f"{len(legacy_stress)} != expected {systems}"
        )

    return errors


def _fresh_enough(entry, since_epoch: Optional[float]) -> bool:
    """Return True if a dir entry is new enough (mtime >= since_epoch).

    Returns True when ``since_epoch`` is None (no freshness filter) and False
    if the entry's mtime cannot be read.
    """
    if since_epoch is None:
        return True
    try:
        return entry.stat().st_mtime >= since_epoch
    except OSError:
        return False


def scan_logs(log_root: Path, since_epoch: Optional[float]) -> List[str]:
    """Scan ``*.out``/``*.log`` files under a directory for failure signatures.

    Recursively walks ``log_root``, optionally restricted to files newer than
    ``since_epoch``, and records the first line of each log matching
    ``FAIL_PATTERNS`` (tracebacks, OOM, CANCELLED, quota, etc.). Returns a list
    of ``path:lineno: line`` strings.
    """
    errors = []  # type: List[str]
    if not log_root.is_dir():
        return errors

    for path in log_root.rglob("*"):
        if not path.is_file():
            continue
        if since_epoch is not None and path.stat().st_mtime < since_epoch:
            continue
        name = path.name
        if not (name.endswith(".out") or name.endswith(".log")):
            continue
        try:
            with path.open("r", errors="replace") as handle:
                for lineno, line in enumerate(handle, start=1):
                    if FAIL_PATTERNS.search(line):
                        errors.append(f"{path}:{lineno}: {line.strip()}")
                        break
        except OSError as exc:
            errors.append(f"{path}: unable to read log: {exc}")
    return errors


def parse_groups(raw: str) -> Set[str]:
    """Parse the ``--groups`` argument into a set of group names.

    Accepts comma- or space-separated tokens; ``ALL`` expands to
    {SG, DSM, NDSM}. Raises ``SystemExit`` on any unrecognized token.
    """
    groups = set()  # type: Set[str]
    for item in raw.replace(",", " ").split():
        item = item.upper()
        if item == "ALL":
            groups.update(("SG", "DSM", "NDSM"))
        elif item in {"SG", "DSM", "NDSM"}:
            groups.add(item)
        else:
            raise SystemExit(f"Invalid group '{item}'")
    return groups


def main() -> int:
    """Parse CLI args, validate all requested groups, and return an exit code.

    Iterates the requested temperatures and present groups, collecting errors
    from ``validate_group`` (and optionally ``scan_logs``). Prints OK/FAIL
    lines and returns 0 if everything passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Analysis repository root")
    parser.add_argument("--source-base", default=DEFAULT_SOURCE_BASE)
    parser.add_argument("--temps", nargs="*", default=list(DEFAULT_TEMPS))
    parser.add_argument("--groups", default="ALL")
    parser.add_argument("--since-epoch", type=float, default=None)
    parser.add_argument("--scan-logs", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    source_base = Path(args.source_base)
    requested = parse_groups(args.groups)
    all_errors = []  # type: List[str]

    for temp in args.temps:
        groups = expected_groups(source_base, temp, requested)
        if not groups:
            print(f"TEMP_{temp}: no requested source groups present; skipped")
            continue
        for group in groups:
            errors = validate_group(root, temp, group, args.since_epoch)
            if errors:
                all_errors.extend(errors)
            else:
                print(f"TEMP_{temp} {group}: OK")

    if args.scan_logs:
        log_errors = scan_logs(root / "logs", args.since_epoch)
        if log_errors:
            all_errors.append(f"log scan found {len(log_errors)} suspect log(s)")
            all_errors.extend(log_errors[:80])

    if all_errors:
        print("\nRCC VALIDATION FAILED", file=sys.stderr)
        for err in all_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("\nRCC validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
