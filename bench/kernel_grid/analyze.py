"""Per-cell and per-pair runtime verdicts from time_cells.py output.

usage: python analyze.py <out_dir> <old tree> <new tree> [> report.txt]

<old tree> and <new tree> are the tree roots given to capture.py (the label each capture
records ends in the tree's directory name).  Comparisons, each candidate over its baseline:

  shipped change   shipped@<new tree> over shipped@<old tree>, for pairs listed because their
                   shipped C++ changed and the old kernel was correct;
  new compiler     new@<new tree> over shipped@<new tree>, for pairs listed because the new
                   compiler's C++ declares C_nonempty (and over shipped@<old tree> as well,
                   reported beside it); "new, other setting" is the same comparison for the
                   kernel the other register-blocking setting emits, where it differs;
  control          the first arm called twice per counted call over the first arm, in every
                   cell: a ratio near 2 that every cell must report as a regression.

Statistic (fixed before any run; README.md):
  * Arms are timed in the same rounds, so each round gives one paired ratio,
    candidate / baseline.  The estimate is the median of those ratios pooled over processes.
  * Its 95% interval comes from a two-level bootstrap: processes with replacement, then
    rounds within each drawn process with replacement, 2000 draws, percentile bounds.
  * The A/A floor of a cell is the same estimate and interval for each arm over its own
    copy.  floor = the largest distance from 1 of either A/A interval bound, over the
    candidate's and the baseline's copies.
  * Verdict per cell:
    - NEUTRAL-OR-BETTER when the comparison's upper bound is at most 1 + floor;
    - REGRESSION when its lower bound is above 1 + floor;
    - UNRESOLVED otherwise.

Every captured cell of every pair gets one row per comparison, timed or not.  A row whose
kernels were not both correct says which was not:
  * CANDIDATE INCORRECT: the candidate raised or returned wrong values.  A correctness
    finding, whatever the baseline did.
  * BASELINE INCORRECT: the baseline raised or returned wrong values, so there is no correct
    kernel to be slower than and the cell owes no timing (Bobby, 2026-10-12 07:47).
  * NOT CAPTURED / NOT TIMED: the run is incomplete for that cell.

Verdict per pair and comparison, the first that applies:
  candidate incorrect  some cell's candidate was incorrect;
  baseline incorrect   some cell's baseline was incorrect (its timed cells are still printed);
  regression           some cell is a REGRESSION;
  incomplete           some cell was not captured or not timed;
  pass                 every cell is NEUTRAL-OR-BETTER;
  unresolved           otherwise.

The report ends with the control's result.  A control that does not fire in every timed cell
means the run measured nothing, and no verdict above it stands.
"""

from __future__ import annotations

import glob
import json
import os
import random
import statistics
import sys
from collections import defaultdict

OUT = sys.argv[1]
OLD = os.path.basename(os.path.realpath(sys.argv[2]))
NEW = os.path.basename(os.path.realpath(sys.argv[3]))
DRAWS = 2000


def tree_of(arm: str) -> str:
    return arm.split("@", 1)[1].split("#")[0].split("_", 1)[1]


def variant_of(arm: str) -> str:
    return arm.split("@", 1)[0]


def paired(times, cand: str, base: str):
    """{process: [candidate / baseline per round]} for rounds both arms ran."""

    by = defaultdict(dict)
    for (arm, process, round_index), value in times.items():
        if arm in (cand, base):
            by[(process, round_index)][arm] = value
    ratios = defaultdict(list)
    for (process, _), arms in sorted(by.items()):
        if cand in arms and base in arms:
            ratios[process].append(arms[cand] / arms[base])
    return ratios


def estimate(ratios, rng):
    pooled = [r for values in ratios.values() for r in values]
    if not pooled:
        return None
    point = statistics.median(pooled)
    processes = sorted(ratios)
    draws = []
    for _ in range(DRAWS):
        sample = []
        for process in rng.choices(processes, k=len(processes)):
            values = ratios[process]
            sample.extend(rng.choices(values, k=len(values)))
        draws.append(statistics.median(sample))
    draws.sort()
    return point, draws[int(0.025 * DRAWS)], draws[int(0.975 * DRAWS) - 1]


def verdict(comparison, floor):
    _, low, high = comparison
    if high <= 1 + floor:
        return "NEUTRAL-OR-BETTER"
    if low > 1 + floor:
        return "REGRESSION"
    return "UNRESOLVED"


def capture_records():
    """{key: {arm: capture record}} over every capture file; a later record replaces an
    earlier one for the same cell and arm."""

    cells = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(OUT, "capture.*.jsonl"))):
        for line in open(path):
            record = json.loads(line)
            if "key" in record:
                arm = f"{record['variant']}@{record['label']}"
                cells[record["key"]][arm] = record
    return cells


def timing_records():
    """({key: {(arm, process, round): ns per call}}, {key: why the timing skipped it})."""

    times = defaultdict(dict)
    skipped = {}
    for path in sorted(glob.glob(os.path.join(OUT, "times.*.jsonl"))):
        for line in open(path):
            record = json.loads(line)
            if "skipped" in record:
                skipped[record["key"]] = record["skipped"]
            if "arm" not in record:
                continue
            times[record["key"]][
                (record["arm"], record["process"], record["round"])
            ] = record["ns_per_call"]
    return times, skipped


def comparisons_of(why, by_tree, arms):
    """[(comparison, candidate (variant, tree), baseline (variant, tree))] for one cell."""

    listed = []
    if "shipped_changed_correct_before" in why:
        listed.append(("shipped change", ("shipped", NEW), ("shipped", OLD)))
    if "new_c_nonempty" in why:
        listed.append(("new compiler", ("new", NEW), ("shipped", NEW)))
        listed.append(("new vs old shipped", ("new", NEW), ("shipped", OLD)))
        other = by_tree.get(("new_other", NEW))
        # The other setting's kernel is compared when it is not ``new``'s machine code.
        if other is not None and not arms[other].get("same_kernel_as"):
            listed.append(("new, other setting", ("new_other", NEW), ("shipped", NEW)))
    return listed


def outcome_note(record):
    note = record["outcome"]
    if record.get("failed_families"):
        note += " on " + "+".join(record["failed_families"])
    return note


PAIR_VERDICTS = (
    ("candidate incorrect", lambda r: r.startswith("CANDIDATE INCORRECT")),
    ("baseline incorrect", lambda r: r.startswith("BASELINE INCORRECT")),
    ("regression", lambda r: r == "REGRESSION"),
    ("incomplete", lambda r: "NOT CAPTURED" in r or r.startswith("NOT TIMED")),
)


def pair_verdict(results):
    for name, matches in PAIR_VERDICTS:
        if any(matches(result) for result in results):
            return name
    if all(result == "NEUTRAL-OR-BETTER" for result in results):
        return "pass"
    return "unresolved"


def main():
    rng = random.Random(20261012)
    times, skipped = timing_records()
    cells = capture_records()
    rows = []
    control_fired = control_cells = 0
    for key in sorted(cells):
        arms = cells[key]
        first = next(iter(arms.values()))
        by_tree = {(variant_of(arm), tree_of(arm)): arm for arm in arms}
        cell = times.get(key, {})
        timed_arms = {arm for arm, _, _ in cell}
        for what, cand_key, base_key in comparisons_of(first["why"], by_tree, arms):
            result = None
            for role, wanted in (("CANDIDATE", cand_key), ("BASELINE", base_key)):
                arm = by_tree.get(wanted)
                if arm is None:
                    result = f"{role} NOT CAPTURED ({wanted[0]}@{wanted[1]})"
                elif arms[arm]["outcome"] != "correct":
                    result = (
                        f"{role} INCORRECT ({wanted[0]}@{wanted[1]} "
                        f"{outcome_note(arms[arm])})"
                    )
                if result is not None:
                    break
            if result is not None:
                rows.append((first, what, None, None, result))
                continue
            cand, base = by_tree[cand_key], by_tree[base_key]
            comparison = estimate(paired(cell, cand, base), rng)
            if comparison is None:
                reason = skipped.get(key, "no timing record")
                rows.append((first, what, None, None, f"NOT TIMED ({reason})"))
                continue
            floors = []
            for arm in (cand, base):
                aa = estimate(paired(cell, arm + "#aa", arm), rng)
                if aa is not None:
                    floors.append(max(abs(aa[1] - 1), abs(aa[2] - 1)))
            floor = max(floors) if floors else float("nan")
            rows.append((first, what, comparison, floor, verdict(comparison, floor)))
        for control in sorted(arm for arm in timed_arms if arm.endswith("#x2")):
            primary = control[: -len("#x2")]
            comparison = estimate(paired(cell, control, primary), rng)
            aa = estimate(paired(cell, primary + "#aa", primary), rng)
            floor = max(abs(aa[1] - 1), abs(aa[2] - 1)) if aa else float("nan")
            control_cells += 1
            control_fired += (
                verdict(comparison, floor) == "REGRESSION" and 1.6 < comparison[0] < 2.4
            )
    by_pair = defaultdict(list)
    print(
        f"{'pair':58s} {'dtype':8s} {'cell':8s} {'comparison':18s} "
        f"{'ratio':>7s} {'95% CI':>17s} {'floor':>6s}  verdict"
    )
    for record, what, comparison, floor, result in rows:
        cellname = f"{record['size']}/{record['fill']}"
        by_pair[(record["name"], record["dtype"], what)].append((cellname, result))
        head = (
            f"{record['name'][:58]:58s} {record['dtype']:8s} {cellname:8s} {what:18s}"
        )
        if comparison is None:
            print(f"{head}  {result}")
            continue
        point, low, high = comparison
        print(f"{head} {point:7.3f} [{low:6.3f}, {high:6.3f}] {floor:6.3f}  {result}")
    print()
    summary = defaultdict(lambda: defaultdict(int))
    for (name, dtype, what), results in sorted(by_pair.items()):
        outcome = pair_verdict([result for _, result in results])
        summary[what][outcome] += 1
        if outcome != "pass":
            detail = "; ".join(
                f"{cellname} {result}"
                for cellname, result in results
                if result != "NEUTRAL-OR-BETTER"
            )
            print(f"{outcome:19s} {what:18s} {name} [{dtype}]: {detail}")
    print()
    for what, counts in summary.items():
        print(f"{what}: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    print(
        f"control (first arm called twice): fired in {control_fired} of {control_cells} "
        "timed cells"
        + (
            ""
            if control_cells and control_fired == control_cells
            else "  -- THE RUN MEASURED NOTHING"
        )
    )


if __name__ == "__main__":
    main()
