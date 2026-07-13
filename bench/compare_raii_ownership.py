#!/usr/bin/env python3
"""Compare matched Scorch RAII benchmark JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare matched JSON outputs from bench_raii_ownership.py and "
            "write per-case, per-family, and Markdown summaries."
        )
    )
    parser.add_argument("before", help="baseline benchmark JSON")
    parser.add_argument("after", help="candidate benchmark JSON")
    parser.add_argument(
        "--output-prefix",
        default="/tmp/scorch-raii-comparison",
        help="path prefix for generated .json, .csv, and .md reports",
    )
    args = parser.parse_args()
    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())

    def key(row: dict[str, object]) -> tuple[object, object]:
        return row["family"], row["case"]

    old = {key(row): row for row in before["results"]}
    new = {key(row): row for row in after["results"]}
    if old.keys() != new.keys():
        missing_after = sorted(old.keys() - new.keys())
        missing_before = sorted(new.keys() - old.keys())
        raise SystemExit(
            f"case mismatch; missing_after={missing_after}, missing_before={missing_before}"
        )

    rows = []
    for case_key in old:
        b, a = old[case_key], new[case_key]
        ratio = a["median_us"] / b["median_us"]
        rows.append(
            {
                "family": case_key[0],
                "case": case_key[1],
                "before_us": b["median_us"],
                "after_us": a["median_us"],
                "after_over_before": ratio,
                "delta_pct": (ratio - 1.0) * 100.0,
                "speedup": 1.0 / ratio,
                "before_p10_us": b["p10_us"],
                "before_p90_us": b["p90_us"],
                "after_p10_us": a["p10_us"],
                "after_p90_us": a["p90_us"],
                "before_cv_pct": b["cv_pct"],
                "after_cv_pct": a["cv_pct"],
                "nonoverlap_regression": a["p10_us"] > b["p90_us"],
                "nonoverlap_improvement": a["p90_us"] < b["p10_us"],
                "review_regression": ratio > 1.05,
            }
        )

    families = []
    for family in dict.fromkeys(row["family"] for row in rows):
        selected = [row for row in rows if row["family"] == family]
        ratios = [row["after_over_before"] for row in selected]
        families.append(
            {
                "family": family,
                "cases": len(selected),
                "geomean_after_over_before": geometric_mean(ratios),
                "geomean_delta_pct": (geometric_mean(ratios) - 1.0) * 100.0,
                "median_after_over_before": statistics.median(ratios),
                "worst_after_over_before": max(ratios),
                "regressions_gt_5pct": sum(ratio > 1.05 for ratio in ratios),
                "improvements_gt_5pct": sum(ratio < 0.95 for ratio in ratios),
                "nonoverlap_regressions": sum(
                    row["nonoverlap_regression"] for row in selected
                ),
            }
        )

    prefix = Path(args.output_prefix)
    payload = {
        "before_metadata": before["metadata"],
        "after_metadata": after["metadata"],
        "overall": {
            "cases": len(rows),
            "geomean_after_over_before": geometric_mean(
                [row["after_over_before"] for row in rows]
            ),
            "regressions_gt_5pct": sum(row["review_regression"] for row in rows),
            "nonoverlap_regressions": sum(row["nonoverlap_regression"] for row in rows),
            "improvements_gt_5pct": sum(
                row["after_over_before"] < 0.95 for row in rows
            ),
        },
        "families": families,
        "cases": rows,
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Scorch RAII performance comparison",
        "",
        f"Overall geomean after/before: {payload['overall']['geomean_after_over_before']:.4f}x",
        f"Cases >5% slower: {payload['overall']['regressions_gt_5pct']}/{len(rows)}; "
        f"non-overlapping p10/p90 regressions: {payload['overall']['nonoverlap_regressions']}",
        "",
        "| Family | Cases | Geomean Δ | Worst ratio | >5% regressions | Non-overlap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in families:
        lines.append(
            f"| {family['family']} | {family['cases']} | "
            f"{family['geomean_delta_pct']:+.2f}% | "
            f"{family['worst_after_over_before']:.3f}x | "
            f"{family['regressions_gt_5pct']} | "
            f"{family['nonoverlap_regressions']} |"
        )
    lines.extend(["", "## Cases requiring review", ""])
    review = sorted(
        (row for row in rows if row["review_regression"]),
        key=lambda row: row["after_over_before"],
        reverse=True,
    )
    if review:
        for row in review:
            lines.append(
                f"- {row['family']} / {row['case']}: "
                f"{row['before_us']:.3f} → {row['after_us']:.3f} us "
                f"({row['delta_pct']:+.2f}%; p10/p90 non-overlap="
                f"{row['nonoverlap_regression']})"
            )
    else:
        lines.append("- None.")
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload["overall"], indent=2))


if __name__ == "__main__":
    main()
