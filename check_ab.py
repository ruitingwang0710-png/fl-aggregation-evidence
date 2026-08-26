"""Check C2 differences and direct byte equality of worlds A and B."""

from __future__ import annotations

import csv
from pathlib import Path


def directory_bytes(path: Path) -> dict[str, bytes]:
    """Return every regular file name and its complete contents."""
    return {file.name: file.read_bytes() for file in sorted(path.iterdir()) if file.is_file()}


def main() -> None:
    with Path("results/summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    print("=== C2 maximum absolute differences ===")
    for row in rows:
        if row["claim"] == "C2" and row["case"].endswith(("E2", "E3")):
            print(
                f"  {row['case']:<12} executed={row['executed']:<7} "
                f"verdict={row['verdict']:<20} difference={row['max_abs_diff']}"
            )

    print("\n=== Direct byte comparison of A and B bundles ===")
    for level in ("E0", "E1", "E2", "E3"):
        a_files = directory_bytes(Path("evidence/A") / level / "bundle")
        b_files = directory_bytes(Path("evidence/B") / level / "bundle")
        if a_files == b_files:
            print(f"  {level}: byte-identical")
            continue

        all_names = sorted(a_files.keys() | b_files.keys())
        differences = [name for name in all_names if a_files.get(name) != b_files.get(name)]
        print(f"  {level}: differ; files={differences}")


if __name__ == "__main__":
    main()
