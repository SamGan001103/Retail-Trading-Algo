from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trading_algo.data_export import convert_databento_mbp10_to_orderflow_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Databento MBP-10 DBN data (.dbn/.dbn.zst or .zip bundle) "
            "into ny_structure-compatible orderflow CSV."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input path: Databento .zip download, a .dbn/.dbn.zst file, or a directory of DBN files.",
    )
    parser.add_argument(
        "--output",
        default=str(Path("data") / "mnq_databento_orderflow.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--append", action="store_true", help="Append to output CSV if it exists.")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help=(
            "Optional case-insensitive glob filter for source filenames. "
            "Repeatable. Example: --include '*20260218*'"
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Optional case-insensitive glob exclusion filter for source filenames. Repeatable.",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Optional cap on source files; 0 means unlimited.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap on output rows; 0 means unlimited.")
    parser.add_argument(
        "--include-depth-json",
        action="store_true",
        default=True,
        help="Persist depth_bids/depth_asks JSON ladders.",
    )
    parser.add_argument(
        "--no-depth-json",
        action="store_false",
        dest="include_depth_json",
        help="Disable depth JSON columns.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_files = None if int(args.max_files) <= 0 else int(args.max_files)
    max_rows = None if int(args.max_rows) <= 0 else int(args.max_rows)
    try:
        stats = convert_databento_mbp10_to_orderflow_csv(
            input_path=str(args.input),
            output_csv=str(args.output),
            append=bool(args.append),
            include_depth_json=bool(args.include_depth_json),
            include=list(args.include or []),
            exclude=list(args.exclude or []),
            max_files=max_files,
            max_rows=max_rows,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    print("DATABENTO IMPORT COMPLETE")
    print(f"input_path={stats.input_path}")
    print(f"output_csv={stats.output_csv}")
    print(f"files_processed={stats.files_processed}")
    print(f"rows_written={stats.rows_written}")
    print(f"skipped_rows={stats.skipped_rows}")
    print(f"source_files={len(stats.source_files)}")
    print(f"start_utc={stats.start_utc}")
    print(f"end_utc={stats.end_utc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
