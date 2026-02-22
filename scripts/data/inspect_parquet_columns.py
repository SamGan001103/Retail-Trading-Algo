from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Parquet schema and sample rows for backtest mapping.")
    parser.add_argument(
        "--input",
        default=str(Path("data") / "parquet" / "mbp10.parquet"),
        help="Input Parquet file or directory containing parquet files.",
    )
    parser.add_argument("--rows", type=int, default=5, help="Sample row count from the first batch.")
    parser.add_argument("--batch-size", type=int, default=5_000, help="Batch size for reading sample rows.")
    return parser.parse_args()


def _discover_sources(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise RuntimeError(f"Expected .parquet file, got: {path}")
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.rglob("*.parquet") if p.is_file())
        if files:
            return files
        raise RuntimeError(f"No .parquet files found in directory: {path}")
    raise FileNotFoundError(f"Input path not found: {path}")


def main() -> int:
    args = parse_args()
    path = Path(args.input)
    rows = max(1, int(args.rows))
    batch_size = max(rows, int(args.batch_size))

    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - runtime dependency path
        raise RuntimeError("Parquet inspection requires `pyarrow`. Install with `py -3.11 -m pip install pyarrow`.") from exc

    sources = _discover_sources(path)
    first = sources[0]
    pf = pq.ParquetFile(str(first))
    print(f"source={first}")
    print(f"row_groups={pf.num_row_groups}")
    print("columns:")
    for name in pf.schema_arrow.names:
        print(f" - {name}")

    batch_iter = pf.iter_batches(batch_size=batch_size)
    try:
        first_batch = next(batch_iter)
    except StopIteration:
        print("sample_rows=0")
        return 0
    df = first_batch.to_pandas().head(rows)
    print("")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
