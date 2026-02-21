from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_databento_module() -> Any:
    try:
        import databento as db  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - runtime dependency path
        raise RuntimeError(
            "Missing Databento SDK. Install with: "
            "`py -3.11 -m pip install databento`."
        ) from exc
    return db


def _job_id(job: Any) -> str:
    if isinstance(job, dict):
        value = job.get("id")
        if value is None:
            raise RuntimeError(f"Batch submit response missing id: {job}")
        return str(value)
    value = getattr(job, "id", None)
    if value is None:
        raise RuntimeError(f"Batch submit response missing id: {job}")
    return str(value)


def _job_id_list(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        try:
            out.append(_job_id(item))
        except Exception:
            continue
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit a Databento Historical batch job for CSV output and download files locally. "
            "Designed for daily splits consumed by scripts/data/backtest_databento_daily.py."
        )
    )
    parser.add_argument("--api-key", default="", help="Databento API key. Defaults to env DATABENTO_API_KEY.")
    parser.add_argument("--dataset", default="GLBX.MDP3", help="Dataset code.")
    parser.add_argument("--schema", default="mbp-10", help="Schema name.")
    parser.add_argument("--symbols", default="MNQ.c.0", help="Comma-separated symbols (e.g. MNQ.c.0,NQ.c.0).")
    parser.add_argument("--stype-in", default="parent", help="Input symbology type.")
    parser.add_argument("--start", required=True, help="UTC start timestamp, e.g. 2026-01-28T00:00:00.")
    parser.add_argument("--end", required=True, help="UTC end timestamp, e.g. 2026-02-19T00:00:00.")
    parser.add_argument(
        "--split-duration",
        default="day",
        choices=["none", "day", "week", "month"],
        help="Batch split duration. day is recommended for daily backtests.",
    )
    parser.add_argument(
        "--encoding",
        default="csv",
        choices=["csv", "dbn", "json"],
        help="Output encoding. csv is recommended for no-DBN-decoder workflow.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "none"],
        help="Output compression preference.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(pathlib.Path("data") / "databento_daily_csv"),
        help="Download output directory.",
    )
    parser.add_argument("--poll-sec", type=float, default=2.0, help="Polling interval while waiting for job completion.")
    parser.add_argument("--max-wait-sec", type=int, default=3600, help="Max wait time for batch job completion.")
    parser.add_argument("--keep-zip", action="store_true", help="Keep archive zip when downloading full job.")
    parser.add_argument("--job-id", default="", help="Existing Databento batch job ID. If set, skip submit and download this job.")
    parser.add_argument("--download-only", action="store_true", help="Require --job-id and skip submit.")
    return parser.parse_args()


def _submit_job(client: Any, args: argparse.Namespace) -> str:
    symbols_arg: str | list[str]
    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip() != ""]
    symbols_arg = symbols if len(symbols) > 1 else (symbols[0] if symbols else "")
    if symbols_arg == "":
        raise RuntimeError("No symbols provided.")

    submit_kwargs = {
        "dataset": str(args.dataset),
        "schema": str(args.schema),
        "symbols": symbols_arg,
        "stype_in": str(args.stype_in),
        "start": str(args.start),
        "end": str(args.end),
        "split_duration": str(args.split_duration),
        "encoding": str(args.encoding),
        "compression": str(args.compression),
    }

    try:
        job = client.batch.submit_job(**submit_kwargs)
    except TypeError as exc:  # pragma: no cover - compatibility path
        raise RuntimeError(
            "Your Databento SDK likely does not support these batch parameters. "
            "Upgrade with: `py -3.11 -m pip install --upgrade databento`."
        ) from exc
    return _job_id(job)


def _wait_for_done(client: Any, job_id: str, poll_sec: float, max_wait_sec: int) -> None:
    started = time.monotonic()
    while True:
        done_jobs = _job_id_list(client.batch.list_jobs("done"))
        if job_id in done_jobs:
            return

        failed_jobs = _job_id_list(client.batch.list_jobs("failed"))
        if job_id in failed_jobs:
            raise RuntimeError(f"Batch job failed: {job_id}")

        elapsed = time.monotonic() - started
        if elapsed >= float(max_wait_sec):
            raise TimeoutError(f"Timed out waiting for batch job {job_id} after {max_wait_sec}s")
        time.sleep(max(0.2, float(poll_sec)))


def main() -> int:
    args = parse_args()
    db = _load_databento_module()

    api_key = (args.api_key or os.getenv("DATABENTO_API_KEY") or "").strip()
    if api_key == "":
        raise RuntimeError("Missing Databento API key. Set DATABENTO_API_KEY or pass --api-key.")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = db.Historical(api_key)

    if args.download_only and str(args.job_id).strip() == "":
        raise RuntimeError("--download-only requires --job-id.")

    if str(args.job_id).strip() != "":
        job_id = str(args.job_id).strip()
        print(f"batch_mode=download_only job_id={job_id}")
    else:
        job_id = _submit_job(client, args)
        print(f"batch_submitted job_id={job_id}")
        _wait_for_done(client, job_id, poll_sec=float(args.poll_sec), max_wait_sec=int(args.max_wait_sec))
        print(f"batch_done job_id={job_id}")

    # Most recent docs expose keep_zip on download; call with keyword if available.
    try:
        files = client.batch.download(job_id=job_id, output_dir=output_dir, keep_zip=bool(args.keep_zip))
    except TypeError:
        files = client.batch.download(job_id=job_id, output_dir=output_dir)

    # SDK may return list[Path] or similar path-like entries.
    resolved = [pathlib.Path(f) for f in list(files)]
    csv_files = sorted([p for p in resolved if p.suffix.lower() == ".csv"])

    print("DATABENTO BATCH DOWNLOAD COMPLETE")
    print(f"job_id={job_id}")
    print(f"output_dir={output_dir}")
    print(f"files_downloaded={len(resolved)}")
    print(f"csv_files={len(csv_files)}")
    if csv_files:
        print(f"first_csv={csv_files[0]}")
        print(f"last_csv={csv_files[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
