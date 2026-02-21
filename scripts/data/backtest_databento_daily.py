from __future__ import annotations

import argparse
import csv
import fnmatch
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trading_algo.data_export.databento_orderflow import _is_dbn_filename, convert_databento_mbp10_to_orderflow_csv
from trading_algo.runtime.mode_runner import run_backtest

_DAY_RE = re.compile(r"(20\d{6})")
_TS_COLUMNS = ("timestamp", "datetime", "time", "date", "ts_event", "ts_recv")


@dataclass(frozen=True)
class _SourceRef:
    container: str  # zip | fs
    path: str
    fmt: str  # dbn | csv
    day_tag: str | None


def _is_csv_filename(name: str) -> bool:
    return name.strip().lower().endswith(".csv")


def _extract_day_tag(name: str) -> str | None:
    match = _DAY_RE.search(Path(name).name)
    if match is None:
        return None
    return match.group(1)


def _matches(name: str, include: list[str], exclude: list[str]) -> bool:
    lowered = name.lower()
    if include and not any(fnmatch.fnmatch(lowered, p.lower()) for p in include):
        return False
    if exclude and any(fnmatch.fnmatch(lowered, p.lower()) for p in exclude):
        return False
    return True


def _add_source(out: list[_SourceRef], *, container: str, raw_path: str) -> None:
    fmt = "dbn" if _is_dbn_filename(raw_path) else ("csv" if _is_csv_filename(raw_path) else "")
    if fmt == "":
        return
    out.append(
        _SourceRef(
            container=container,
            path=raw_path,
            fmt=fmt,
            day_tag=_extract_day_tag(raw_path),
        )
    )


def _discover_sources(input_path: Path, include: list[str], exclude: list[str]) -> list[_SourceRef]:
    if not input_path.exists():
        raise RuntimeError(
            f"Input path not found: {input_path}. "
            "Provide a Databento DBN/CSV file, .zip bundle, or directory."
        )
    discovered: list[_SourceRef] = []
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path, "r") as archive:
            for info in sorted(archive.infolist(), key=lambda x: x.filename):
                if info.is_dir():
                    continue
                if not (_is_dbn_filename(info.filename) or _is_csv_filename(info.filename)):
                    continue
                if not _matches(info.filename, include, exclude):
                    continue
                _add_source(discovered, container="zip", raw_path=info.filename)
        return discovered

    if input_path.is_dir():
        for path in sorted(input_path.rglob("*")):
            if not path.is_file():
                continue
            if not (_is_dbn_filename(path.name) or _is_csv_filename(path.name)):
                continue
            if not _matches(path.name, include, exclude):
                continue
            _add_source(discovered, container="fs", raw_path=str(path))
        return discovered

    if input_path.is_file() and (_is_dbn_filename(input_path.name) or _is_csv_filename(input_path.name)):
        if _matches(input_path.name, include, exclude):
            _add_source(discovered, container="fs", raw_path=str(input_path))
        return discovered

    raise RuntimeError(f"Unsupported Databento input path: {input_path}")


def _within_day_bounds(day_tag: str | None, start_day: str | None, end_day: str | None) -> bool:
    if day_tag is None:
        return True
    if start_day is not None and day_tag < start_day:
        return False
    if end_day is not None and day_tag > end_day:
        return False
    return True


def _to_utc_day(raw: str | None) -> str | None:
    text = str(raw or "").strip()
    if text == "":
        return None

    try:
        whole = int(text)
    except ValueError:
        whole = 0
    if whole != 0:
        abs_whole = abs(whole)
        if abs_whole >= 100_000_000_000_000_000:  # ns
            sec, ns = divmod(whole, 1_000_000_000)
            dt = datetime.fromtimestamp(sec, tz=timezone.utc).replace(microsecond=ns // 1000)
            return dt.strftime("%Y%m%d")
        if abs_whole >= 100_000_000_000_000:  # us
            sec, us = divmod(whole, 1_000_000)
            dt = datetime.fromtimestamp(sec, tz=timezone.utc).replace(microsecond=us)
            return dt.strftime("%Y%m%d")
        if abs_whole >= 100_000_000_000:  # ms
            dt = datetime.fromtimestamp(whole / 1000.0, tz=timezone.utc)
            return dt.strftime("%Y%m%d")
        if abs_whole >= 1_000_000_000:  # sec
            dt = datetime.fromtimestamp(float(whole), tz=timezone.utc)
            return dt.strftime("%Y%m%d")

    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(iso_text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y%m%d")


def _pick_timestamp_value(row: dict[str, str]) -> str | None:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for key in _TS_COLUMNS:
        value = lowered.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text != "":
            return text
    return None


def _safe_label(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned[:80] if cleaned else "source"


def _extract_zip_member(input_zip: Path, member: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_zip, "r") as archive:
        with archive.open(member, "r") as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out)


def _split_csv_by_day(
    csv_path: Path,
    *,
    tmp_dir: Path,
    source_label: str,
    start_day: str | None,
    end_day: str | None,
) -> list[tuple[str, Path]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    split_dir = tmp_dir / "_split_days"
    split_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[str, TextIO] = {}
    writers: dict[str, csv.DictWriter] = {}
    paths: dict[str, Path] = {}

    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            if not fieldnames:
                raise RuntimeError(f"CSV has no header: {csv_path}")
            for row in reader:
                day = _to_utc_day(_pick_timestamp_value(row))
                if day is None:
                    continue
                if not _within_day_bounds(day, start_day, end_day):
                    continue
                if day not in writers:
                    out_path = split_dir / f"{_safe_label(source_label)}_{day}.csv"
                    out_f = out_path.open("w", newline="", encoding="utf-8")
                    handles[day] = out_f
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writers[day] = writer
                    paths[day] = out_path
                    writer.writeheader()
                writers[day].writerow(row)
    finally:
        for fh in handles.values():
            try:
                fh.close()
            except OSError:
                pass

    out = [(day, paths[day]) for day in sorted(paths)]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Databento backtests one day at a time from DBN or CSV sources: "
            "day source -> backtest -> append telemetry CSVs -> delete temp files (default)."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Databento .zip, .dbn/.dbn.zst, .csv, or a directory containing DBN/CSV files.",
    )
    parser.add_argument("--strategy", default="ny_structure", help="Backtest strategy key.")
    parser.add_argument("--hold-bars", type=int, default=120, help="Hold bars argument for strategy config.")
    parser.add_argument("--profile", choices=["normal", "debug"], default="normal", help="Runtime profile.")
    parser.add_argument(
        "--tmp-dir",
        default=str(Path("data") / "tmp_daily_backtest"),
        help="Temporary directory used for per-day converted/split files.",
    )
    parser.add_argument("--keep-csv", action="store_true", help="Keep per-day temporary CSV files.")
    parser.add_argument("--start-day", default="", help="Optional lower day bound YYYYMMDD.")
    parser.add_argument("--end-day", default="", help="Optional upper day bound YYYYMMDD.")
    parser.add_argument("--max-days", type=int, default=0, help="Optional cap on processed days; 0 means unlimited.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with next day if one day fails.")
    parser.add_argument("--list-only", action="store_true", help="List selected sources and exit.")
    parser.add_argument("--include", action="append", default=[], help="Optional include glob for source filenames.")
    parser.add_argument("--exclude", action="append", default=[], help="Optional exclude glob for source filenames.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Daily batch replay should keep moving even when a day has out-of-session rows
    # or minor ordering anomalies; explicit env overrides still take precedence.
    if (os.getenv("BACKTEST_PREFLIGHT_STRICT") or "").strip() == "":
        os.environ["BACKTEST_PREFLIGHT_STRICT"] = "false"
    if (os.getenv("BACKTEST_PREFLIGHT_MIN_SESSION_ROWS") or "").strip() == "":
        os.environ["BACKTEST_PREFLIGHT_MIN_SESSION_ROWS"] = "0"

    input_path = Path(args.input)
    include = [x.strip() for x in list(args.include or []) if x.strip() != ""]
    exclude = [x.strip() for x in list(args.exclude or []) if x.strip() != ""]
    start_day = args.start_day.strip() or None
    end_day = args.end_day.strip() or None
    max_days = max(0, int(args.max_days))
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    discovered = _discover_sources(input_path, include, exclude)
    selected = [s for s in discovered if _within_day_bounds(s.day_tag, start_day, end_day)]
    if not selected:
        raise RuntimeError("No Databento sources selected for backtest.")

    print(f"daily_backtest_plan sources={len(selected)} input={input_path} tmp_dir={tmp_dir}")
    for idx, source in enumerate(selected, start=1):
        day = source.day_tag or "unknown"
        print(f"{idx:03d}. day={day} format={source.fmt} source={source.path}")
    if args.list_only:
        return 0

    completed = 0
    failed = 0
    processed_days = 0

    for idx, source in enumerate(selected, start=1):
        if max_days > 0 and processed_days >= max_days:
            break
        source_label = Path(source.path).name
        print(f"\nSOURCE START format={source.fmt} source={source.path}")

        try:
            if source.fmt == "dbn":
                day_tag = source.day_tag or f"day{idx:03d}"
                day_csv = tmp_dir / f"orderflow_{day_tag}.csv"
                print(f"DAILY RUN START day={day_tag} source={source.path}")
                if source.container == "zip":
                    convert_stats = convert_databento_mbp10_to_orderflow_csv(
                        input_path=str(input_path),
                        output_csv=str(day_csv),
                        append=False,
                        include_depth_json=True,
                        include=[source.path],
                        exclude=[],
                        max_files=1,
                        max_rows=None,
                    )
                else:
                    convert_stats = convert_databento_mbp10_to_orderflow_csv(
                        input_path=str(source.path),
                        output_csv=str(day_csv),
                        append=False,
                        include_depth_json=True,
                        include=[],
                        exclude=[],
                        max_files=1,
                        max_rows=None,
                    )
                print(
                    f"conversion rows_written={convert_stats.rows_written} "
                    f"skipped_rows={convert_stats.skipped_rows} files={convert_stats.files_processed}"
                )
                if convert_stats.rows_written <= 0:
                    raise RuntimeError(f"No rows written for day={day_tag}; skipping backtest.")
                run_backtest(str(day_csv), strategy_name=str(args.strategy), hold_bars=int(args.hold_bars), profile=str(args.profile))
                processed_days += 1
                completed += 1
                print(f"DAILY RUN DONE day={day_tag}")
                if (not args.keep_csv) and day_csv.exists():
                    try:
                        day_csv.unlink()
                    except OSError:
                        pass
                continue

            # CSV source path
            local_csv = Path(source.path)
            extracted_temp = False
            if source.container == "zip":
                local_csv = tmp_dir / "_zip_csv" / f"{idx:03d}_{Path(source.path).name}"
                _extract_zip_member(input_path, source.path, local_csv)
                extracted_temp = True

            csv_days: list[tuple[str, Path]] = []
            known_day = source.day_tag
            if known_day is not None:
                if _within_day_bounds(known_day, start_day, end_day):
                    csv_days = [(known_day, local_csv)]
            else:
                csv_days = _split_csv_by_day(
                    local_csv,
                    tmp_dir=tmp_dir,
                    source_label=source_label,
                    start_day=start_day,
                    end_day=end_day,
                )

            if not csv_days:
                raise RuntimeError(f"No day-splittable CSV rows found for source={source.path}")

            for day_tag, day_csv in csv_days:
                if max_days > 0 and processed_days >= max_days:
                    break
                print(f"DAILY RUN START day={day_tag} source={source.path}")
                run_backtest(str(day_csv), strategy_name=str(args.strategy), hold_bars=int(args.hold_bars), profile=str(args.profile))
                processed_days += 1
                completed += 1
                print(f"DAILY RUN DONE day={day_tag}")
                if (not args.keep_csv) and day_csv != local_csv and day_csv.exists():
                    try:
                        day_csv.unlink()
                    except OSError:
                        pass

            if (not args.keep_csv) and extracted_temp and local_csv.exists():
                try:
                    local_csv.unlink()
                except OSError:
                    pass

        except Exception as exc:  # pragma: no cover - runtime path
            failed += 1
            print(f"SOURCE FAILED source={source.path} error={exc}")
            if "Missing Databento DBN reader dependency" in str(exc):
                print(
                    "hint=DBN decoding dependency missing. "
                    "Either install `databento` package, or download Databento historical data as CSV "
                    "(split by day) and pass CSV input to this script."
                )
            if not args.continue_on_error:
                raise

    if (not args.keep_csv) and tmp_dir.exists():
        try:
            shutil.rmtree(tmp_dir)
        except OSError:
            pass

    print(f"\nDAILY BACKTEST COMPLETE completed={completed} failed={failed} days={processed_days}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
