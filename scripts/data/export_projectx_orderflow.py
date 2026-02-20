from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from trading_algo.config import env_bool, env_float
from trading_algo.data_export import capture_projectx_orderflow_csv


def _default_output(symbol: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.strip().upper() or "MNQ"
    return str(Path("data") / f"{safe_symbol}_orderflow_{stamp}.csv")


def parse_args() -> argparse.Namespace:
    default_symbol = (os.getenv("SYMBOL") or "MNQ").strip().upper()
    default_live = env_bool("LIVE", False)
    parser = argparse.ArgumentParser(
        description=(
            "Capture ProjectX realtime quote/trade/depth snapshots and write "
            "orderflow CSV compatible with ny_structure backtest."
        )
    )
    parser.add_argument("--symbol", default=default_symbol, help="Contract search text (e.g., MNQ).")
    parser.add_argument("--live", action="store_true", default=default_live, help="Resolve live contract instead of demo.")
    parser.add_argument(
        "--output",
        default=_default_output(default_symbol),
        help="Output CSV path. Default: data/<SYMBOL>_orderflow_<UTCSTAMP>.csv",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=env_float("EXPORT_CAPTURE_DURATION_SEC", 600.0),
        help="Capture duration in seconds.",
    )
    parser.add_argument(
        "--poll-sec",
        type=float,
        default=env_float("EXPORT_CAPTURE_POLL_SEC", 0.05),
        help="Polling interval for stream snapshots in seconds.",
    )
    parser.add_argument(
        "--heartbeat-sec",
        type=float,
        default=env_float("EXPORT_CAPTURE_HEARTBEAT_SEC", 5.0),
        help="Progress print interval in seconds.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row cap; 0 means unlimited.")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV and continue seq.")
    parser.add_argument(
        "--require-depth",
        action="store_true",
        default=env_bool("EXPORT_CAPTURE_REQUIRE_DEPTH", True),
        help="Skip rows without usable depth (recommended for ny_structure).",
    )
    parser.add_argument(
        "--allow-no-depth",
        action="store_false",
        dest="require_depth",
        help="Allow rows even when depth is missing.",
    )
    parser.add_argument(
        "--include-depth-json",
        action="store_true",
        default=True,
        help="Persist depth_bids/depth_asks JSON columns.",
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
    max_rows = None if int(args.max_rows) <= 0 else int(args.max_rows)

    stats = capture_projectx_orderflow_csv(
        output_csv=str(args.output),
        symbol=str(args.symbol).strip().upper(),
        live=bool(args.live),
        duration_sec=max(0.0, float(args.duration_sec)),
        poll_sec=max(0.01, float(args.poll_sec)),
        max_rows=max_rows,
        append=bool(args.append),
        include_depth_json=bool(args.include_depth_json),
        require_depth=bool(args.require_depth),
        heartbeat_sec=max(0.5, float(args.heartbeat_sec)),
    )

    print("EXPORT COMPLETE")
    print(f"symbol={str(args.symbol).strip().upper()} live={bool(args.live)} contract_id={stats.contract_id}")
    print(f"output_csv={stats.output_csv}")
    print(f"rows_written={stats.rows_written}")
    print(
        f"skipped_no_depth={stats.skipped_no_depth} skipped_no_price={stats.skipped_no_price} "
        f"duplicates_skipped={stats.duplicates_skipped}"
    )
    print(f"start_utc={stats.start_utc}")
    print(f"end_utc={stats.end_utc}")
    print(f"duration_sec={stats.duration_sec:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
