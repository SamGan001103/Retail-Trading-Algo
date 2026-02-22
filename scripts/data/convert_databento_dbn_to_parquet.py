from __future__ import annotations

import argparse
import fnmatch
import shutil
import tempfile
import zipfile
from pathlib import Path


def _is_dbn_filename(name: str) -> bool:
    lowered = name.strip().lower()
    return lowered.endswith(".dbn") or lowered.endswith(".dbn.zst")


def _load_dbn_store_class() -> type:
    modules = []
    try:
        import databento_dbn as dbn  # type: ignore[import-not-found]

        modules.append(dbn)
    except Exception:
        pass
    try:
        import databento as db  # type: ignore[import-not-found]

        modules.append(db)
    except Exception:
        pass
    for module in modules:
        store_cls = getattr(module, "DBNStore", None)
        if store_cls is not None:
            return store_cls
    raise RuntimeError(
        "Missing Databento DBN reader dependency. Install with `py -3.11 -m pip install databento`."
    )


def _matches(name: str, include: list[str], exclude: list[str]) -> bool:
    lowered = name.lower()
    if include and not any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in include):
        return False
    if exclude and any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in exclude):
        return False
    return True


def _strip_dbn_suffix(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(".dbn.zst"):
        return name[: -len(".dbn.zst")]
    if lowered.endswith(".dbn"):
        return name[: -len(".dbn")]
    return Path(name).stem


def _discover_sources(input_path: Path, include: list[str], exclude: list[str], max_files: int) -> tuple[str, list[str]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    discovered: list[str] = []
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path, "r") as archive:
            for info in sorted(archive.infolist(), key=lambda x: x.filename):
                if info.is_dir():
                    continue
                if not _is_dbn_filename(info.filename):
                    continue
                if not _matches(info.filename, include, exclude):
                    continue
                discovered.append(info.filename)
        if max_files > 0:
            discovered = discovered[:max_files]
        return "zip", discovered

    if input_path.is_dir():
        for path in sorted(input_path.rglob("*")):
            if not path.is_file():
                continue
            if not _is_dbn_filename(path.name):
                continue
            if not _matches(path.name, include, exclude):
                continue
            discovered.append(str(path))
        if max_files > 0:
            discovered = discovered[:max_files]
        return "fs", discovered

    if input_path.is_file() and _is_dbn_filename(input_path.name):
        if _matches(input_path.name, include, exclude):
            discovered.append(str(input_path))
        return "fs", discovered

    raise RuntimeError(f"Unsupported input path for DBN conversion: {input_path}")


def _resolve_targets(sources: list[str], output_path: Path) -> dict[str, Path]:
    if not sources:
        return {}
    if len(sources) == 1 and output_path.suffix.lower() == ".parquet":
        return {sources[0]: output_path}

    out_dir = output_path if output_path.suffix.lower() != ".parquet" else output_path.parent / output_path.stem
    targets: dict[str, Path] = {}
    for source in sources:
        safe = _strip_dbn_suffix(Path(source).name).replace("/", "_").replace("\\", "_")
        targets[source] = out_dir / f"{safe}.parquet"
    return targets


def _convert_file(*, dbn_file: Path, out_file: Path, dbn_store_class: type, overwrite: bool) -> bool:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists() and (not overwrite):
        print(f"skip reason=exists output={out_file}")
        return False

    store = dbn_store_class.from_file(str(dbn_file))
    try:
        store.to_parquet(str(out_file))
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    print(f"wrote output={out_file}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Databento DBN (.dbn/.dbn.zst or .zip bundle) to Parquet once, "
            "then reuse Parquet for fast streaming backtests."
        )
    )
    parser.add_argument("--input", required=True, help="Input .dbn/.dbn.zst, .zip, or directory of DBN files.")
    parser.add_argument(
        "--output",
        default=str(Path("data") / "parquet" / "mbp10.parquet"),
        help=(
            "Output Parquet path. For one source, provide a .parquet file. "
            "For multiple sources, provide a directory."
        ),
    )
    parser.add_argument("--include", action="append", default=[], help="Optional include glob for source filenames.")
    parser.add_argument("--exclude", action="append", default=[], help="Optional exclude glob for source filenames.")
    parser.add_argument("--max-files", type=int, default=0, help="Optional max source files to process; 0 means all.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output parquet files.")
    parser.add_argument("--list-only", action="store_true", help="List selected sources and output targets, then exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    include = [x.strip() for x in list(args.include or []) if x.strip() != ""]
    exclude = [x.strip() for x in list(args.exclude or []) if x.strip() != ""]
    max_files = max(0, int(args.max_files))

    source_type, sources = _discover_sources(input_path, include, exclude, max_files)
    if not sources:
        raise RuntimeError("No DBN sources selected. Check input path and include/exclude filters.")
    targets = _resolve_targets(sources, output_path)

    print(f"dbn_to_parquet_plan input={input_path} sources={len(sources)} source_type={source_type}")
    for idx, source in enumerate(sources, start=1):
        print(f"{idx:03d}. source={source} output={targets[source]}")
    if args.list_only:
        return 0

    dbn_store_class = _load_dbn_store_class()
    converted = 0
    skipped = 0

    if source_type == "zip":
        with zipfile.ZipFile(input_path, "r") as archive, tempfile.TemporaryDirectory(prefix="dbn_parquet_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            for source in sources:
                temp_file = tmp_root / Path(source).name
                with archive.open(source, "r") as src, temp_file.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                wrote = _convert_file(
                    dbn_file=temp_file,
                    out_file=targets[source],
                    dbn_store_class=dbn_store_class,
                    overwrite=bool(args.overwrite),
                )
                converted += 1 if wrote else 0
                skipped += 0 if wrote else 1
                try:
                    temp_file.unlink()
                except OSError:
                    pass
    else:
        for source in sources:
            wrote = _convert_file(
                dbn_file=Path(source),
                out_file=targets[source],
                dbn_store_class=dbn_store_class,
                overwrite=bool(args.overwrite),
            )
            converted += 1 if wrote else 0
            skipped += 0 if wrote else 1

    print(f"dbn_to_parquet_complete converted={converted} skipped={skipped} total={len(sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
