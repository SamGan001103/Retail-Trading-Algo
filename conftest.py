from pathlib import Path
import shutil
import sys
from uuid import uuid4

import pytest

# Keep import bootstrap in one place for local pytest runs without installation.
SRC_PATH = str(Path(__file__).resolve().parent / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


@pytest.fixture
def tmp_path():
    """Provide a per-test writable directory inside the repo workspace.

    In restricted Windows environments, pytest's built-in tmp_path fixture may
    fail due temp directory ACL handling. This fixture keeps tests deterministic
    by using a local workspace temp root.
    """
    root = Path(__file__).resolve().parent / "tmp_testdata"
    root.mkdir(parents=True, exist_ok=True)
    test_dir = root / f"case_{uuid4().hex[:10]}"
    test_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield test_dir
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
