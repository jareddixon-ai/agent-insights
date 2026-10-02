"""Shared test setup: make src/ importable and build the fake data once."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """Run the generator in a temp folder so tests never touch your real data/ folder."""
    work = tmp_path_factory.mktemp("work")
    subprocess.run([sys.executable, str(ROOT / "src" / "generate_fake_data.py")],
                   cwd=work, check=True, capture_output=True)
    return work / "data" / "fake"
