"""Shared pytest fixtures for the LIMO finetune test suite.

The limo/utils.py module uses a flat-script import pattern (`from utils import *`)
rather than a proper package, and it loads `dm.pkl` from the current working
directory at import time. To make these tests runnable from any cwd:

1. Prepend `limo/limo/` to sys.path so `import utils` resolves.
2. Chdir into `limo/limo/` so the module-level `pickle.load(open('dm.pkl'))`
   in utils.py finds the molecular data module.

Both have to happen BEFORE any test module imports `utils`, which is why this
file does it at module-import time (pytest imports conftest.py before
collection).
"""
import os
import sys

# --- repo paths ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
LIMO_DIR = os.path.join(REPO_ROOT, "limo")
CONFIG_DIR = os.path.join(REPO_ROOT, "configs", "finetune")

# Make `import utils` and friends work from inside limo/limo/.
if LIMO_DIR not in sys.path:
    sys.path.insert(0, LIMO_DIR)

# utils.py's tail does `pickle.load(open('dm.pkl', 'rb'))` — needs cwd=limo/.
# Chdir at conftest-import time so any subsequent `import utils` works.
os.chdir(LIMO_DIR)

import pytest  # noqa: E402  — must come after sys.path/chdir setup


@pytest.fixture(scope="session")
def dm():
    """The pickled MolDataModule (limo/dm.pkl), source of the SELFIES vocab.

    Exposed as a fixture so tests can hash it / inspect alphabet size / verify
    `max_len`. Not strictly required for the filter tests but documented here
    per the test-plan contract.

    Safety note: the actual `pickle.load(open('dm.pkl', 'rb'))` happens inside
    `limo/utils.py` at its module-level import. dm.pkl is a repo-committed
    artifact (sha256 tracked in docs/finetune/data/MANIFEST.csv) — trusted as
    part of the project, not loaded from untrusted external input. This fixture
    only re-exposes the already-loaded object; it does not itself unpickle.
    """
    import utils
    assert utils.dm is not None, "utils.dm failed to load; check limo/dm.pkl"
    return utils.dm


@pytest.fixture(scope="session")
def thresholds():
    """Load configs/finetune/filter_thresholds.yaml — the single source of truth
    for the composite filter thresholds. Passed to `one_hots_to_passes_all`."""
    import yaml
    cfg_path = os.path.join(CONFIG_DIR, "filter_thresholds.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)
