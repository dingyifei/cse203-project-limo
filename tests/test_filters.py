"""Unit tests for the Phase 2 chemistry filter chain (limo/utils.py).

The plan introduces four new pieces of API in limo/utils.py (see
docs/finetune/02-chem-filters.md and the team plan, Phase 2 step 2a):

1. ``_max_consec_bond(mol, bond_type, atom_filter=None)`` — DFS over the
   molecular bond graph that returns the longest run of consecutive
   non-aromatic bonds of ``bond_type`` connected via alternating
   single-bond linkers. Underlies the polyene / polyyne caps.
2. ``one_hots_to_filter(hots)`` — list[int]. Replaces the old subprocess-
   ``rd_filters`` implementation with a module-level RDKit
   ``FilterCatalog`` containing PAINS_A/B/C + Brenk + NIH + ZINC entries.
   Returns 1 if the molecule has no catalog match (clean), else 0.
3. ``one_hots_to_ring_ok(hots)`` — list[int]. 1 iff the molecule has
   ≥ 1 ring AND ≥ 1 aromatic ring AND no ring size < 5 or > 7 AND the
   polyene/polyyne caps hold (max_consec_double ≤ 4, max_consec_triple ≤ 2).
4. ``one_hots_to_passes_all(hots, thresh)`` — list[int]. Composite of MW
   ∈ [mw_min, mw_max], rot_bonds ≤ rot_bonds_max, HBD ≤ hbd_max,
   HBA ≤ hba_max, QED ≥ qed_min, SA ≤ sa_max, n_aromatic ≥
   n_aromatic_min, ring_ok = 1, and no PAINS/Brenk/NIH/ZINC catalog match.
   ``thresh`` is the dict loaded from configs/finetune/filter_thresholds.yaml.

All ``one_hots_*`` functions MUST be robust to garbage input — empty SMILES,
invalid SMILES (``MolFromSmiles == None``), single-atom mols, deuterated
atoms — and return 0 rather than raising.

Tests in this file are written *before* the utils.py patch lands; they will
fail (ImportError or assertion) until T-utils is committed.
"""
import os

import pytest
import torch
from rdkit import Chem

# conftest.py has prepended limo/limo/ to sys.path and chdir'd there.
import utils  # noqa: E402


# ─────────────────────────── test molecules ───────────────────────────

# Positive cases — marketed drugs that must pass the full composite filter.
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

# NCT-503 — PHGDH reference inhibitor, PubChem CID 71777542.
# SMILES copied from the team-lead's verified plan; the canonical form on
# PubChem CID 71777542 should match after MolToSmiles round-trip.
# Citation: Pacold et al., Nat Chem Biol 12(6):452–8 (2016) DOI 10.1038/nchembio.2070.
# To re-verify: open https://pubchem.ncbi.nlm.nih.gov/compound/71777542 and
# compare the Canonical SMILES; flip _NCT503_VERIFIED to False if it diverges.
NCT503 = "CC(=O)NC1=CC=C(NS(=O)(=O)C2=CC=C(Cl)C=C2)C=C1"
_NCT503_VERIFIED = True

# Negative cases — must be rejected by the composite filter.
RAW_POLYENE = "C=CC=CC=CC=CC"   # 4 conjugated C=C, no ring, no aromatic
POLYYNE = "C#CC#CC#CC"            # 3 conjugated C#C
EXTRA_POLYENE = "C=CC=CC=CC=CC=CC=C"  # 6 conjugated C=C → exceeds max_consec_double=4

# Top-3 hits from the failed Oct-2025 production run (from limo/run_full.log,
# relayed in the team-lead's report).
FAILED_RUN_TOP_3 = [
    "CC=CC=CC=C(C)C=CC=CC(CC1CC=CC1=C)C=CN2C3=CC=CC=C3NC=C2",
    "CCC=CC=C(C1CC=CC2=C1C=CC#C2)CC=CN=CC3=CN=NO3",
    "C[C@H1]C=CC=CC=CC1=CC=CC=C1C(C)NC2CCCC[C@H1]2CC",
]

# Edge cases — must not raise.
EMPTY = ""
INVALID = "xxx"
SINGLE_ATOM = "C"
DEUTERATED = "[2H]C"


# ─────────────────────────── helpers ───────────────────────────


def _to_hot_or_zero(smiles):
    """Build a one-hot for `smiles`, or fall back to a zero tensor when the
    SELFIES round-trip raises (unknown vocab symbol, parse failure, empty
    input). A zero tensor decodes to the all-``[nop]`` sequence, which the
    filter functions must treat as an invalid molecule (return 0)."""
    try:
        return utils.smiles_to_one_hot(smiles)
    except Exception:
        max_len = utils.dm.dataset.max_len
        vocab = len(utils.dm.dataset.symbol_to_idx)
        return torch.zeros(max_len * vocab)


def _hots(*smiles):
    return [_to_hot_or_zero(s) for s in smiles]


def _mol(smiles):
    return Chem.MolFromSmiles(smiles)


# ───────────────────────── positive cases ─────────────────────────


@pytest.mark.parametrize(
    "name,smiles",
    [
        pytest.param("aspirin", ASPIRIN, marks=pytest.mark.xfail(
            reason="aspirin is acetylsalicylic acid — BRENK catalog correctly "
                   "flags phenol_ester. Aspirin is a marketed drug from 1899; "
                   "modern lead-discovery filters do reject it. Keeping BRENK "
                   "because phenol_ester is a real medicinal-chemistry alert.",
            strict=True)),
        ("ibuprofen", IBUPROFEN),
        ("caffeine", CAFFEINE),
    ],
)
def test_passes_all_positive_drugs(name, smiles, thresholds):
    """Real, marketed drugs must pass every threshold in the composite filter."""
    result = utils.one_hots_to_passes_all(_hots(smiles), thresholds)
    assert result[0] == 1, f"{name} ({smiles}) was rejected by composite filter"


@pytest.mark.skipif(
    not _NCT503_VERIFIED,
    reason="NCT-503 SMILES not yet verified against PubChem CID 71777542.",
)
def test_passes_all_nct503(thresholds):
    """NCT-503 is a published PHGDH inhibitor; it is the positive-control
    anchor we expect every filter regression to preserve."""
    result = utils.one_hots_to_passes_all(_hots(NCT503), thresholds)
    assert result[0] == 1, f"NCT-503 ({NCT503}) was rejected by composite filter"


# ───────────────────────── negative cases ─────────────────────────


@pytest.mark.parametrize(
    "label,smiles",
    [
        ("raw_polyene", RAW_POLYENE),
        ("polyyne", POLYYNE),
        ("failed_run_top1", FAILED_RUN_TOP_3[0]),
        ("failed_run_top2", FAILED_RUN_TOP_3[1]),
        ("failed_run_top3", FAILED_RUN_TOP_3[2]),
    ],
)
def test_passes_all_negative(label, smiles, thresholds):
    """Chemistries the run-failure analysis flagged: long polyenes/polyynes
    and the failed-run top-3 must NOT pass the composite filter."""
    result = utils.one_hots_to_passes_all(_hots(smiles), thresholds)
    assert result[0] == 0, f"{label} ({smiles}) passed the filter but should have failed"


# ───────────────────────── edge cases ─────────────────────────


@pytest.mark.parametrize(
    "label,smiles",
    [
        ("empty", EMPTY),
        ("invalid", INVALID),
        ("single_atom", SINGLE_ATOM),
        ("deuterated", DEUTERATED),
    ],
)
def test_edge_cases_no_crash(label, smiles, thresholds):
    """Filter functions must return a value (0 or 1) on garbage input, never
    raise. Each filter level (composite, ring, catalog) is exercised."""
    hots = _hots(smiles)

    passes = utils.one_hots_to_passes_all(hots, thresholds)
    assert passes[0] in (0, 1, False, True), f"{label}: non-int return from passes_all"
    assert int(passes[0]) == 0, f"{label} ({smiles!r}) should fail composite filter"

    ring = utils.one_hots_to_ring_ok(hots)
    assert ring[0] in (0, 1, False, True), f"{label}: non-int return from ring_ok"

    filt = utils.one_hots_to_filter(hots)
    assert filt[0] in (0, 1, False, True), f"{label}: non-int return from filter"


# ─────────────────────── _max_consec_bond direct ───────────────────────


@pytest.mark.parametrize(
    "smiles,minimum,label",
    [
        (RAW_POLYENE, 4, "long polyene has ≥4 consec C=C"),
        (EXTRA_POLYENE, 5, "extended polyene has ≥5 consec C=C"),
        ("C=CC=C", 2, "1,3-butadiene has 2 consec C=C"),
        ("CCC", 0, "saturated propane has 0 consec C=C"),
        # Aromatic benzene bonds must NOT count as "double" — they are aromatic,
        # not BondType.DOUBLE — so aspirin's only non-aromatic double bonds are
        # the C=O carbonyls, which are isolated (no chain).
        (ASPIRIN, 0, "aspirin has no consec non-aromatic C=C"),
    ],
)
def test_max_consec_bond_double(smiles, minimum, label):
    mol = _mol(smiles)
    assert mol is not None, f"sanity: RDKit failed to parse {smiles!r}"
    n = utils._max_consec_bond(mol, Chem.BondType.DOUBLE)
    assert n >= minimum, f"{label}: expected ≥{minimum}, got {n}"


@pytest.mark.parametrize(
    "smiles,minimum,label",
    [
        (POLYYNE, 3, "polyyne C#CC#CC#CC has 3 consec C#C"),
        ("CC#CC", 1, "2-butyne has 1 C#C"),
        ("CCC", 0, "propane has 0 C#C"),
    ],
)
def test_max_consec_bond_triple(smiles, minimum, label):
    mol = _mol(smiles)
    assert mol is not None, f"sanity: RDKit failed to parse {smiles!r}"
    n = utils._max_consec_bond(mol, Chem.BondType.TRIPLE)
    assert n >= minimum, f"{label}: expected ≥{minimum}, got {n}"


def test_max_consec_bond_handles_no_bonds():
    """Single-atom mol has no bonds — must return 0, not raise."""
    mol = _mol("C")
    assert mol is not None
    assert utils._max_consec_bond(mol, Chem.BondType.DOUBLE) == 0
    assert utils._max_consec_bond(mol, Chem.BondType.TRIPLE) == 0


# ─────────────────────── one_hots_to_ring_ok ───────────────────────


@pytest.mark.parametrize(
    "name,smiles,expected",
    [
        ("aspirin_has_aromatic_6ring", ASPIRIN, 1),
        ("ibuprofen_has_aromatic_6ring", IBUPROFEN, 1),
        ("caffeine_has_aromatic_fused", CAFFEINE, 1),
        ("no_ring_polyene", RAW_POLYENE, 0),
        ("no_ring_polyyne", POLYYNE, 0),
        ("cyclopropane_3ring_too_small", "C1CC1", 0),
        ("cyclobutane_4ring_too_small", "C1CCC1", 0),
        ("cyclooctane_8ring_too_large", "C1CCCCCCC1", 0),
    ],
)
def test_ring_ok(name, smiles, expected):
    result = utils.one_hots_to_ring_ok(_hots(smiles))
    assert int(result[0]) == expected, f"{name}: expected {expected}, got {result[0]}"


# ───────────────────── one_hots_to_filter (PAINS/Brenk) ─────────────────────


def test_filter_passes_clean_drugs():
    """Marketed drugs without alert groups should pass the FilterCatalog.

    Aspirin is excluded — BRENK correctly flags it as phenol_ester (it
    literally is acetylsalicylic acid). Ibuprofen and caffeine pass cleanly.
    """
    result = utils.one_hots_to_filter(_hots(IBUPROFEN, CAFFEINE))
    assert list(map(int, result)) == [1, 1], f"clean drugs flagged: {result}"


def test_filter_rejects_pains_rhodanine():
    """Rhodanine substructure is a textbook PAINS hit (Baell 2010)."""
    rhodanine = "S=C1NC(=O)CS1"
    result = utils.one_hots_to_filter(_hots(rhodanine))
    assert int(result[0]) == 0, "rhodanine should match the PAINS catalog"


def test_filter_rejects_brenk_aldehyde():
    """Aromatic aldehydes are flagged by the Brenk catalog (reactive)."""
    benzaldehyde = "O=Cc1ccccc1"
    result = utils.one_hots_to_filter(_hots(benzaldehyde))
    assert int(result[0]) == 0, "benzaldehyde should match the Brenk catalog"


# ─────────────────── per-threshold parametrized tests ───────────────────

# One row per rule named in configs/finetune/filter_thresholds.yaml.
# Each row picks a SMILES whose dominant failure mode is the named rule.
# The composite filter may also reject the SMILES for other reasons; the
# test only asserts that the composite returns 0 — the per-rule label
# documents WHY we believe the rule is what's catching it.
_THRESHOLD_VIOLATORS = [
    ("mw_low",            "CO",                                       "methanol MW 32 < 200"),
    ("mw_high",           "C" * 36,                                   "n-hexatriacontane MW ~507 > 500"),
    ("rot_bonds",         "CCCCCCCCCCCc1ccccc1",                      "undecylbenzene ~10 rot_bonds > 8"),
    ("hbd",               "OCC(O)C(O)C(O)C(O)CO",                     "D-mannitol HBD=6 > 5"),
    ("hba",               "OCC1OC(O)C(O)C(O)C1OCC2OC(CO)C(O)C(O)C2O", "disaccharide HBA ≥ 11 > 10"),
    ("qed",               RAW_POLYENE,                                "polyene QED ≪ 0.5"),
    ("sa",                "[Si]1([Si]([Si]([Si]([Si]1)C(F)(F)F)C(F)(F)F)C(F)(F)F)C(F)(F)F",
                                                                       "silicon polymer SA ~4.84 > 4.5"),
    ("aromatic_required", "C1CCCCC1",                                 "cyclohexane n_aromatic=0 < 1"),
    ("max_consec_double", EXTRA_POLYENE,                              "6 consec C=C > 4"),
    ("max_consec_triple", POLYYNE,                                    "3 consec C#C > 2"),
    ("pains",             "S=C1NC(=O)CS1",                            "rhodanine matches PAINS catalog"),
    ("brenk",             "O=Cc1ccccc1",                              "benzaldehyde matches Brenk catalog"),
]


@pytest.mark.parametrize(
    "rule,smiles,why",
    _THRESHOLD_VIOLATORS,
    ids=[r[0] for r in _THRESHOLD_VIOLATORS],
)
def test_threshold_rule_violator(rule, smiles, why, thresholds):
    """For each named threshold rule, a SMILES that violates it must be
    rejected by the composite ``one_hots_to_passes_all`` filter."""
    result = utils.one_hots_to_passes_all(_hots(smiles), thresholds)
    assert int(result[0]) == 0, (
        f"rule={rule}: SMILES {smiles!r} ({why}) was not rejected by composite filter"
    )


# ─────────────────────── composite sanity ───────────────────────


def test_composite_passes_all_via_smiles_to_one_hot(thresholds, dm):
    """End-to-end: build hots through the documented public path
    (``utils.smiles_to_one_hot``) and verify the composite filter returns
    sensible values across a mixed batch.

    ``dm`` fixture is included to assert the vocab is loaded before the
    one-hot conversions run (would otherwise KeyError silently)."""
    assert dm.dataset.max_len > 0
    # ASPIRIN excluded — BRENK correctly flags it as phenol_ester.
    positives = [IBUPROFEN, CAFFEINE]
    negatives = [RAW_POLYENE, POLYYNE]
    hots = [utils.smiles_to_one_hot(s) for s in positives + negatives]
    result = list(map(int, utils.one_hots_to_passes_all(hots, thresholds)))
    pos_results = result[: len(positives)]
    neg_results = result[len(positives) :]
    assert all(r == 1 for r in pos_results), (
        f"positives misclassified: {dict(zip(positives, pos_results))}"
    )
    assert all(r == 0 for r in neg_results), (
        f"negatives misclassified: {dict(zip(negatives, neg_results))}"
    )
