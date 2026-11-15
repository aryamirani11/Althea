import numpy as np
import pandas as pd
import pytest

from src.data import (
    A1_COLS,
    A2_COLS,
    N_LOCI,
    alleles,
    allele_coding,
    case_pseudocontrol,
    dosage,
    is_founder,
    load_raw,
    trios,
    untransmitted,
)
from src.tdt import tdt


def _trio_frame(n_families: int, father, mother, child) -> pd.DataFrame:
    """Synthetic trios where only loc1 is genotyped (others missing)."""
    rows = []
    for f in range(n_families):
        pid = f"F{f}"
        for pid_, iid, fid, mid, crohn, (g1, g2) in [
            (pid, 1, 0, 0, 0, father),
            (pid, 2, 0, 0, 0, mother),
            (pid, 3, 1, 2, 2, child),
        ]:
            row = {"pid": pid_, "id": iid, "fid": fid, "mid": mid, "sex": 1, "crohn": crohn}
            row.update({c: 0 for c in A1_COLS + A2_COLS})
            row["loc1.a1"], row["loc1.a2"] = g1, g2
            rows.append(row)
    return pd.DataFrame(rows)


def test_dosage_is_order_invariant():
    df = load_raw()
    a1, a2 = alleles(df)
    coding = allele_coding(df)
    np.testing.assert_array_equal(dosage(a1, a2, coding), dosage(a2, a1, coding))


def test_untransmitted_allele():
    p1 = np.array([1, 1, 2, 0])
    p2 = np.array([3, 1, 4, 3])
    passed = np.array([3, 1, 0, 3])
    # het & known -> other allele; hom -> same; child missing / parent missing -> unknown
    np.testing.assert_array_equal(untransmitted(p1, p2, passed), [1, 1, 0, 0])


def test_tdt_counts_and_direction():
    # Het father (1/2) and het mother (1/2) both pass allele 2 to the child.
    # Minor allele must be 2 for the test to be meaningful, so make it rarer:
    # 6 families pass 2/2, the parents carry 1 more often via homozygous extras.
    df = _trio_frame(6, father=(1, 2), mother=(1, 2), child=(2, 2))
    extra = _trio_frame(3, father=(1, 1), mother=(1, 1), child=(1, 1))
    extra["pid"] = "X" + extra["pid"]
    df = pd.concat([df, extra], ignore_index=True)
    res = tdt(df).iloc[0]
    assert res["minor_allele"] == 2
    assert res["minor_transmitted"] == 12  # 6 fathers + 6 mothers
    assert res["major_transmitted"] == 0
    assert res["p_value"] < 1e-3


def test_children_are_phased_father_then_mother():
    df = load_raw()
    a1, a2 = alleles(df)
    for t in trios(df):
        called = (a1[t.child] != 0) & (a1[t.father] != 0) & (a2[t.father] != 0)
        from_father = (a1[t.child] == a1[t.father]) | (a1[t.child] == a2[t.father])
        assert from_father[called].all()


def test_naive_labels_are_parent_vs_child():
    df = load_raw()
    founders = is_founder(df)
    assert (df.loc[founders, "crohn"].isin([0, 2])).all()
    assert (df.loc[~founders, "crohn"] == 2).all()
    assert (df.loc[df["crohn"] == 0].index.isin(df.index[founders])).all()


def test_case_pseudocontrol_shape():
    X, y, groups = case_pseudocontrol(load_raw())
    assert X.shape == (258, N_LOCI)
    assert y.sum() == 129
    assert len(set(groups)) == 129
