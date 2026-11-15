"""Loading the Crohn's trio data and turning it into usable genotypes.

The dataset is family-based: 129 parent-parent-child trios (387 people)
genotyped at 103 SNPs in the IBD5 region on chromosome 5q31.

Two facts about the file matter for everything downstream:

* ``crohn`` is a pedigree phenotype code: 2 = affected, 0 = unknown.
  Every "0" is a parent whose disease status was never recorded,
  so they are *not* healthy controls.
* The two allele columns per SNP are stored differently for parents and
  children. Children are phased (``a1`` = from father, ``a2`` = from
  mother); parents are stored in a fixed sorted order. Genotypes are
  therefore always encoded order-free (minor-allele counts) before modelling.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "crohn.csv"
N_LOCI = 103
MISSING = 0
AFFECTED = 2
ID_COLS = ["pid", "id", "fid", "mid"]

A1_COLS = [f"loc{i}.a1" for i in range(1, N_LOCI + 1)]
A2_COLS = [f"loc{i}.a2" for i in range(1, N_LOCI + 1)]
SNP_NAMES = [f"loc{i}" for i in range(1, N_LOCI + 1)]


def load_raw(path: str | pathlib.Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV exactly as distributed (one row per person)."""
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def alleles(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the two allele matrices, each shaped (people, SNPs)."""
    return df[A1_COLS].to_numpy(), df[A2_COLS].to_numpy()


def is_founder(df: pd.DataFrame) -> np.ndarray:
    """True for people with no parents in the file (the trio parents)."""
    return ((df["fid"] == 0) & (df["mid"] == 0)).to_numpy()


@dataclass(frozen=True)
class AlleleCoding:
    """Per-SNP major/minor allele, estimated from founders only."""

    major: np.ndarray
    minor: np.ndarray
    minor_freq: np.ndarray


def allele_coding(df: pd.DataFrame) -> AlleleCoding:
    """Pick the minor allele at each SNP from the parents' genotypes.

    Using founders keeps the reference independent of affection status
    (children are all affected and would bias the frequencies).
    """
    a1, a2 = alleles(df)
    f = is_founder(df)
    major, minor, freq = [], [], []
    for j in range(N_LOCI):
        pool = np.concatenate([a1[f, j], a2[f, j]])
        pool = pool[pool != MISSING]
        values, counts = np.unique(pool, return_counts=True)
        order = np.argsort(-counts, kind="stable")
        values, counts = values[order], counts[order]
        if len(values) == 0:  # SNP never called in parents
            major.append(MISSING), minor.append(MISSING), freq.append(0.0)
            continue
        major.append(values[0])
        minor.append(values[1] if len(values) > 1 else values[0])
        freq.append(counts[1] / counts.sum() if len(values) > 1 else 0.0)
    return AlleleCoding(np.array(major), np.array(minor), np.array(freq))


def dosage(a1: np.ndarray, a2: np.ndarray, coding: AlleleCoding) -> np.ndarray:
    """Count minor alleles per SNP (0, 1 or 2); NaN where a call is missing.

    This encoding is order-invariant: swapping a1 and a2 gives the same
    value, which removes the storage artefact entirely.
    """
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    out = (a1 == coding.minor).astype(float) + (a2 == coding.minor).astype(float)
    out[(a1 == MISSING) | (a2 == MISSING)] = np.nan
    return out


@dataclass(frozen=True)
class Trio:
    family: str
    child: int
    father: int
    mother: int


def trios(df: pd.DataFrame) -> list[Trio]:
    """Row indices of each (child, father, mother) trio."""
    pos = {(p, i): k for k, (p, i) in enumerate(zip(df["pid"], df["id"]))}
    out = []
    for k, row in enumerate(df.itertuples(index=False)):
        if row.fid == 0 and row.mid == 0:
            continue
        out.append(Trio(row.pid, k, pos[(row.pid, row.fid)], pos[(row.pid, row.mid)]))
    return out


def untransmitted(parent_a1, parent_a2, transmitted) -> np.ndarray:
    """The allele a parent did *not* pass on, per SNP (0 if unknowable).

    Known when the parent is homozygous, or heterozygous with the
    transmitted allele observed.
    """
    p1 = np.asarray(parent_a1)
    p2 = np.asarray(parent_a2)
    t = np.asarray(transmitted)
    out = np.full_like(t, MISSING)
    hom = (p1 == p2) & (p1 != MISSING)
    out[hom] = p1[hom]
    het = (p1 != p2) & (p1 != MISSING) & (p2 != MISSING) & (t != MISSING)
    out[het & (t == p1)] = p2[het & (t == p1)]
    out[het & (t == p2)] = p1[het & (t == p2)]
    return out


def case_pseudocontrol(df: pd.DataFrame, coding: AlleleCoding | None = None):
    """Build a matched case / pseudo-control dataset from the trios.

    Each affected child is a case. Its pseudo-control is the genotype made
    of the two alleles the parents did *not* transmit — a "genetic sibling"
    that shares the same parents and ancestry. Any difference between the
    two groups can only come from which alleles were inherited, so the
    comparison is immune to population stratification.

    Returns (X, y, groups): minor-allele dosages, 1/0 labels, family ids.
    """
    coding = coding or allele_coding(df)
    a1, a2 = alleles(df)
    rows, labels, groups = [], [], []
    for t in trios(df):
        pat, mat = a1[t.child], a2[t.child]  # children are phased: a1 paternal
        u_pat = untransmitted(a1[t.father], a2[t.father], pat)
        u_mat = untransmitted(a1[t.mother], a2[t.mother], mat)
        rows += [dosage(pat, mat, coding), dosage(u_pat, u_mat, coding)]
        labels += [1, 0]
        groups += [t.family, t.family]
    X = pd.DataFrame(np.vstack(rows), columns=SNP_NAMES)
    return X, np.array(labels), np.array(groups)
