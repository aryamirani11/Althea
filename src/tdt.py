"""Transmission Disequilibrium Test (Spielman et al., 1993).

For every parent who is heterozygous at a SNP, we know which of their two
alleles went to the affected child. Under no association each allele is
passed on half the time. If the minor allele is transmitted more (or less)
often than 50%, it is associated with the disease.

    b = minor allele transmitted, major not
    c = major allele transmitted, minor not
    chi2 = (b - c)^2 / (b + c)        (1 degree of freedom)
    transmission odds ratio = b / c
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .data import MISSING, N_LOCI, SNP_NAMES, AlleleCoding, alleles, allele_coding, trios


def tdt(df: pd.DataFrame, coding: AlleleCoding | None = None, alpha: float = 0.05) -> pd.DataFrame:
    """Run the TDT at every SNP. Returns one row per SNP, sorted by position."""
    coding = coding or allele_coding(df)
    a1, a2 = alleles(df)
    b = np.zeros(N_LOCI, dtype=int)
    c = np.zeros(N_LOCI, dtype=int)

    for t in trios(df):
        # Children are phased: a1 came from the father, a2 from the mother.
        for parent, passed in ((t.father, a1[t.child]), (t.mother, a2[t.child])):
            p1, p2 = a1[parent], a2[parent]
            informative = (p1 != p2) & (p1 != MISSING) & (p2 != MISSING) & (passed != MISSING)
            b += informative & (passed == coding.minor)
            c += informative & (passed == coding.major)

    n = b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.where(n > 0, (b - c) ** 2 / n, np.nan)
        odds = np.where(c > 0, b / c, np.nan)
    p = chi2.sf(stat, df=1)
    threshold = alpha / N_LOCI

    return pd.DataFrame(
        {
            "snp": SNP_NAMES,
            "position": np.arange(1, N_LOCI + 1),
            "minor_allele": coding.minor,
            "minor_freq": coding.minor_freq.round(3),
            "minor_transmitted": b,
            "major_transmitted": c,
            "chi2": np.round(stat, 3),
            "p_value": p,
            "transmission_or": np.round(odds, 3),
            "significant_bonferroni": p < threshold,
        }
    )


def bonferroni_threshold(alpha: float = 0.05) -> float:
    return alpha / N_LOCI
