# Althea — Crohn's disease genetics from family trios

Althea analyses a family-based genetic dataset for Crohn's disease in two parts:

1. **Association.** A transmission disequilibrium test (TDT) identifies the SNPs whose alleles are passed to affected children more often than chance. **10 of 103 SNPs** reach significance, clustered in the IBD5 risk region.
2. **Prediction.** A case vs pseudo-control model, validated with family-grouped nested cross-validation, measures how much risk information these SNPs carry: **ROC-AUC 0.65**.

![TDT association](figures/tdt_association.png)

---

## The data

`data/crohn.csv` is a public Crohn's disease dataset from Daly et al. (2001), as distributed with the R package `gap`. It has 387 people in **129 parent–parent–child trios**, genotyped at **103 SNPs** across the IBD5 region on chromosome 5q31.

| Who | Rows | `crohn` value | What it means |
|---|---|---|---|
| Affected children | 129 | 2 | Crohn's disease |
| Parents, affected | 15 | 2 | Crohn's disease |
| Parents, status not recorded | 243 | 0 | Unknown |

Each SNP is stored as two allele columns (`locN.a1`, `locN.a2`), with alleles coded 1–4 and 0 for missing. Children's alleles are phased (`a1` from the father, `a2` from the mother), which is what makes transmission analysis possible.

## Design choices

- **No healthy controls, so a family-based design.** Label `0` means "status unknown", not "healthy", so a standard case vs control comparison isn't valid. Both analyses compare each affected child with their own parents instead, which also rules out confounding from ancestry (population stratification).
- **Order-free genotype encoding.** Parents' and children's alleles are stored in different orders, so every model input is a minor-allele count (0, 1 or 2) rather than the raw allele columns.
- **Family-grouped validation.** Cases and pseudo-controls from the same family always land in the same fold, so the model is never tested on a family it trained on.

## 1. Association: the transmission disequilibrium test

A heterozygous parent passes each of their two alleles on 50% of the time by chance. If one allele goes to affected children significantly more often than that, it is associated with the disease.

**10 of 103 SNPs pass Bonferroni correction** (p < 0.05/103 ≈ 4.9 × 10⁻⁴). The strongest:

| SNP | Minor allele transmitted | Not transmitted | Transmission OR | p |
|---|---|---|---|---|
| loc28 | 88 | 38 | 2.32 | 8.4 × 10⁻⁶ |
| loc34 | 80 | 34 | 2.35 | 1.6 × 10⁻⁵ |
| loc26 | 90 | 41 | 2.20 | 1.9 × 10⁻⁵ |
| loc74 | 82 | 36 | 2.28 | 2.3 × 10⁻⁵ |
| loc27 | 82 | 37 | 2.22 | 3.7 × 10⁻⁵ |

The cluster of hits at loc26–loc39 and loc74–loc79 is consistent with the IBD5 risk haplotype reported for this region. Full results are in [`results/tdt_results.csv`](results/tdt_results.csv).

## 2. Prediction: case vs pseudo-control

Each affected child is paired with a **pseudo-control**: the genotype made from the two alleles their parents did *not* pass on. It is like a genetic sibling with the same parents and ancestry who didn't inherit the child's alleles.

- 129 cases vs 129 pseudo-controls, using minor-allele counts (order-free)
- L2-regularised logistic regression, with C tuned in a nested inner loop
- Outer loop: 5-fold `StratifiedGroupKFold` repeated 10 times, **keeping both members of a family in the same fold**

<img src="figures/roc_pseudocontrol.png" width="380" alt="ROC curve">

**ROC-AUC 0.65 ± 0.06.** That's a real but modest signal. 103 SNPs in one region carry some information about inheritance of risk, and they are nowhere near a diagnostic test. This matches what is known about complex diseases like Crohn's, where many loci each contribute a small effect.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py        # ~1 minute; writes results/ and figures/
pytest -q            # 6 tests
```

## Project structure

```
├── data/crohn.csv              # raw trio data
├── src/
│   ├── data.py                 # loading, trio structure, order-free encoding, pseudo-controls
│   ├── tdt.py                  # transmission disequilibrium test
│   ├── model.py                # case vs pseudo-control model, nested grouped CV
│   └── plots.py                # figures
├── tests/test_althea.py        # encoding, TDT, phasing and data checks
├── results/                    # tdt_results.csv, metrics.json
├── figures/                    # PNGs used in this README
└── run.py                      # runs everything
```

## Limitations

- **Small sample.** 129 trios gives limited power, so the TDT will miss weaker effects and the AUC estimate is noisy (fold range roughly 0.54–0.77).
- **No physical positions.** SNPs are plotted in marker order, not base-pair position, because the dataset doesn't include coordinates.
- **Correlated SNPs.** SNPs in this region are in strong linkage disequilibrium, so Bonferroni is conservative and the 10 hits are not 10 independent signals. A haplotype-based analysis would be the natural next step.
- **Missing genotypes** (about 13% of pseudo-control calls) are mean-imputed inside each training fold.
- **No external replication.** All results come from this single cohort.

## References

- Spielman RS, McGinnis RE, Ewens WJ (1993). Transmission test for linkage disequilibrium: the insulin gene region and insulin-dependent diabetes mellitus. *Am J Hum Genet* 52:506–516.
- Daly MJ, Rioux JD, Schaffner SF, Hudson TJ, Lander ES (2001). High-resolution haplotype structure in the human genome. *Nature Genetics* 29:229–232.
- Rioux JD et al. (2001). Genetic variation in the 5q31 cytokine gene cluster confers susceptibility to Crohn disease. *Nature Genetics* 29:223–228.

## License

MIT
