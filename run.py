"""Run the full Althea analysis: association test, leakage audit, honest model.

    python run.py

Writes results/ (CSV + JSON) and figures/ (PNG). Takes about a minute.
"""

from __future__ import annotations

import json
import pathlib

from src import plots
from src.audit import run_audit
from src.data import case_pseudocontrol, is_founder, load_raw
from src.model import evaluate
from src.tdt import bonferroni_threshold, tdt

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = load_raw()
    print(f"Loaded {len(df)} people in {df['pid'].nunique()} families, 103 SNPs\n")

    # 1. Family-based association test
    assoc = tdt(df)
    assoc.to_csv(RESULTS / "tdt_results.csv", index=False)
    hits = assoc[assoc["significant_bonferroni"]].sort_values("p_value")
    print(f"[1] TDT: {len(hits)} SNPs pass Bonferroni (p < {bonferroni_threshold():.1e})")
    print(hits[["snp", "minor_transmitted", "major_transmitted", "transmission_or", "p_value"]]
          .to_string(index=False), "\n")

    # 2. Leakage audit of a naive setup
    audit, flip = run_audit(df)
    print("[2] Leakage audit (ROC-AUC, repeated 5-fold CV)")
    for name, m in audit["auc"].items():
        print(f"    {name:<45} {m['auc_mean']:.3f} ± {m['auc_sd']:.3f}")
    print()

    # 3. Honest predictive baseline
    X, y, groups = case_pseudocontrol(df)
    honest = evaluate(X, y, groups)
    print(f"[3] Case vs pseudo-control, family-grouped CV: "
          f"AUC {honest['auc_mean']:.3f} ± {honest['auc_sd']:.3f}\n")

    metrics = {"audit": audit, "case_pseudocontrol_model": {k: v for k, v in honest.items()
                                                            if not k.startswith("_")}}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plots.tdt_plot(assoc, bonferroni_threshold(), FIGURES / "tdt_association.png")
    plots.leakage_plot(audit, honest, FIGURES / "leakage_audit.png")
    plots.flip_plot(flip, is_founder(df), FIGURES / "allele_order.png")
    plots.roc_plot(honest, FIGURES / "roc_pseudocontrol.png")
    print(f"Saved results to {RESULTS.name}/ and figures to {FIGURES.name}/")


if __name__ == "__main__":
    main()
