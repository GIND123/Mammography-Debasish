"""Print the key numbers of one or more exported run summaries."""
import json
import os
import sys

raw = sys.argv[1]
for name in sys.argv[2:]:
    p = os.path.join(raw, name, "summary.json")
    if not os.path.exists(p):
        print(f"== {name}: no summary")
        continue
    j = json.load(open(p))
    lc = j.get("local_cv", {})
    print(f"== {name}")
    if lc:
        ci = lc.get("ci95", {})
        print("  local CV:", "  ".join(f"{k}={lc[k]:.3f}" + (f" [{ci[k][0]:.3f},{ci[k][1]:.3f}]" if k in ci else "")
                                      for k in ("accuracy", "balanced_accuracy", "macro_f1", "qwk", "adjacent_accuracy")
                                      if k in lc))
        print(f"  ece={lc.get('ece', 0):.3f} nll={lc.get('nll', 0):.3f} auc_ovr={lc.get('macro_auc_ovr', float('nan')):.3f}")
        print("  per-class (P/R/n):", {k: (round(v["precision"], 2), round(v["recall"], 2), v["support"])
                                       for k, v in lc.get("per_class", {}).items()})
        print("  confusion:", lc.get("confusion"))
        dv = lc.get("dense_vs_nondense", {})
        print("  dense vs non-dense:", {k: round(dv[k], 3) for k in ("auc", "accuracy", "sensitivity", "specificity") if k in dv})
        for v in ("cc_only", "mlo_only"):
            if v in lc:
                print(f"  {v}:", {k: round(x, 3) for k, x in lc[v].items()})
        if "per_fold" in lc:
            print("  per fold qwk:", [round(f["qwk"], 3) for f in lc["per_fold"].values()])
    for ev in ("rsna_test", "cbis_test", "birads_ext"):
        e = j.get(ev)
        if not e:
            continue
        d, m = e.get("density", {}), e.get("malignant", {})
        s = f"  {ev} (n={e.get('n_cases')}):"
        if d:
            s += " density " + " ".join(f"{k}={d[k]:.3f}" for k in ("accuracy", "qwk", "macro_f1"))
        if m:
            s += " | suspicion " + " ".join(f"{k}={m[k]:.3f}" for k in ("auc", "average_precision", "sensitivity", "specificity") if k in m)
            if "ci95" in m:
                s += f" auc95={m['ci95'].get('auc')}"
        print(s)
