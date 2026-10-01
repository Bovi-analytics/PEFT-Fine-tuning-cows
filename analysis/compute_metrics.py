# -*- coding: utf-8 -*-
"""Recomputes the reported test metrics from a confusion-matrix file of this repository.

Usage: python analysis/compute_metrics.py results/original_evaluation/confusion_matrices.json [--per-class]
Rows of each matrix are annotated behaviors, columns predictions; classes 0-5 are MmCows (cow) behaviors and 6-8
PlayBehavior (calf) behaviors. Wilson 95% intervals assume independent crops and therefore understate the uncertainty
(consecutive frames and simultaneous camera views of the same animal are correlated).
"""
import json, math, sys

def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def metrics(m):
    K = len(m); rows = [sum(r) for r in m]; cols = [sum(m[i][j] for i in range(K)) for j in range(K)]; n = sum(rows)
    diag = [m[i][i] for i in range(K)]
    P = [diag[j] / cols[j] if cols[j] else 0.0 for j in range(K)]
    R = [diag[i] / rows[i] for i in range(K)]
    F = [2 * P[i] * R[i] / (P[i] + R[i]) if P[i] + R[i] else 0.0 for i in range(K)]
    cow, calf = range(6), range(6, 9)
    return dict(n=n, acc=sum(diag) / n, ci=wilson(sum(diag), n), P=P, R=R, F1=F,
                weighted_f1=sum(F[i] * rows[i] for i in range(K)) / n, macro_f1=sum(F) / K, balanced_acc=sum(R) / K,
                acc_mmcows=sum(m[i][i] for i in cow) / sum(rows[i] for i in cow),
                acc_playbehavior=sum(m[i][i] for i in calf) / sum(rows[i] for i in calf),
                cow_to_calf=sum(m[i][j] for i in cow for j in calf), calf_to_cow=sum(m[i][j] for i in calf for j in cow))

if __name__ == "__main__":
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    labels = data.get("labels")
    print(f"{'model':32s} {'acc %':>7s} {'95% CI':>15s} {'wF1':>6s} {'mF1':>6s} {'bAcc':>6s} {'MmCows':>7s} {'PlayB.':>7s}")
    for key, entry in data["models"].items():
        x = metrics(entry["matrix"])
        lo, hi = x["ci"]
        print(f"{entry.get('name', key):32s} {100 * x['acc']:7.2f} {100 * lo:7.2f}–{100 * hi:6.2f} {x['weighted_f1']:6.3f} "
              f"{x['macro_f1']:6.3f} {x['balanced_acc']:6.3f} {100 * x['acc_mmcows']:7.2f} {100 * x['acc_playbehavior']:7.2f}")
        if "--per-class" in sys.argv:
            for i, lab in enumerate(labels or range(9)):
                print(f"    {lab:18s} P {x['P'][i]:.3f}  R {x['R'][i]:.3f}  F1 {x['F1'][i]:.3f}")
