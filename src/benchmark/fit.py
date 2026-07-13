"""Fit BKT parameters via grid search — find params that maximize AUC on ASSISTments.

Method: grid search (brute-force but interpretable). The sophisticated
alternative is Expectation-Maximization; grid search is chosen here for
transparency and because it produces a defensible, reproducible result.

Scope: GLOBAL params (one set for all skills). Per-skill fitting is more
granular but overfits low-data skills — documented as future work.
"""

import itertools
from sklearn.metrics import roc_auc_score

from src.analytics.bkt import BKTParams
from src.benchmark.evaluate import load_and_clean, evaluate_bkt


def grid_search(df):
    """Search a grid of BKT params; return (best_params, best_auc, all_results)."""
    # Candidate values per parameter — kept small so the search is tractable
    grid = {
        "p_L0":    [0.1, 0.2, 0.3, 0.4],
        "p_T":     [0.1, 0.2, 0.3, 0.4],
        "p_slip":  [0.05, 0.1, 0.2, 0.3],
        "p_guess": [0.1, 0.2, 0.3],
    }
    # Constraint: slip + guess < 1 (BKT identifiability — a known requirement)
    combos = [
        (l0, t, s, g)
        for l0, t, s, g in itertools.product(
            grid["p_L0"], grid["p_T"], grid["p_slip"], grid["p_guess"]
        )
        if s + g < 1.0
    ]
    print(f"Searching {len(combos)} parameter combinations...")

    best_auc, best_params, results = 0.0, None, []
    for i, (l0, t, s, g) in enumerate(combos, 1):
        params = BKTParams(p_L0=l0, p_T=t, p_slip=s, p_guess=g)
        preds, actuals = evaluate_bkt(df, params)
        auc = roc_auc_score(actuals, preds)
        results.append((params, auc))
        if auc > best_auc:
            best_auc, best_params = auc, params
        if i % 20 == 0:
            print(f"  ...{i}/{len(combos)} done (best so far: {best_auc:.4f})")

    return best_params, best_auc, results


def main():
    df = load_and_clean()

    # Baseline: the hand-picked defaults from before
    default_preds, default_actuals = evaluate_bkt(df, BKTParams())
    default_auc = roc_auc_score(default_actuals, default_preds)

    print(f"\nDefault (hand-picked) AUC: {default_auc:.4f}\n")

    best_params, best_auc, _ = grid_search(df)

    print("\n" + "=" * 50)
    print(f"  Default params AUC: {default_auc:.4f}")
    print(f"  Fitted params AUC:  {best_auc:.4f}")
    print(f"  Improvement:        {best_auc - default_auc:+.4f}")
    print(f"\n  Best params: L0={best_params.p_L0}, T={best_params.p_T}, "
          f"slip={best_params.p_slip}, guess={best_params.p_guess}")
    print("=" * 50)


if __name__ == "__main__":
    main()