"""Faithful headless transcription of the acquired system's mathematics.

This module exists for one purpose: to let the extraction be *proven* faithful
rather than asserted. It reproduces the acquired Streamlit application's
optimizer and Monte Carlo exactly — same variables, same constraints, same
hard-coded constants, same global-seed sequence — with the UI removed and inputs
passed as plain dicts.

Nothing here should be used in production. It is the reference against which the
new engine's parity mode is diffed, and the fixture that lets historic results
be reproduced when a customer asks why last quarter's number changed.

Line references point at the acquired single-file application.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pulp as pl

__all__ = ["legacy_optimize", "legacy_monte_carlo", "legacy_marginal_risks"]


def legacy_marginal_risks(
    risks: Mapping[str, float], macro_multiplier: float, exponent: float = 0.85
) -> dict[str, float]:
    """``(macro * risk) ** 0.85``, computed per node before optimization."""
    return {
        node: float((macro_multiplier * risks[node]) ** exponent) for node in risks
    }


def legacy_optimize(
    *,
    nodes: Sequence[str],
    costs: Mapping[str, float],
    marginal_risks: Mapping[str, float],
    lead_times: Mapping[str, float],
    opt_weight: float,
    total_budget: float,
    dependencies: Sequence[tuple[str, str]] = (),
    bundles: Sequence[tuple[str, float, Sequence[str]]] = (),
) -> dict[str, object]:
    """Commercial mode: maximise the weighted sum under a net budget cap."""
    prob = pl.LpProblem("Commercial_Supply_Chain_Hybrid_Optimization", pl.LpMaximize)
    y = {n: pl.LpVariable(f"y_{n}", cat="Binary") for n in nodes}
    x = {
        n: pl.LpVariable(f"x_{n}", lowBound=0.0, upBound=1.0, cat="Continuous")
        for n in nodes
    }

    for n in nodes:
        prob += x[n] >= 0.3 * y[n]
        prob += x[n] <= 1.0 * y[n]

    bundle_vars: dict[str, dict[str, object]] = {}
    for index, (name, discount, required) in enumerate(bundles):
        b_var = pl.LpVariable(f"bundle_{index}", cat="Binary")
        bundle_vars[name] = {"var": b_var, "discount": float(discount)}
        for rn in required:
            if rn in nodes:
                prob += b_var <= y[rn]

    for dependent, prerequisite in dependencies:
        if dependent in nodes and prerequisite in nodes and dependent != prerequisite:
            prob += x[dependent] <= x[prerequisite]

    prob += pl.lpSum(
        (opt_weight * marginal_risks[n] + (1 - opt_weight) * lead_times[n]) * x[n]
        for n in nodes
    )

    total_discounts = (
        pl.lpSum(bundle_vars[b]["var"] * bundle_vars[b]["discount"] for b in bundle_vars)
        if bundle_vars
        else 0
    )
    prob += pl.lpSum(costs[n] * x[n] for n in nodes) - total_discounts <= total_budget

    prob.solve(pl.PULP_CBC_CMD(msg=False))

    scales = {
        n: float(pl.value(x[n])) if pl.value(x[n]) is not None else 0.0 for n in nodes
    }
    base_cost = sum(costs[n] * s for n, s in scales.items())
    applied_discounts = 0.0
    active_bundles: list[str] = []
    for name, data in bundle_vars.items():
        if pl.value(data["var"]) == 1:
            applied_discounts += float(data["discount"])
            active_bundles.append(name)

    return {
        "status": pl.LpStatus[prob.status],
        "scales": scales,
        "gross_cost": base_cost,
        "discounts": applied_discounts,
        "net_cost": base_cost - applied_discounts,
        "risk_drop": sum(marginal_risks[n] * s for n, s in scales.items()),
        "lead_time": sum(lead_times[n] * s for n, s in scales.items()),
        "objective": pl.value(prob.objective),
        "active_bundles": active_bundles,
    }


def legacy_target_optimize(
    *,
    nodes: Sequence[str],
    costs: Mapping[str, float],
    marginal_risks: Mapping[str, float],
    effective_baseline_risk: float,
    target_risk_goal: float,
    dependencies: Sequence[tuple[str, str]] = (),
    bundles: Sequence[tuple[str, float, Sequence[str]]] = (),
) -> dict[str, object]:
    """Target mode: minimise net capital subject to a risk-reduction floor."""
    prob = pl.LpProblem("Target_Goal_Hybrid_Optimization", pl.LpMinimize)
    y = {n: pl.LpVariable(f"y_{n}", cat="Binary") for n in nodes}
    x = {
        n: pl.LpVariable(f"x_{n}", lowBound=0.0, upBound=1.0, cat="Continuous")
        for n in nodes
    }

    for n in nodes:
        prob += x[n] >= 0.3 * y[n]
        prob += x[n] <= 1.0 * y[n]

    bundle_vars: dict[str, dict[str, object]] = {}
    for index, (name, discount, required) in enumerate(bundles):
        b_var = pl.LpVariable(f"bundle_{index}", cat="Binary")
        bundle_vars[name] = {"var": b_var, "discount": float(discount)}
        for rn in required:
            if rn in nodes:
                prob += b_var <= y[rn]

    for dependent, prerequisite in dependencies:
        if dependent in nodes and prerequisite in nodes and dependent != prerequisite:
            prob += x[dependent] <= x[prerequisite]

    total_discounts = (
        pl.lpSum(bundle_vars[b]["var"] * bundle_vars[b]["discount"] for b in bundle_vars)
        if bundle_vars
        else 0
    )
    prob += pl.lpSum(costs[n] * x[n] for n in nodes) - total_discounts
    required_risk_drop = max(0.0, effective_baseline_risk - target_risk_goal)
    prob += pl.lpSum(marginal_risks[n] * x[n] for n in nodes) >= required_risk_drop

    prob.solve(pl.PULP_CBC_CMD(msg=False))

    scales = {
        n: float(pl.value(x[n])) if pl.value(x[n]) is not None else 0.0 for n in nodes
    }
    base_cost = sum(costs[n] * s for n, s in scales.items())
    applied_discounts = 0.0
    for name, data in bundle_vars.items():
        if pl.value(data["var"]) == 1:
            applied_discounts += float(data["discount"])

    return {
        "status": pl.LpStatus[prob.status],
        "scales": scales,
        "gross_cost": base_cost,
        "discounts": applied_discounts,
        "net_cost": base_cost - applied_discounts,
        "risk_drop": sum(marginal_risks[n] * s for n, s in scales.items()),
        "required_risk_drop": required_risk_drop,
        "objective": pl.value(prob.objective),
    }


def legacy_monte_carlo(
    base_risk: float,
    scales_dict: Mapping[str, float],
    marginal_dict: Mapping[str, float],
    corr_matrix: np.ndarray,
    nodes_list: Sequence[str],
    iterations: int,
) -> dict[str, float]:
    """The acquired Monte Carlo, including the global seed and the 0.4 floor.

    Reproduced draw-for-draw: ``np.random.seed(42)`` followed by one
    ``np.random.normal(0, 1, k)`` call per iteration.
    """
    np.random.seed(42)
    k = len(nodes_list)
    if k == 0 or iterations <= 0:
        return {"P50_Risk": base_risk, "P90_Risk": base_risk, "Std_Dev": 0.0}

    matrix = np.asarray(corr_matrix, dtype=float).copy()
    matrix = np.clip(matrix, -0.99, 0.99)
    np.fill_diagonal(matrix, 1.0)

    try:
        chol = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        min_eig = np.min(np.real(np.linalg.eigvals(matrix)))
        matrix = matrix + np.eye(k) * (-min_eig + 1e-5)
        chol = np.linalg.cholesky(matrix)

    simulated: list[float] = []
    for _ in range(iterations):
        uncorrelated_z = np.random.normal(0, 1, k)
        correlated_z = np.dot(chol, uncorrelated_z)
        shocks = 1.0 + (0.12 * correlated_z)
        simulated_drop = sum(
            marginal_dict.get(n, 0.0) * scales_dict.get(n, 0.0) * max(0.4, shocks[i])
            for i, n in enumerate(nodes_list)
        )
        simulated.append(max(0.0, base_risk - simulated_drop))

    array = np.array(simulated)
    return {
        "P50_Risk": float(np.percentile(array, 50)),
        "P90_Risk": float(np.percentile(array, 90)),
        "Std_Dev": float(np.std(array)),
    }
