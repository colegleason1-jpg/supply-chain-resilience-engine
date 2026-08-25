"""Translation between spreadsheet-shaped user input and `scrcae` domain objects.

This is the only module allowed to know about both a `pandas.DataFrame` and a
`SupplyNetwork`, and it is where the acquired system's data bugs lived. It contains
no optimisation and no risk arithmetic: it builds validated domain objects and hands
them to the engine.

Three rules it enforces, each of them a fix for a specific finding:

* **Nothing is silently coerced into a model.** A row that cannot be understood is
  reported as a rejection with a reason, not dropped or defaulted. The acquired
  system's editor accepted blank costs and NaN risk values and passed them to the
  solver as zeros.
* **Bundle membership is data, not widget order.** `required_nodes` came from
  `nodes[:2]` — the first two rows of whatever the editor happened to be showing
  (F5). Membership is now an explicit column that the user fills in, and a bundle
  naming an unknown node is a rejection rather than a silently smaller bundle.
* **Correlations are read by node id, positionally verified.** The acquired system
  looked up correlation cells by exact string match on a display name and fell back
  to 0.35 on any exception, including a typo (F11). A mismatch is now an error.

The functions here return `(object, Rejections)` rather than raising, because a
spreadsheet with one bad row should still let a user see the other nineteen.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from scrcae import (
    Bundle,
    Dependency,
    Intervention,
    LegacyTruncatedNormalShock,
    LegacyWeightedObjective,
    LognormalShock,
    MinimizeCapitalObjective,
    MonetaryNPVObjective,
    OptimizationRequest,
    PriceBook,
    SimulationRequest,
    SupplyNetwork,
)
from scrcae.domain import DomainError
from scrcae.risk import AllocationConcaveResponse, LinearResponse, ParameterPowerResponse

# Column names are the app's contract with the user's spreadsheet. Defined once so a
# rename is a one-line change rather than a search across view code.
COL_NODE_ID = "Node ID"
COL_NAME = "Node Name"
COL_ACTION = "Action"
COL_COST = "Cost"
COL_RISK = "Risk Reduction (%)"
COL_LEAD_TIME = "Lead Time Saved (Days)"
COL_CARBON = "Carbon Impact (Tons)"
COL_MIN_SCALE = "Min Funding Scale"
COL_MAX_SCALE = "Max Funding Scale"

NODE_COLUMNS = (
    COL_NODE_ID,
    COL_NAME,
    COL_ACTION,
    COL_COST,
    COL_RISK,
    COL_LEAD_TIME,
    COL_CARBON,
    COL_MIN_SCALE,
    COL_MAX_SCALE,
)

COL_BUNDLE_NAME = "Bundle Name"
COL_BUNDLE_DISCOUNT = "Discount ($)"
COL_BUNDLE_NODES = "Required Nodes"

COL_DEP_DEPENDENT = "Dependent Node"
COL_DEP_PREREQUISITE = "Prerequisite Node"

# The market exposure table (F11). The acquired system inferred this relationship by
# comparing a market's sector name to a node's display name with `==`, and applied one
# market to every node whenever the column it compared was missing. It is now four
# explicit columns the user fills in and can see.
COL_EXP_NODE_ID = "Node ID"
COL_EXP_SYMBOL = "Market Symbol"
COL_EXP_ANCHOR = "Anchor Price"
COL_EXP_ELASTICITY = "Risk Elasticity"

#: How an elasticity came to hold its value. Deliberately not a column in the
#: exposure editor: a free-text "source" field is a field in which a user can type
#: the word "calibrated", which is the precise laundering this rebuild exists to
#: stop. It is derived instead, by matching the entered number against a stored fit.
ASSERTED = "asserted"

#: Delivery history, the input to elasticity calibration.
COL_HIST_NODE_ID = "Node ID"
COL_HIST_PERIOD = "Period"
COL_HIST_DISRUPTION = "Disruption Rate %"

HISTORY_COLUMNS = (
    COL_HIST_NODE_ID,
    COL_HIST_PERIOD,
    COL_HIST_DISRUPTION,
)

#: A stored elasticity fit. Kept as a table rather than an opaque blob so it saves
#: with the portfolio through the same path as every other input, and so a user can
#: hand an auditor the evidence behind a number instead of describing it.
COL_CAL_NODE_ID = "Node ID"
COL_CAL_SYMBOL = "Market Symbol"
COL_CAL_ELASTICITY = "Fitted Elasticity"
COL_CAL_ANCHOR = "Anchor Price"
COL_CAL_LOW = "Interval Low"
COL_CAL_HIGH = "Interval High"
COL_CAL_R2 = "R Squared"
COL_CAL_OBS = "Periods"
COL_CAL_ABOVE = "Periods Above Anchor"
COL_CAL_METHOD = "Method"

CALIBRATION_COLUMNS = (
    COL_CAL_NODE_ID,
    COL_CAL_SYMBOL,
    COL_CAL_ELASTICITY,
    COL_CAL_ANCHOR,
    COL_CAL_LOW,
    COL_CAL_HIGH,
    COL_CAL_R2,
    COL_CAL_OBS,
    COL_CAL_ABOVE,
    COL_CAL_METHOD,
)

EXPOSURE_COLUMNS = (
    COL_EXP_NODE_ID,
    COL_EXP_SYMBOL,
    COL_EXP_ANCHOR,
    COL_EXP_ELASTICITY,
)


@dataclass
class Rejections:
    """Input the adapter refused, with a reason per item.

    Carried alongside the result rather than raised so that one unusable row does not
    hide the rest of the sheet. The app is required to display these: silently
    accepting 19 of 20 rows and optimising over the 19 is how a user ends up
    presenting a plan that omits an intervention they thought they had funded.
    """

    items: list[str] = field(default_factory=list)

    def add(self, what: str, why: str) -> None:
        self.items.append(f"{what}: {why}")

    def extend(self, other: "Rejections") -> None:
        self.items.extend(other.items)

    def __bool__(self) -> bool:
        return bool(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


def _number(value: Any) -> float | None:
    """Parse a spreadsheet cell as a finite float, or return None.

    Returns None rather than 0.0 for unparseable input. Defaulting a missing cost to
    zero produces a free intervention, which the optimizer will happily fund; that is
    exactly the class of error this function exists to make impossible.
    """
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")
        if not cleaned:
            return None
        try:
            value = float(cleaned)
        except ValueError:
            return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def default_nodes_frame() -> pd.DataFrame:
    """The starting portfolio, matching the acquired system's demo data.

    Kept identical so that a user who knew the old app sees the same starting point
    and can compare answers, which is the only way the change in results is
    explainable rather than merely different.
    """
    return pd.DataFrame(
        [
            {
                COL_NODE_ID: "N1",
                COL_NAME: "Dual_Source_Ports",
                COL_ACTION: "Qualify second port of entry",
                COL_COST: 180_000.0,
                COL_RISK: 12.5,
                COL_LEAD_TIME: 9.0,
                COL_CARBON: 40.0,
                COL_MIN_SCALE: 0.3,
                COL_MAX_SCALE: 1.0,
            },
            {
                COL_NODE_ID: "N2",
                COL_NAME: "Buffer_Warehouse",
                COL_ACTION: "Regional buffer stock",
                COL_COST: 240_000.0,
                COL_RISK: 9.5,
                COL_LEAD_TIME: 6.0,
                COL_CARBON: 65.0,
                COL_MIN_SCALE: 0.3,
                COL_MAX_SCALE: 1.0,
            },
            {
                COL_NODE_ID: "N3",
                COL_NAME: "Supplier_Audit",
                COL_ACTION: "Tier-2 supplier audit programme",
                COL_COST: 70_000.0,
                COL_RISK: 4.25,
                COL_LEAD_TIME: 2.5,
                COL_CARBON: 5.0,
                COL_MIN_SCALE: 0.3,
                COL_MAX_SCALE: 1.0,
            },
            {
                COL_NODE_ID: "N4",
                COL_NAME: "Nearshore_Assembly",
                COL_ACTION: "Shift assembly nearshore",
                COL_COST: 520_000.0,
                COL_RISK: 18.0,
                COL_LEAD_TIME: 21.0,
                COL_CARBON: 130.0,
                COL_MIN_SCALE: 0.3,
                COL_MAX_SCALE: 1.0,
            },
            {
                COL_NODE_ID: "N5",
                COL_NAME: "Freight_Contracts",
                COL_ACTION: "Lock multi-carrier freight capacity",
                COL_COST: 95_000.0,
                COL_RISK: 5.5,
                COL_LEAD_TIME: 4.0,
                COL_CARBON: 15.0,
                COL_MIN_SCALE: 0.3,
                COL_MAX_SCALE: 1.0,
            },
        ]
    )


def default_bundles_frame() -> pd.DataFrame:
    """Bundle membership is an explicit column, not the first two rows (F5)."""
    return pd.DataFrame(
        [
            {
                COL_BUNDLE_NAME: "Port_Warehouse_Synergy",
                COL_BUNDLE_DISCOUNT: 50_000.0,
                COL_BUNDLE_NODES: "N1, N2",
            }
        ]
    )


def default_dependencies_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[COL_DEP_DEPENDENT, COL_DEP_PREREQUISITE])


def default_correlation_frame(node_ids: Sequence[str], rho: float = 0.35) -> pd.DataFrame:
    data: dict[str, Any] = {COL_NODE_ID: list(node_ids)}
    for col in node_ids:
        data[col] = [1.0 if col == row else rho for row in node_ids]
    return pd.DataFrame(data)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def node_ids_in(frame: pd.DataFrame) -> list[str]:
    """The usable node IDs in a frame, in sheet order, without duplicates.

    Used to keep dependent editors (bundles, correlations) in step with the
    intervention table without re-parsing the whole thing.
    """
    if frame is None or frame.empty or COL_NODE_ID not in frame.columns:
        return []
    seen: list[str] = []
    for raw in frame[COL_NODE_ID]:
        node_id = _text(raw)
        if node_id and node_id not in seen:
            seen.append(node_id)
    return seen


def align_correlation_frame(
    frame: pd.DataFrame | None, node_ids: Sequence[str], rho: float = 0.35
) -> pd.DataFrame:
    """Reshape a correlation frame to the current node set, preserving what we can.

    Adding an intervention must not discard the correlations already entered for the
    others, and removing one must not leave a stale row that silently reappears if the
    node comes back. Pairs that were never specified default to ``rho``, which is an
    assumption — the same 0.35 the acquired system used — but here it is applied
    visibly in an editor the user can see and change, rather than inside an exception
    handler (F11).
    """
    ids = list(dict.fromkeys(str(n) for n in node_ids))
    fresh = default_correlation_frame(ids, rho=rho)
    if frame is None or frame.empty or COL_NODE_ID not in frame.columns:
        return fresh

    previous = {_text(row[COL_NODE_ID]): row for _, row in frame.iterrows()}
    for i, row_id in enumerate(ids):
        old = previous.get(row_id)
        if old is None:
            continue
        for col_id in ids:
            if col_id == row_id or col_id not in old.index:
                continue
            value = _number(old[col_id])
            if value is not None:
                fresh.loc[i, col_id] = value
    return fresh


def build_interventions(
    frame: pd.DataFrame,
) -> tuple[tuple[Intervention, ...], Rejections]:
    """Parse the node editor into interventions, rejecting what cannot be trusted."""
    rejections = Rejections()
    interventions: list[Intervention] = []
    seen: set[str] = set()

    if frame is None or frame.empty:
        return (), rejections

    missing = [c for c in (COL_NODE_ID, COL_COST, COL_RISK) if c not in frame.columns]
    if missing:
        rejections.add("Node table", f"missing required column(s): {', '.join(missing)}")
        return (), rejections

    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        label = f"Row {position}"
        node_id = _text(row.get(COL_NODE_ID))
        if not node_id:
            # A blank id is how a user leaves a trailing empty row in an editor.
            # Only complain if the row has content elsewhere.
            if any(_number(row.get(c)) for c in (COL_COST, COL_RISK)):
                rejections.add(label, "has values but no Node ID")
            continue
        if node_id in seen:
            rejections.add(f"{label} ({node_id})", "duplicate Node ID")
            continue

        cost = _number(row.get(COL_COST))
        risk = _number(row.get(COL_RISK))
        if cost is None:
            rejections.add(f"{label} ({node_id})", "Cost is missing or not a number")
            continue
        if risk is None:
            rejections.add(
                f"{label} ({node_id})", "Risk Reduction (%) is missing or not a number"
            )
            continue
        if cost < 0:
            rejections.add(f"{label} ({node_id})", "Cost is negative")
            continue
        if risk < 0:
            rejections.add(f"{label} ({node_id})", "Risk Reduction (%) is negative")
            continue

        min_scale = _number(row.get(COL_MIN_SCALE))
        max_scale = _number(row.get(COL_MAX_SCALE))
        min_scale = 0.3 if min_scale is None else min_scale
        max_scale = 1.0 if max_scale is None else max_scale
        if min_scale > max_scale:
            rejections.add(
                f"{label} ({node_id})",
                f"minimum funding scale {min_scale:g} exceeds maximum {max_scale:g}",
            )
            continue

        try:
            interventions.append(
                Intervention(
                    node_id=node_id,
                    name=_text(row.get(COL_NAME)) or node_id,
                    action=_text(row.get(COL_ACTION)),
                    cost=cost,
                    risk_reduction_pts=risk,
                    lead_time_saved_days=_number(row.get(COL_LEAD_TIME)) or 0.0,
                    carbon_tons=_number(row.get(COL_CARBON)) or 0.0,
                    min_funding_scale=min_scale,
                    max_funding_scale=max_scale,
                )
            )
        except DomainError as exc:
            # The domain model rejects things the adapter does not know to check.
            # Surfacing its message verbatim keeps one source of truth for validity.
            rejections.add(f"{label} ({node_id})", str(exc))
            continue
        seen.add(node_id)

    return tuple(interventions), rejections


def parse_node_list(value: Any) -> tuple[str, ...]:
    """Parse a comma- or semicolon-separated node list from one cell."""
    text = _text(value)
    if not text:
        return ()
    parts = [p.strip() for chunk in text.split(";") for p in chunk.split(",")]
    return tuple(p for p in parts if p)


def build_bundles(
    frame: pd.DataFrame, known_ids: Iterable[str]
) -> tuple[tuple[Bundle, ...], Rejections]:
    """Parse bundles, requiring explicit membership.

    A bundle that names an unknown node is rejected outright rather than quietly
    shrunk to its recognised members. Shrinking changes which discount the model can
    claim and for what, and the user would never see that it happened.
    """
    rejections = Rejections()
    bundles: list[Bundle] = []
    known = set(known_ids)

    if frame is None or frame.empty:
        return (), rejections

    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        name = _text(row.get(COL_BUNDLE_NAME))
        discount = _number(row.get(COL_BUNDLE_DISCOUNT))
        members = parse_node_list(row.get(COL_BUNDLE_NODES))
        label = f"Bundle row {position}"

        if not name and discount is None and not members:
            continue
        if not name:
            rejections.add(label, "no Bundle Name")
            continue
        label = f"Bundle '{name}'"
        if discount is None:
            rejections.add(label, "Discount ($) is missing or not a number")
            continue
        if not members:
            rejections.add(
                label,
                "no Required Nodes. List the node ids the discount depends on, "
                "e.g. 'N1, N2'",
            )
            continue

        unknown = [m for m in members if m not in known]
        if unknown:
            rejections.add(label, f"names unknown node(s): {', '.join(unknown)}")
            continue

        try:
            bundles.append(
                Bundle(name=name, discount=discount, required_nodes=tuple(members))
            )
        except DomainError as exc:
            rejections.add(label, str(exc))
            continue

    return tuple(bundles), rejections


def build_dependencies(
    frame: pd.DataFrame, known_ids: Iterable[str]
) -> tuple[tuple[Dependency, ...], Rejections]:
    rejections = Rejections()
    deps: list[Dependency] = []
    known = set(known_ids)
    seen: set[tuple[str, str]] = set()

    if frame is None or frame.empty:
        return (), rejections

    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        dependent = _text(row.get(COL_DEP_DEPENDENT))
        prerequisite = _text(row.get(COL_DEP_PREREQUISITE))
        label = f"Dependency row {position}"

        if not dependent and not prerequisite:
            continue
        if not dependent or not prerequisite:
            rejections.add(label, "needs both a dependent and a prerequisite node")
            continue
        unknown = [n for n in (dependent, prerequisite) if n not in known]
        if unknown:
            rejections.add(label, f"names unknown node(s): {', '.join(unknown)}")
            continue
        if dependent == prerequisite:
            rejections.add(label, f"'{dependent}' cannot depend on itself")
            continue
        if (dependent, prerequisite) in seen:
            continue
        seen.add((dependent, prerequisite))
        deps.append(Dependency(dependent=dependent, prerequisite=prerequisite))

    return tuple(deps), rejections


def build_network(
    nodes_frame: pd.DataFrame,
    baseline_risk_pts: float,
    bundles_frame: pd.DataFrame | None = None,
    dependencies_frame: pd.DataFrame | None = None,
) -> tuple[SupplyNetwork | None, Rejections]:
    """Assemble a validated `SupplyNetwork`, or None with reasons why not.

    Returning None rather than an empty network matters: an empty network solves
    successfully and reports zero risk reduction for zero capital, which reads as a
    legitimate "do nothing" recommendation rather than as broken input.
    """
    rejections = Rejections()
    interventions, node_rejections = build_interventions(nodes_frame)
    rejections.extend(node_rejections)

    if not interventions:
        rejections.add("Portfolio", "no usable interventions were found")
        return None, rejections

    known = [i.node_id for i in interventions]
    bundles, bundle_rejections = build_bundles(bundles_frame, known)
    dependencies, dep_rejections = build_dependencies(dependencies_frame, known)
    rejections.extend(bundle_rejections)
    rejections.extend(dep_rejections)

    baseline = _number(baseline_risk_pts)
    if baseline is None or baseline < 0:
        rejections.add("Baseline risk", "must be a non-negative number")
        return None, rejections

    try:
        network = SupplyNetwork(
            interventions=interventions,
            baseline_risk_pts=baseline,
            dependencies=dependencies,
            bundles=bundles,
        )
    except DomainError as exc:
        # Dependency cycles and discount-exceeds-gross land here. The domain model is
        # the authority on structural validity; the adapter does not second-guess it.
        rejections.add("Network", str(exc))
        return None, rejections

    return network, rejections


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #


def build_correlation_matrix(
    frame: pd.DataFrame, node_ids: Sequence[str]
) -> tuple[np.ndarray, Rejections]:
    """Read the correlation editor into a matrix ordered to match ``node_ids``.

    The acquired system matched cells by display name inside a bare ``except:`` that
    substituted 0.35 for anything it could not find (F11). A renamed node therefore
    silently changed the correlation structure of the simulation. Here a missing or
    unparseable cell is a rejection, and the identity diagonal is asserted rather
    than assumed.
    """
    rejections = Rejections()
    size = len(node_ids)
    matrix = np.eye(size)
    if size == 0:
        return np.zeros((0, 0)), rejections

    if frame is None or frame.empty or COL_NODE_ID not in frame.columns:
        rejections.add(
            "Correlation matrix",
            "not available for the current nodes; assuming independence",
        )
        return matrix, rejections

    index = {_text(v): i for i, v in enumerate(frame[COL_NODE_ID])}
    for i, row_id in enumerate(node_ids):
        if row_id not in index:
            rejections.add("Correlation matrix", f"no row for node '{row_id}'")
            continue
        for j, col_id in enumerate(node_ids):
            if i == j:
                continue
            if col_id not in frame.columns:
                rejections.add("Correlation matrix", f"no column for node '{col_id}'")
                continue
            value = _number(frame.iloc[index[row_id]][col_id])
            if value is None:
                rejections.add(
                    "Correlation matrix", f"cell ({row_id}, {col_id}) is not a number"
                )
                continue
            matrix[i, j] = max(-0.99, min(0.99, value))

    # Symmetrise explicitly. An editor lets a user fill one triangle and forget the
    # other, and an asymmetric "correlation" matrix is not a correlation matrix; the
    # engine's repair step would otherwise silently absorb the asymmetry.
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)
    return matrix, rejections


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #

#: Mode identifiers. Strings rather than an enum so they survive Streamlit's
#: session-state round-tripping without import gymnastics.
MODE_MONETARY = "monetary"
MODE_LEGACY = "legacy"


def build_objective(
    mode: str,
    *,
    prices: PriceBook | None = None,
    legacy_weight: float = 0.5,
    required_risk_reduction_pts: float | None = None,
):
    """Choose the objective. Target mode minimises capital in either mode."""
    if required_risk_reduction_pts is not None:
        return MinimizeCapitalObjective()
    if mode == MODE_LEGACY:
        return LegacyWeightedObjective(weight=legacy_weight)
    if prices is None:
        raise ValueError("monetary mode requires a PriceBook")
    return MonetaryNPVObjective(prices=prices)


def build_risk_response(mode: str, exponent: float = 0.85):
    """Legacy mode transforms the risk *parameter*; monetary mode the *allocation*.

    This one line is finding F2. The acquired system raised the risk parameter to
    0.85, which is not a diminishing return on spending at all — it reweights nodes
    against each other and leaves the model perfectly linear in funding. Monetary
    mode applies the exponent to the allocation, which is what the documentation
    always claimed was happening.
    """
    if mode == MODE_LEGACY:
        return ParameterPowerResponse(exponent=exponent)
    if exponent >= 1.0:
        return LinearResponse()
    return AllocationConcaveResponse(exponent=exponent)


def build_shock(mode: str, sigma: float = 0.12):
    """Legacy keeps the truncated normal and its upward bias (F7, F8)."""
    if mode == MODE_LEGACY:
        return LegacyTruncatedNormalShock(sigma=sigma)
    return LognormalShock(sigma=sigma)


def build_optimization_request(
    network: SupplyNetwork,
    *,
    mode: str,
    budget: float | None,
    macro_multiplier: float = 1.0,
    prices: PriceBook | None = None,
    legacy_weight: float = 0.5,
    exponent: float = 0.85,
    target_risk_pts: float | None = None,
    node_macro_multipliers: Mapping[str, float] | None = None,
) -> OptimizationRequest:
    """Assemble the request. Mode selects a coherent bundle of model choices.

    Mode is deliberately not a menu of independent switches. "Legacy" means the
    weighted objective *and* the parameter-power response *and* the truncated-normal
    shock *and* the post-hoc risk cap, because that combination is what reproduces
    the acquired system's numbers. Letting a user mix half of one mode into the other
    would produce results that are neither reproducible nor defensible.
    """
    legacy = mode == MODE_LEGACY
    required = None
    if target_risk_pts is not None:
        # Derived from the macro-inflated baseline, which is the baseline the user
        # sees on screen and sets the target against (ADR-001 F18).
        effective_baseline = min(100.0, network.baseline_risk_pts * macro_multiplier)
        required = max(0.0, effective_baseline - float(target_risk_pts))

    return OptimizationRequest(
        network=network,
        objective=build_objective(
            mode,
            prices=prices,
            legacy_weight=legacy_weight,
            required_risk_reduction_pts=required,
        ),
        risk_response=build_risk_response(mode, exponent),
        budget=budget,
        required_risk_reduction_pts=required,
        macro_multiplier=macro_multiplier,
        # Per-node market exposure is a solve-time overlay. It never enters the node
        # table, so repeated feed syncs cannot compound into the user's own figures
        # the way the acquired system's write-back did (F11).
        node_macro_multipliers=dict(node_macro_multipliers or {}),
        enforce_risk_cap=not legacy,
        legacy_bundle_activation=legacy,
    )


def build_simulation_request(
    result,
    network: SupplyNetwork,
    correlation: np.ndarray,
    *,
    mode: str,
    iterations: int = 10_000,
    seed: int = 42,
    sigma: float = 0.12,
) -> SimulationRequest:
    """Simulate the portfolio the optimizer actually chose.

    Per-node reduction is read from the result's allocations, which the engine
    recomputes from the risk response rather than reading back out of the linearised
    program. Taking it from the LP would report the approximation's view of the
    answer instead of the answer.
    """
    node_order = tuple(i.node_id for i in network)
    reductions = {a.node_id: a.risk_reduction_pts for a in result.allocations}
    return SimulationRequest(
        baseline_risk_pts=result.reporting_baseline_risk_pts,
        risk_reduction_by_node={n: reductions.get(n, 0.0) for n in node_order},
        correlation=correlation,
        node_order=node_order,
        iterations=iterations,
        seed=seed,
        shock_model=build_shock(mode, sigma),
    )


# --------------------------------------------------------------------------- #
# Market exposure (F11)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class NodeExposure:
    """One node's sensitivity to one market.

    ``elasticity`` is the proportional change in the node's risk-reduction
    parameter per unit proportional deviation of the market price from
    ``anchor``. An elasticity of 0.2 means "a 10% move above anchor makes this
    node's intervention 2% more valuable".

    It is an uncalibrated assumption, exactly like the response exponent, and it
    is stored as a named per-row number rather than one global constant so that
    it can be argued with per node and replaced by a calibration later. The
    acquired system's equivalent was a single hardcoded 0.15 applied to whatever
    it managed to match.
    """

    node_id: str
    symbol: str
    anchor: float
    elasticity: float
    #: ``"asserted"`` or ``"calibrated (<method>)"``. Never entered by a user — see
    #: ``resolve_elasticity_sources`` for why it is derived rather than typed.
    elasticity_source: str = ASSERTED

    @property
    def is_calibrated(self) -> bool:
        return self.elasticity_source != ASSERTED


def default_exposure_frame(node_ids: Sequence[str]) -> pd.DataFrame:
    """An empty exposure sheet, one blank row per node.

    Blank rather than pre-populated with a guessed market. A pre-filled mapping
    would be indistinguishable on screen from one the user had reviewed, and the
    whole point of this table is that the relationship is asserted by a person.
    """
    ids = list(dict.fromkeys(str(n) for n in node_ids))
    return pd.DataFrame(
        {
            COL_EXP_NODE_ID: ids,
            COL_EXP_SYMBOL: ["" for _ in ids],
            COL_EXP_ANCHOR: [None for _ in ids],
            COL_EXP_ELASTICITY: [None for _ in ids],
        }
    )


def align_exposure_frame(
    frame: pd.DataFrame | None, node_ids: Sequence[str]
) -> pd.DataFrame:
    """Reshape an exposure sheet to the current node set, preserving entered rows.

    Mirrors ``align_correlation_frame``: adding an intervention must not discard
    the exposures already entered for the others. Rows naming nodes that no longer
    exist are dropped here rather than carried forward, because the engine rejects
    unknown node ids and a stale row would turn an unrelated edit into a hard
    failure the user cannot connect to anything they did.
    """
    ids = list(dict.fromkeys(str(n) for n in node_ids))
    if frame is None or frame.empty or COL_EXP_NODE_ID not in frame.columns:
        return default_exposure_frame(ids)

    keep = [c for c in EXPOSURE_COLUMNS if c in frame.columns]
    trimmed = frame[keep].copy()
    trimmed[COL_EXP_NODE_ID] = trimmed[COL_EXP_NODE_ID].map(_text)
    trimmed = trimmed[trimmed[COL_EXP_NODE_ID].isin(ids)]

    for column in EXPOSURE_COLUMNS:
        if column not in trimmed.columns:
            trimmed[column] = "" if column == COL_EXP_SYMBOL else None

    present = set(trimmed[COL_EXP_NODE_ID])
    missing = [n for n in ids if n not in present]
    if missing:
        trimmed = pd.concat(
            [trimmed, default_exposure_frame(missing)], ignore_index=True
        )
    return trimmed[list(EXPOSURE_COLUMNS)].reset_index(drop=True)


def build_node_exposures(
    frame: pd.DataFrame | None, node_ids: Sequence[str]
) -> tuple[tuple[NodeExposure, ...], Rejections]:
    """Parse the exposure editor, refusing anything that cannot be trusted.

    A blank row is not an error — it means "this node has no market exposure",
    which is the correct answer for most nodes. A *partially* filled row is an
    error, because a symbol with no anchor cannot be turned into a deviation and
    quietly skipping it would leave the user believing a mapping was active.
    """
    rejections = Rejections()
    known = set(str(n) for n in node_ids)
    exposures: list[NodeExposure] = []
    seen: set[tuple[str, str]] = set()

    if frame is None or frame.empty:
        return (), rejections

    if COL_EXP_NODE_ID not in frame.columns:
        rejections.add(
            "Exposure table", f"missing required column: {COL_EXP_NODE_ID}"
        )
        return (), rejections

    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        node_id = _text(row.get(COL_EXP_NODE_ID))
        symbol = _text(row.get(COL_EXP_SYMBOL)).upper()
        anchor = _number(row.get(COL_EXP_ANCHOR))
        elasticity = _number(row.get(COL_EXP_ELASTICITY))
        label = f"Exposure row {position}"

        if not node_id and not symbol and anchor is None and elasticity is None:
            continue
        if not node_id:
            rejections.add(label, "no node id")
            continue
        if node_id not in known:
            rejections.add(
                label, f"node '{node_id}' is not in the intervention table"
            )
            continue
        if not symbol:
            # Anchor or elasticity entered without a market is the signature of a
            # half-finished edit, and is reported rather than ignored.
            if anchor is not None or elasticity is not None:
                rejections.add(label, "market symbol missing")
            continue
        if anchor is None:
            rejections.add(
                label,
                f"'{symbol}' has no anchor price, so a deviation cannot be computed",
            )
            continue
        if anchor <= 0:
            rejections.add(label, f"anchor price must be positive, got {anchor:g}")
            continue
        if elasticity is None:
            rejections.add(
                label, f"'{symbol}' has no risk elasticity, so its effect is undefined"
            )
            continue
        if elasticity < 0:
            # A negative elasticity says a price rise makes the node safer. That may
            # be true of a genuine hedge, but it is a strong claim and the acquired
            # system produced it by accident via `- min(change, 0) * 0.5`, so it has
            # to be asked for explicitly rather than typed by mistake.
            rejections.add(
                label,
                f"risk elasticity must be non-negative, got {elasticity:g}; "
                "a market rise that reduces risk needs a stated hedge, not a sign flip",
            )
            continue
        if (node_id, symbol) in seen:
            rejections.add(label, f"duplicate exposure of '{node_id}' to '{symbol}'")
            continue

        seen.add((node_id, symbol))
        exposures.append(
            NodeExposure(
                node_id=node_id,
                symbol=symbol,
                anchor=float(anchor),
                elasticity=float(elasticity),
            )
        )

    return tuple(exposures), rejections


# --------------------------------------------------------------------------- #
# Delivery history and stored calibrations
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DeliveryRecord:
    """One node's realised delivery performance in one month.

    ``disruption_rate_pct`` is a percentage of shipments that went wrong, on whatever
    definition the business already uses — late, short, failed, diverted. The
    estimator is scale-free in it, so the definition need only be consistent across
    periods; it does not need to match anyone else's.
    """

    node_id: str
    period: str
    disruption_rate_pct: float


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    """A stored elasticity fit, as saved with the portfolio."""

    node_id: str
    symbol: str
    elasticity: float
    anchor: float
    interval_low: float | None
    interval_high: float | None
    r_squared: float | None
    observations: int
    periods_above_anchor: int
    method: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.node_id, self.symbol)


#: Tolerance for deciding whether the elasticity in the table is still the one that
#: was fitted. Not exact equality: the value survives a CSV round-trip on the way to
#: and from a saved portfolio, and a difference in the twelfth decimal is a
#: serialisation artefact rather than a user changing their mind. Six decimals is far
#: finer than any elasticity a fit can resolve, so a real edit always trips it.
CALIBRATION_MATCH_TOLERANCE = 1e-6


def default_history_frame() -> pd.DataFrame:
    """An empty delivery history sheet.

    No rows, unlike the exposure sheet's one-blank-row-per-node: history is a long
    table with many periods per node, and pre-seeding it with node names would invite
    a user to fill in one period per node and calibrate on a sample of one.
    """
    return pd.DataFrame({column: pd.Series(dtype="object") for column in HISTORY_COLUMNS})


def normalise_period(value: Any) -> str:
    """Reduce a period label to a ``YYYY-MM`` month key, or return ``""``.

    Accepts what people actually paste: ``2026-07``, ``2026-07-14``, ``07/2026``,
    ``Jul 2026``, a pandas timestamp. Monthly is the granularity market history is
    matched at, so a daily date is truncated to its month rather than rejected —
    but anything unparseable is refused rather than guessed, because a silently
    mis-parsed period joins a node's delivery record to the wrong month's price.
    """
    text = _text(value)
    if not text:
        return ""
    match = re.fullmatch(r"(\d{4})[-/](\d{1,2})(?:[-/]\d{1,2})?", text)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        return f"{year:04d}-{month:02d}" if 1 <= month <= 12 else ""
    match = re.fullmatch(r"(\d{1,2})[-/](\d{4})", text)
    if match:
        month, year = int(match.group(1)), int(match.group(2))
        return f"{year:04d}-{month:02d}" if 1 <= month <= 12 else ""
    try:
        stamp = pd.Timestamp(text)
    except Exception:  # noqa: BLE001 - unparseable is a rejection, not a crash
        return ""
    if pd.isna(stamp):
        return ""
    return f"{stamp.year:04d}-{stamp.month:02d}"


def align_history_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Give the history sheet its columns without touching its rows.

    Unlike ``align_exposure_frame``, this does not drop rows naming unknown nodes.
    Delivery history is a record of what happened; renaming an intervention is not a
    reason to delete evidence, and a user who renames a node back would otherwise
    find their history gone. Rows that cannot be attached to a node are reported at
    parse time instead, where the user can see and fix them.
    """
    if frame is None or frame.empty:
        return default_history_frame()
    trimmed = frame.copy()
    for column in HISTORY_COLUMNS:
        if column not in trimmed.columns:
            trimmed[column] = None
    return trimmed[list(HISTORY_COLUMNS)].reset_index(drop=True)


def build_delivery_records(
    frame: pd.DataFrame | None, node_ids: Sequence[str]
) -> tuple[tuple[DeliveryRecord, ...], Rejections]:
    """Parse the delivery history sheet, naming every row it cannot use.

    Duplicates are rejected rather than averaged. Two rows for the same node and
    month mean either a double paste or two incompatible definitions of disruption,
    and averaging them would produce a number belonging to neither.
    """
    rejections = Rejections()
    known = {str(n) for n in node_ids}
    records: list[DeliveryRecord] = []
    seen: set[tuple[str, str]] = set()

    if frame is None or frame.empty:
        return (), rejections

    missing = [c for c in (COL_HIST_NODE_ID, COL_HIST_PERIOD) if c not in frame.columns]
    if missing:
        rejections.add(
            "Delivery history", f"missing required column(s): {', '.join(missing)}"
        )
        return (), rejections

    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        node_id = _text(row.get(COL_HIST_NODE_ID))
        raw_period = row.get(COL_HIST_PERIOD)
        rate = _number(row.get(COL_HIST_DISRUPTION))
        label = f"History row {position}"

        if not node_id and not _text(raw_period) and rate is None:
            continue
        if not node_id:
            rejections.add(label, "no node id")
            continue
        if node_id not in known:
            rejections.add(
                label, f"node '{node_id}' is not in the intervention table"
            )
            continue
        period = normalise_period(raw_period)
        if not period:
            rejections.add(
                label,
                f"period {_text(raw_period)!r} is not a month this can read; "
                "use YYYY-MM",
            )
            continue
        if rate is None:
            rejections.add(label, f"{node_id} {period} has no disruption rate")
            continue
        if rate < 0:
            rejections.add(
                label, f"disruption rate cannot be negative, got {rate:g}"
            )
            continue
        if rate > 100:
            # Above 100% of shipments is a unit error — almost always a fraction
            # entered as a percentage of a percentage, or a raw incident count.
            rejections.add(
                label,
                f"disruption rate of {rate:g} is above 100%; this column is a "
                "percentage of shipments, not a count",
            )
            continue
        if (node_id, period) in seen:
            rejections.add(label, f"duplicate record for '{node_id}' in {period}")
            continue

        seen.add((node_id, period))
        records.append(
            DeliveryRecord(
                node_id=node_id, period=period, disruption_rate_pct=float(rate)
            )
        )

    return tuple(records), rejections


def default_calibration_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {column: pd.Series(dtype="object") for column in CALIBRATION_COLUMNS}
    )


def align_calibration_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return default_calibration_frame()
    trimmed = frame.copy()
    for column in CALIBRATION_COLUMNS:
        if column not in trimmed.columns:
            trimmed[column] = None
    return trimmed[list(CALIBRATION_COLUMNS)].reset_index(drop=True)


def build_calibration_records(
    frame: pd.DataFrame | None,
) -> tuple[tuple[CalibrationRecord, ...], Rejections]:
    """Parse stored fits, discarding any row that cannot vouch for a number.

    A stored calibration exists solely to justify an elasticity. A row missing its
    fitted value, anchor or method cannot do that, so it is dropped and reported
    rather than half-trusted — the consequence being that the elasticity it was
    vouching for reverts to ``asserted``, which is the safe direction to fail.
    """
    rejections = Rejections()
    records: list[CalibrationRecord] = []
    if frame is None or frame.empty:
        return (), rejections

    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        node_id = _text(row.get(COL_CAL_NODE_ID))
        symbol = _text(row.get(COL_CAL_SYMBOL)).upper()
        elasticity = _number(row.get(COL_CAL_ELASTICITY))
        anchor = _number(row.get(COL_CAL_ANCHOR))
        method = _text(row.get(COL_CAL_METHOD))
        label = f"Stored calibration row {position}"

        if not node_id and not symbol and elasticity is None:
            continue
        if not node_id or not symbol:
            rejections.add(label, "does not name both a node and a market")
            continue
        if elasticity is None or anchor is None or anchor <= 0:
            rejections.add(
                label, f"{node_id}/{symbol} has no usable fitted elasticity or anchor"
            )
            continue
        if not method:
            rejections.add(
                label,
                f"{node_id}/{symbol} does not record the method that produced it, "
                "so it cannot be used to call an elasticity calibrated",
            )
            continue

        observations = _number(row.get(COL_CAL_OBS))
        above = _number(row.get(COL_CAL_ABOVE))
        records.append(
            CalibrationRecord(
                node_id=node_id,
                symbol=symbol,
                elasticity=float(elasticity),
                anchor=float(anchor),
                interval_low=_number(row.get(COL_CAL_LOW)),
                interval_high=_number(row.get(COL_CAL_HIGH)),
                r_squared=_number(row.get(COL_CAL_R2)),
                observations=int(observations) if observations is not None else 0,
                periods_above_anchor=int(above) if above is not None else 0,
                method=method,
            )
        )

    return tuple(records), rejections


def resolve_elasticity_sources(
    exposures: Sequence[NodeExposure],
    calibrations: Sequence[CalibrationRecord],
) -> tuple[NodeExposure, ...]:
    """Label each exposure ``asserted`` or ``calibrated``, by evidence not by claim.

    An exposure counts as calibrated only when a stored fit for the same node and
    market records the *same* elasticity and the *same* anchor as the row now holds.
    That makes the label tamper-evident rather than declarative: overtype a
    calibrated elasticity and it silently becomes asserted again, which is exactly
    right, because a number a person changed is a number a person is asserting.

    The anchor is part of the match because the elasticity is defined relative to it.
    A fit against an anchor of 80 says nothing about deviations measured from 95, and
    keeping the old label after an anchor edit would carry the authority of a
    measurement across to a quantity that was never measured.
    """
    by_key = {record.key: record for record in calibrations}
    resolved: list[NodeExposure] = []
    for exposure in exposures:
        record = by_key.get((exposure.node_id, exposure.symbol))
        matched = (
            record is not None
            and abs(record.elasticity - exposure.elasticity) <= CALIBRATION_MATCH_TOLERANCE
            and abs(record.anchor - exposure.anchor) <= CALIBRATION_MATCH_TOLERANCE
        )
        source = f"calibrated ({record.method})" if matched and record else ASSERTED
        resolved.append(
            NodeExposure(
                node_id=exposure.node_id,
                symbol=exposure.symbol,
                anchor=exposure.anchor,
                elasticity=exposure.elasticity,
                elasticity_source=source,
            )
        )
    return tuple(resolved)
