"""Supply-network domain model.

Extracted from the acquired Streamlit monolith. In the acquired system the
domain lived as loose parallel dicts (``costs``, ``risks``, ``lead_times``,
``marginal_risks``) built inline from a Streamlit ``data_editor`` DataFrame.
Here it is an explicit, validated, framework-free domain model.

NetworkX is deliberately absent: graph traversal is an implementation detail of
dependency analysis, not the domain representation itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator, Sequence

__all__ = [
    "Intervention",
    "Dependency",
    "Bundle",
    "SupplyNetwork",
    "DomainError",
]


class DomainError(ValueError):
    """Raised when a domain object violates its invariants."""


@dataclass(frozen=True, slots=True)
class Intervention:
    """A candidate resilience investment at one supply-network node.

    Attributes
    ----------
    node_id:
        Stable identifier. Must be unique within a network.
    name:
        Human-readable node name.
    action:
        Description of the intervention performed at this node.
    cost:
        Full capital cost, in currency units, of funding this intervention at
        100% scale.
    risk_reduction_pts:
        Reduction in system risk, in *percentage points of baseline system
        risk*, delivered by funding at 100% scale.

        This is an internal model input. It is not a probability and not a
        financial loss estimate. The acquired system carried this same caveat
        and it is preserved deliberately.
    lead_time_saved_days:
        Reduction in lead time, in days, at 100% scale.
    carbon_tons:
        Change in carbon footprint, in tonnes, at 100% scale. Negative values
        denote avoided emissions.
    min_funding_scale:
        Minimum economic scale (MES). If the intervention is funded at all, it
        must be funded at least this fraction. The acquired system hard-coded
        this at 0.3 for every node; it is now per-node and explicit.
    max_funding_scale:
        Maximum fraction of ``cost`` that may be deployed.
    """

    node_id: str
    name: str = ""
    action: str = ""
    cost: float = 0.0
    risk_reduction_pts: float = 0.0
    lead_time_saved_days: float = 0.0
    carbon_tons: float = 0.0
    min_funding_scale: float = 0.3
    max_funding_scale: float = 1.0

    def __post_init__(self) -> None:
        if not str(self.node_id).strip():
            raise DomainError("Intervention.node_id must be a non-empty string")
        if self.cost < 0:
            raise DomainError(f"{self.node_id}: cost must be non-negative, got {self.cost}")
        if self.risk_reduction_pts < 0:
            raise DomainError(
                f"{self.node_id}: risk_reduction_pts must be non-negative, "
                f"got {self.risk_reduction_pts}"
            )
        if self.lead_time_saved_days < 0:
            raise DomainError(
                f"{self.node_id}: lead_time_saved_days must be non-negative, "
                f"got {self.lead_time_saved_days}"
            )
        if not 0.0 <= self.min_funding_scale <= 1.0:
            raise DomainError(
                f"{self.node_id}: min_funding_scale must lie in [0, 1], "
                f"got {self.min_funding_scale}"
            )
        if not 0.0 < self.max_funding_scale <= 1.0:
            raise DomainError(
                f"{self.node_id}: max_funding_scale must lie in (0, 1], "
                f"got {self.max_funding_scale}"
            )
        if self.min_funding_scale > self.max_funding_scale:
            raise DomainError(
                f"{self.node_id}: min_funding_scale ({self.min_funding_scale}) exceeds "
                f"max_funding_scale ({self.max_funding_scale})"
            )

    @property
    def display_name(self) -> str:
        return self.name or self.node_id


@dataclass(frozen=True, slots=True)
class Dependency:
    """A prerequisite relationship between two interventions.

    Semantics preserved from the acquired system: the dependent node's funding
    scale may not exceed the prerequisite's funding scale
    (``x_dependent <= x_prerequisite``). This is a *cascade cap*, strictly
    stronger than "the prerequisite must be active".
    """

    dependent: str
    prerequisite: str

    def __post_init__(self) -> None:
        if self.dependent == self.prerequisite:
            raise DomainError(
                f"Dependency cannot be self-referential (node {self.dependent!r})"
            )


@dataclass(frozen=True, slots=True)
class Bundle:
    """A synergy discount unlocked when every required node is activated.

    In the acquired system the required-node list came from a Streamlit
    ``multiselect`` widget that silently defaulted to ``nodes[:2]`` — meaning
    the economics of a bundle depended on widget render order. Requirements are
    now explicit data.
    """

    name: str
    discount: float
    required_nodes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise DomainError("Bundle.name must be a non-empty string")
        if self.discount < 0:
            raise DomainError(f"Bundle {self.name!r}: discount must be non-negative")
        if not self.required_nodes:
            raise DomainError(
                f"Bundle {self.name!r}: required_nodes must not be empty. The acquired "
                "system defaulted to the first two nodes; that behaviour is not "
                "reproduced because it made bundle economics order-dependent."
            )


@dataclass(frozen=True, slots=True)
class SupplyNetwork:
    """The full decision problem's structural inputs.

    ``baseline_risk_pts`` is the pre-intervention system risk in percentage
    points. Interventions reduce it; total reduction is capped at the baseline.
    """

    interventions: tuple[Intervention, ...]
    baseline_risk_pts: float
    dependencies: tuple[Dependency, ...] = ()
    bundles: tuple[Bundle, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.interventions:
            raise DomainError("SupplyNetwork requires at least one intervention")
        if not 0.0 <= self.baseline_risk_pts <= 100.0:
            raise DomainError(
                f"baseline_risk_pts must lie in [0, 100], got {self.baseline_risk_pts}"
            )

        ids = [i.node_id for i in self.interventions]
        duplicates = {n for n in ids if ids.count(n) > 1}
        if duplicates:
            raise DomainError(f"Duplicate node_id values: {sorted(duplicates)}")

        known = set(ids)
        by_id = {i.node_id: i for i in self.interventions}
        for dep in self.dependencies:
            missing = {dep.dependent, dep.prerequisite} - known
            if missing:
                raise DomainError(
                    f"Dependency references unknown node(s): {sorted(missing)}"
                )
        for bundle in self.bundles:
            missing = set(bundle.required_nodes) - known
            if missing:
                raise DomainError(
                    f"Bundle {bundle.name!r} references unknown node(s): {sorted(missing)}"
                )

            # A discount larger than the cost of the nodes it applies to is not a
            # discount, it is an income stream. The acquired system placed no bound
            # on the field, so a mistyped figure let the optimizer fund a portfolio
            # at zero or negative net cost and report the budget constraint as
            # satisfied. Property testing found exactly this: two nodes costing
            # 1,000 each, a 2,000 discount, and a budget of zero produced a fully
            # funded plan. The bound is checked at the bundle's own gross cost at
            # full funding, which is the most capital the discount could ever
            # legitimately offset.
            bundle_gross = sum(
                by_id[node].cost * by_id[node].max_funding_scale
                for node in bundle.required_nodes
            )
            if bundle.discount >= bundle_gross:
                raise DomainError(
                    f"Bundle {bundle.name!r}: discount {bundle.discount:.6g} exceeds the "
                    f"gross cost {bundle_gross:.6g} of its required nodes at full "
                    "funding. A discount that cancels or outruns the spend it applies "
                    "to makes the portfolio free, which empties the budget constraint "
                    "of meaning: the optimizer would fund the bundle at a budget of "
                    "zero. Vendor synergy is a fraction of cost, not all of it."
                )

        cycle = _find_dependency_cycle(ids, self.dependencies)
        if cycle:
            raise DomainError(
                "Dependency graph contains a cycle: " + " -> ".join(cycle)
            )

    # -- access -------------------------------------------------------------

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(i.node_id for i in self.interventions)

    def __iter__(self) -> Iterator[Intervention]:
        return iter(self.interventions)

    def __len__(self) -> int:
        return len(self.interventions)

    def __getitem__(self, node_id: str) -> Intervention:
        for intervention in self.interventions:
            if intervention.node_id == node_id:
                return intervention
        raise KeyError(node_id)

    def total_cost_at_full_scale(self) -> float:
        return sum(i.cost for i in self.interventions)

    def with_interventions(self, interventions: Iterable[Intervention]) -> SupplyNetwork:
        """Return a copy carrying replaced interventions.

        Used by adjustment layers (macro, market exposure) which must never
        mutate the network in place.
        """
        return replace(self, interventions=tuple(interventions))

    def with_baseline_risk(self, baseline_risk_pts: float) -> SupplyNetwork:
        return replace(self, baseline_risk_pts=baseline_risk_pts)


def _find_dependency_cycle(
    node_ids: Sequence[str], dependencies: Sequence[Dependency]
) -> list[str] | None:
    """Return a cycle as a node path, or None when the graph is acyclic."""
    edges: dict[str, list[str]] = {n: [] for n in node_ids}
    for dep in dependencies:
        edges[dep.dependent].append(dep.prerequisite)

    WHITE, GREY, BLACK = 0, 1, 2
    colour = {n: WHITE for n in node_ids}
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        colour[node] = GREY
        stack.append(node)
        for nxt in edges[node]:
            if colour[nxt] == GREY:
                return stack[stack.index(nxt):] + [nxt]
            if colour[nxt] == WHITE:
                found = visit(nxt)
                if found:
                    return found
        stack.pop()
        colour[node] = BLACK
        return None

    for node in node_ids:
        if colour[node] == WHITE:
            found = visit(node)
            if found:
                return found
    return None
