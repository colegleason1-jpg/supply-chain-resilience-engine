from __future__ import annotations

import numpy as np
import pytest

from scrcae.domain import Bundle, Dependency, Intervention, SupplyNetwork
from scrcae.optimization import MonetaryNPVObjective, PriceBook


@pytest.fixture
def prices() -> PriceBook:
    """A price book with round numbers so expected values stay hand-checkable."""
    return PriceBook(
        value_per_risk_point=50_000.0,
        value_per_lead_time_day=2_000.0,
        price_per_carbon_ton=100.0,
        benefit_horizon_years=5,
        discount_rate=0.10,
        source="test-fixture",
    )


@pytest.fixture
def npv_objective(prices: PriceBook) -> MonetaryNPVObjective:
    return MonetaryNPVObjective(prices=prices)


@pytest.fixture
def small_network() -> SupplyNetwork:
    """Five nodes, one dependency, one bundle. Deliberately asymmetric costs and
    benefits so the solver has a unique optimum and tests do not hinge on how
    ties are broken."""
    return SupplyNetwork(
        baseline_risk_pts=65.5,
        interventions=(
            Intervention(
                "N1",
                "Dual_Source_Ports",
                "Qualify secondary port",
                cost=180_000.0,
                risk_reduction_pts=12.5,
                lead_time_saved_days=6.0,
                carbon_tons=40.0,
            ),
            Intervention(
                "N2",
                "Buffer_Warehouse",
                "Regional buffer stock",
                cost=240_000.0,
                risk_reduction_pts=9.0,
                lead_time_saved_days=11.0,
                carbon_tons=95.0,
            ),
            Intervention(
                "N3",
                "Supplier_Audit",
                "Tier-2 supplier audit programme",
                cost=70_000.0,
                risk_reduction_pts=4.25,
                lead_time_saved_days=1.5,
                carbon_tons=5.0,
            ),
            Intervention(
                "N4",
                "Rail_Shift",
                "Shift road freight to rail",
                cost=310_000.0,
                risk_reduction_pts=6.75,
                lead_time_saved_days=-0.0,
                carbon_tons=-220.0,
            ),
            Intervention(
                "N5",
                "Inbound_Visibility",
                "Inbound telemetry platform",
                cost=125_000.0,
                risk_reduction_pts=7.4,
                lead_time_saved_days=4.0,
                carbon_tons=12.0,
            ),
        ),
        dependencies=(Dependency(dependent="N2", prerequisite="N3"),),
        bundles=(
            Bundle(
                name="Port_Warehouse_Synergy",
                discount=50_000.0,
                required_nodes=("N1", "N2"),
            ),
        ),
    )


@pytest.fixture
def uniform_correlation(small_network: SupplyNetwork) -> np.ndarray:
    k = len(small_network)
    matrix = np.full((k, k), 0.35)
    np.fill_diagonal(matrix, 1.0)
    return matrix
