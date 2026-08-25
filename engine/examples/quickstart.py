from scrcae import (
    Bundle,
    Dependency,
    Intervention,
    MonetaryNPVObjective,
    OptimizationRequest,
    PriceBook,
    SupplyNetwork,
    solve,
)
from scrcae.risk import AllocationConcaveResponse

network = SupplyNetwork(
    baseline_risk_pts=65.5,
    interventions=(
        Intervention(
            "N1", "Dual_Source_Ports", "Qualify a secondary port",
            cost=180_000.0, risk_reduction_pts=12.5,
            lead_time_saved_days=6.0, carbon_tons=40.0,
        ),
        Intervention(
            "N2", "Buffer_Warehouse", "Regional buffer stock",
            cost=240_000.0, risk_reduction_pts=9.0,
            lead_time_saved_days=11.0, carbon_tons=95.0,
        ),
        Intervention(
            "N3", "Supplier_Audit", "Tier-2 supplier audit programme",
            cost=70_000.0, risk_reduction_pts=4.25,
            lead_time_saved_days=1.5, carbon_tons=5.0,
        ),
    ),
    # N2 may never be funded beyond the scale N3 is funded at.
    dependencies=(Dependency(dependent="N2", prerequisite="N3"),),
    bundles=(
        Bundle(name="Port_Warehouse_Synergy", discount=50_000.0,
               required_nodes=("N1", "N2")),
    ),
)

prices = PriceBook(
    value_per_risk_point=50_000.0,     # currency per risk point removed, per year
    value_per_lead_time_day=2_000.0,   # currency per lead-time day removed, per year
    price_per_carbon_ton=100.0,        # internal carbon price
    benefit_horizon_years=5,
    discount_rate=0.10,
    source="illustrative-not-calibrated",
)

result = solve(
    OptimizationRequest(
        network=network,
        objective=MonetaryNPVObjective(prices=prices),
        risk_response=AllocationConcaveResponse(exponent=0.85),
        budget=400_000.0,
    )
)

print(f"status              {result.status}")
print(f"objective           {result.objective_value:,.0f}  [{result.objective_unit}]")
print(f"net capital         {result.net_capital:,.0f}"
      f" (gross {result.gross_capital:,.0f} - discounts {result.bundle_discounts:,.0f})")
print(f"active bundles      {result.active_bundles}")
print(f"risk {result.baseline_risk_pts:.2f} -> {result.optimized_risk_pts:.2f} pts"
      f"  (reduction {result.risk_reduction_pts:.2f})")
print(f"lead time saved     {result.total_lead_time_saved_days:.2f} days")
print(f"verification        {result.constraint_report.summary()}")
for a in result.active_allocations:
    print(f"  {a.node_id} {a.name:<20} scale {a.funding_scale:.3f}"
          f"  capital {a.capital:>10,.0f}  risk -{a.risk_reduction_pts:.2f} pts")
print(f"input hash          {result.audit['input_hash']}")
print(f"output hash         {result.audit['output_hash']}")
