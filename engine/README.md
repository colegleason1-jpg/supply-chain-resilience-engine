# scrcae — Supply Chain Resilience & Capital Allocation Engine

## 1. What this is and why it exists

`scrcae` is the headless core of the acquired supply-chain resilience product. The
thing we bought was a single 1,146-line Streamlit script. All of its value — a
mixed-integer capital allocation optimizer, a correlated Monte Carlo tail-risk
simulation, a bundle/dependency model — was real, and all of it was welded to the
UI. Coefficients were computed inline from a `data_editor` DataFrame, bundle
requirements were read out of widget state mid-solve, the Monte Carlo called
`np.random.seed(42)` on the process-global generator, and the optimizer existed
twice in subtly divergent copies (once for the live model, once inside the budget
sensitivity sweep). None of it could be tested, because there was nothing to call.

This package is that math, extracted, with the framework removed and the inputs
made explicit. Three consequences are the whole point:

**Headless.** Every entry point is a function over dataclasses. `solve(request)`
and `run(request)` take plain domain objects and return plain result objects.
There is no Streamlit import anywhere in `src/scrcae/`, and no NetworkX either —
graph traversal is an implementation detail of dependency validation, not a
representation of the domain.

**Testable.** 197 tests run against the engine, including known-answer cases whose
optima can be derived by hand, and Hypothesis property tests that generate
networks and assert the engine never violates the constraints it claims to
enforce. Several of the bugs documented in the source comments were found by those
property tests, not by reading: a bundle discount larger than the cost of the nodes
it applied to let the optimizer fund a portfolio at a budget of zero, and a single
shared tangent variable let the risk cap stop binding exactly when the objective
stopped caring about risk.

**Auditable.** The acquired system printed `sha256:8f4c99a...` in its equation
ledger and called it a verification hash. It was a string literal and it hashed
nothing. It also displayed "Verification: zero fractional violations detected",
which was likewise a literal that nothing computed. Here, `build_audit_record`
hashes the actual network, objective, response model and parameters, and
`ConstraintReport` independently re-checks every structural constraint against the
solver's returned allocation vector, at stated tolerances, and can fail.

The UI is now a client of the engine. It renders `OptimizationResult` and
`SimulationResult`; it does not host the model. `OptimizationResult.to_records()`
exists precisely so a DataFrame, a CSV export and an API response are all the same
one-line call.

## 2. Install and dev setup

Requires Python 3.11 or newer. Runtime dependencies are `numpy>=1.26` and
`pulp>=2.8`; test extras are `pytest>=8` and `hypothesis>=6` (see
`pyproject.toml`).

```bash
cd engine
python -m pip install -e ".[test]"
```

Running the tests:

```bash
cd engine                       # must be the working directory
python -m pytest -p no:warnings -q    # python -m pytest, not bare pytest
```

Three things about that command are not incidental:

- `pytest.ini` sets `pythonpath = src`, so tests import `scrcae` from the source
  tree without an install step. That only works if pytest is started from
  the `engine/` directory, because `pythonpath` is resolved relative to the rootdir.
- Use `python -m pytest`. The `pytest` console script is not on `PATH` in this
  environment; invoking it directly fails before it reaches the config.
- `pytest.ini` also sets `--strict-markers` and declares one marker, `slow`, for
  benchmarks and high-iteration runs.

Solver: **CBC via PuLP only** (`pl.PULP_CBC_CMD`). The optimizer passes
`primalTolerance 1e-9`, `integerTolerance 1e-9` and `ratioGap 0`, because CBC's
default primal feasibility tolerance is absolute and on nine-figure currency
coefficients it will happily return a solution that overspends a budget by a
fraction of a cent. Tightening narrows that gap; `_verify` closes it by measuring
the residual slack explicitly instead of assuming it is zero. No other solver is
wired up, and no solver-selection abstraction exists.

## 3. Quickstart

This is `examples/quickstart.py` at the engine root, verbatim. It runs.

```python
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
```

Run it with `PYTHONPATH=src python examples/quickstart.py` from the `engine/` directory. Real
output:

```
status              optimal
objective           3,988,083  [currency (net present value)]
net capital         400,000 (gross 400,000 - discounts 0)
active bundles      ()
risk 65.50 -> 42.71 pts  (reduction 22.79)
lead time saved     14.38 days
verification        6 constraint families verified, 0 violations (tolerance 1e-06)
  N1 Dual_Source_Ports    scale 1.000  capital    180,000  risk -12.50 pts
  N2 Buffer_Warehouse     scale 0.625  capital    150,000  risk -6.04 pts
  N3 Supplier_Audit       scale 1.000  capital     70,000  risk -4.25 pts
input hash          sha256:2aab6f0a8c0e0b713501c9e7b9408418ad4362300d69a9a04b68b6d51ec35e2f
output hash         sha256:522886e34b1ecc7b9190d843325c69ef19cfcee5aacf038775035869e0423c6c
```

Worth reading off that output: the objective is in currency, so 3,988,083 is a net
present value you can put next to a business case rather than a dimensionless
score. The bundle is *not* taken — the discount requires both N1 and N2 funded in
full and the budget cannot reach that, so the engine declines a discount the
acquired system would have claimed at N2's 30% minimum. The dependency binds
N2 ≤ N3. Both hashes are stable across runs; the whole script's output is
byte-identical run to run.

The Monte Carlo stage consumes the optimizer's ground-truth per-node reductions:

```python
from scrcae import LognormalShock, SimulationRequest, UniformCorrelation
from scrcae.stochastic import run

order = network.node_ids
sim = run(
    SimulationRequest(
        baseline_risk_pts=network.baseline_risk_pts,
        risk_reduction_by_node={
            a.node_id: a.risk_reduction_pts for a in result.allocations
        },
        correlation=UniformCorrelation(rho=0.35).matrix(order),
        node_order=order,
        iterations=20_000,
        seed=42,
        shock_model=LognormalShock(sigma=0.12),
    )
)
print(sim.summary())
print(f"P10 {sim.p10_risk_pts:.2f}  ES90 {sim.expected_shortfall_90_pts:.2f}  "
      f"tail spread {sim.tail_spread_pts:.2f}")
print(sim.correlation_repair.summary())
```

```
P50 42.82% (SE 0.019), P90 45.41% (SE 0.022), deterministic 42.71%, bias -0.001 pts, N=20,000, seed=42
P10 39.84  ES90 46.28  tail spread 2.59
matrix was already positive definite (min eigenvalue 0.65)
```

The reported bias of −0.001 points is the check that matters: with a mean-one
shock model the simulated centre sits on the deterministic answer to within Monte
Carlo error, so P50 and the deterministic figure are legitimately comparable.

## 4. Architecture

**`audit.py`** — provenance. `canonical_json` gives a stable encoding (sorted keys,
floats rounded to 12 significant decimals so platform formatting cannot change a
hash for numerically identical inputs), `content_hash` SHA-256s it, and
`build_audit_record` assembles the record attached to every optimization result.
Also owns `ENGINE_VERSION`.

**`domain/network.py`** — the validated, framework-free domain model:
`Intervention`, `Dependency`, `Bundle`, `SupplyNetwork`, all frozen dataclasses
that raise `DomainError` on construction rather than producing a nonsense solve.
This is where the acquired system's loose parallel dicts (`costs`, `risks`,
`lead_times`, `marginal_risks`) became one object with invariants: unique node
ids, funding scales in range, `min_funding_scale <= max_funding_scale`,
dependencies and bundles referencing real nodes, an acyclic dependency graph, and
a bundle discount strictly below the gross cost of its required nodes at full
funding. `with_interventions` and `with_baseline_risk` return copies, so adjustment
layers can never mutate a network in place.

**`calibration/elasticity.py`** — fitting a market risk elasticity from a company's own
delivery history, so the market-exposure multiplier can be measured rather than
asserted. The model is `disruption_rate = f0 · (1 + e · max((price − anchor)/anchor, 0))`,
which is plain OLS with an intercept where `e = slope/intercept`; it is scale-free in the
disruption column, so percentages and fractions give the same answer. The one-sidedness
is not a statistical choice but the same asymmetry the app's runtime policy already
states. `calibrate_elasticity` returns an `ElasticityFit` carrying a moving-block
bootstrap interval (block length `n^(1/3)`, because delivery periods are serially
correlated and textbook standard errors are too narrow), and a `FitQuality` that is a
*named* refusal rather than a boolean — "you have four periods" and "your data show
nothing" call for opposite actions. Every fit, including a refused one, carries the
three standing caveats in its `notes`: it is an association and not a cause, delivery
records measure the risk a node carries rather than the share an intervention removes,
and the interval is approximate. The module documents its own measured coverage.

**`risk/response.py`** — how funding scale becomes risk reduction, as a
`RiskResponseModel` protocol with three implementations. The acquired system
computed `(macro * risk) ** 0.85` per node and then multiplied linearly by `x`,
which is `ParameterPowerResponse` here. That is *linear in funding*: funding a node
at 50% delivers exactly half of full funding, so the exponent produces no
diminishing returns at all, only a reweighting between nodes that nobody ever
articulated. `LinearResponse` is the honest baseline.
`AllocationConcaveResponse` applies the exponent to the allocation
(`(macro * R) * x ** alpha`), which is what the acquired ledger claimed the 0.85
was doing; being non-linear it exposes a `tangents()` outer envelope for the
optimizer. `check_scale` rejects negative scales, which under a linear response
would otherwise silently return negative risk reduction.

**`optimization/objectives.py`** — the objective models and the unit fix. The
acquired system maximised `sum_n [w * R*_n + (1 - w) * L_n] * x_n`, adding
percentage points to days. `w` therefore encoded both a preference and an
undefined unit conversion, absorbed the scale of whatever dataset it was tuned on,
and made the argmax an artefact of which coefficient family carried bigger raw
numbers. `MonetaryNPVObjective` prices every term in currency via a `PriceBook`
(value per risk point, per lead-time day, internal carbon price, horizon, discount
rate), annuitises benefits over the horizon and subtracts capital, so the
objective is an NPV — comparable across scenarios, and negative when a portfolio
destroys value. Optional `strategic_*_multiplier` values let someone deliberately
over-weight resilience on top of a coherent monetary base. `MinimizeCapitalObjective`
is target-risk mode, which was already unit-coherent. `LegacyWeightedObjective` is
retained and self-identifies: its `unit` is the literal string
`"incommensurate (percentage-points + days)"`.

**`optimization/optimizer.py`** — one `solve(OptimizationRequest)` function
replacing the two divergent copies. Preserves the acquired structure (binary
activation `y`, continuous scale `x`, minimum economic scale
`min*y <= x <= max*y`, prerequisite cascade `x_dep <= x_pre`, bundle activation,
net budget cap, target mode) and fixes four things. The baseline risk cap is
enforceable in-model instead of clipped afterwards. A non-linear response is
linearised in *two* directions — inner chords for the objective and the
required-reduction floor, so the model never over-promises, and outer tangents
with a big-M release for the cap, so the cap never binds against an under-report.
Bundle discounts default to requiring full funding (`b <= x`) rather than mere
activation. And every reported quantity is recomputed from
`response.evaluate(...)`, never read back from the linearisation, so
`tangent_points` affects the precision of the chosen allocation and never the
accuracy of the numbers attributed to it.

**`optimization/result.py`** — `OptimizationResult`, `NodeAllocation`,
`SolveStatus`, and the verifier's output. `ConstraintReport` records which
constraint families were checked, any `Violation`s, and `residual_slack`: the
largest overshoot that fell *within* tolerance, per check. That number is kept
rather than discarded because a plan can be feasible to solver tolerance while
still overspending a budget by a measurable amount, and whoever signs the capital
plan is entitled to see it. `risk_cap_was_binding` flags a run where the optimizer
bought risk reduction beyond the baseline and would otherwise have reported that
spend as productive.

**`stochastic/correlation.py`** — correlation construction and repair. The acquired
system restored positive definiteness with `corr += I * (-min_eig + 1e-5)`, which
breaks the unit diagonal and turns the correlation matrix into a covariance matrix
with inflated, unequal variances; Cholesky then widens every node's shock by
`sqrt(1 + ridge)`, so tail risk worsened because the input matrix was awkward.
`repair_to_correlation` clips the spectrum and renormalises the diagonal back to
one, returning a `RepairReport` with the eigenvalues before and after and the
Frobenius shift. `naive_ridge_repair` is kept so the defect can be demonstrated
numerically in a test. `UniformCorrelation` rejects an `rho` below `-1/(k-1)`
instead of quietly handing an impossible matrix to the repair step, which would
return a different correlation structure than the one requested.

**`stochastic/shock.py`** — how a correlated normal becomes a delivery multiplier.
`LognormalShock` uses `exp(sigma*Z - sigma^2/2)`: strictly positive, so no floor
is needed, and exactly mean-one for any sigma, so the simulation is centred on the
deterministic result by construction. `LegacyTruncatedNormalShock` reproduces the
acquired `max(0.4, 1 + 0.12*Z)` and exposes `analytic_mean` and `analytic_bias`
closed forms, so the upward bias from truncating only the left tail is a number
rather than an argument. `DeterministicShock` returns 1.0 and exists to prove the
simulation collapses onto the deterministic answer when uncertainty is off.

**`stochastic/montecarlo.py`** — `run(SimulationRequest)`. Takes an explicit
`seed` into a local `numpy.random.Generator` instead of mutating the global RNG,
draws one `(iterations, k)` block and uses matrix products instead of a Python
loop, clips the deterministic and simulated paths identically, and reports
percentile standard errors alongside P50/P90/P10, mean, standard deviation and
90% expected shortfall — so a user can tell whether their iteration count
supports the precision being displayed. `reduction_bias_pts` surfaces any drift
between the simulated centre and the deterministic answer.

**`legacy/reference.py`** — a faithful headless transcription of the acquired
optimizer and Monte Carlo, hard-coded 0.3 minimum scale, `np.random.seed(42)`,
0.4 floor and all. Nothing here is for production use. It is the reference the
parity tests diff against, and the fixture that lets a historic number be
reproduced when a customer asks why last quarter's answer changed.

## 5. Two modes

### Monetary NPV mode (recommended)

`MonetaryNPVObjective` + a `PriceBook`, `AllocationConcaveResponse` or
`LinearResponse`, `LognormalShock`, `enforce_risk_cap=True` (the default), bundle
discounts gated on full funding (the default). Section 3 is exactly this. Prices,
not weights; currency, not a score; the risk cap enforced inside the model rather
than applied to the answer afterwards.

### Legacy parity mode

```python
solve(
    OptimizationRequest(
        network=network,          # interventions with min_funding_scale=0.3
        objective=LegacyWeightedObjective(weight=0.65),
        risk_response=ParameterPowerResponse(exponent=0.85),
        budget=400_000.0,
        enforce_risk_cap=False,
        legacy_bundle_activation=True,
    )
)
```

Plus `LegacyTruncatedNormalShock(sigma=0.12, floor=0.4)` for the simulation stage.
`min_funding_scale=0.3` is the field default on `Intervention`, so a network built
without overriding it is already at the legacy value; the acquired system
hard-coded 0.3 for every node and now it is per-node and explicit.

This mode exists so that "we rewrote the engine" and "we changed the answers"
remain two separate claims. Every behavioural difference the new engine
introduces should be a difference someone chose; parity tests are what make an
*unchosen* difference visible, and they are the reason each legacy behaviour was
preserved as a named, versioned, opt-in class rather than deleted. `test_parity.py`
runs legacy mode against `legacy/reference.py` across a weight-and-budget grid, a
macro shock, a binding budget, target mode, and the Monte Carlo distribution, and
asserts agreement. One test — `test_the_only_intended_divergences_are_opt_in` — is
specifically there to catch a divergence nobody signed off on.

Results from this mode label themselves. `objective_unit` is
`"incommensurate (percentage-points + days)"`, which lands in the audit record, so
a legacy-mode number cannot be quietly mistaken for a monetary one.

## 5b. Unreachable targets

Ask for a risk target the portfolio cannot hit and the solver says `infeasible`. That
status is correct and the engine keeps it, in the result and in the audit ledger. It
is also, on its own, misleading: the acquired app ran target mode with the budget
pinned to 1e9, so an unreachable target looked like a *funding* shortfall. Its own
shipped default (`target_risk_goal = 20.0`) is unreachable at any price.

So every infeasible result now carries a diagnosis:

```python
result = solve(OptimizationRequest(..., required_risk_reduction_pts=49.6))

result.status                             # 'infeasible'  -- unchanged
result.diagnosis.kind                     # 'exceeds_structural_ceiling'
result.diagnosis.more_capital_would_help  # False
result.diagnosis.shortfall_pts            # 19.1
print(result.explanation)
```

```
A target of 20.0% risk is not reachable with the current intervention portfolio.
Funding every intervention to its maximum delivers 30.5 points of reduction, which
takes risk from 69.6% only as low as 39.1% at a cost of 925,000. The request is
short by 19.1 points. This is not a budget constraint: additional capital does not
change it. Reaching the target requires interventions that are not currently on the
table, or a revised target.
```

The same request with a real budget and a reachable target produces the other
diagnosis, `exceeds_budget`, with `capital_required` set to the exact minimum needed —
so "we need more money" and "we need more options" are never confused for each other.

Use `attainable_frontier(network, response)` directly if you only want the ceiling.
It is solved rather than summed, so dependencies and per-node funding caps are
respected, and it uses the inner linear approximation, so it can understate the
ceiling but never overstate it.

Two things it will not do. It will not relax your target: the attainable figure is
reported beside the request, never in place of it. And it will not guess — if no check
explains the failure, `kind` is `undiagnosed` and the text says so.

`OptimizationResult.explanation` is the string to render in a UI for any status; it
never contains the word "infeasible".

## 6. The audit record

Every `OptimizationResult` carries `result.audit`, a mapping with these keys:

| Key | Contents |
| --- | --- |
| `engine_version` | `ENGINE_VERSION`, currently `"0.1.0"` |
| `generated_at` | UTC ISO-8601 timestamp |
| `python_version` | interpreter version |
| `solver` | `"PULP_CBC_CMD"` |
| `objective_version` | e.g. `"objective/monetary-npv@1.0.0"` |
| `objective_unit` | e.g. `"currency (net present value)"` |
| `objective_parameters_hash` | content hash of the objective, price book included |
| `risk_response_version` | e.g. `"risk-response/allocation-concave@1.0.0"` |
| `risk_response_parameters_hash` | content hash of the response model |
| `network_hash` | content hash of the `SupplyNetwork` |
| `parameters` | budget, `required_risk_reduction_pts`, `macro_multiplier`, `node_macro_multipliers`, `enforce_risk_cap`, `tolerance`, `tangent_points`, `relative_tolerance`, `legacy_bundle_activation` |
| `objective_provenance` | Every price in force, the horizon, the discount rate, the annuity factor, the stated source, and whether that source counts as stated. Recorded outside the hash inputs, so adding it does not change any historic hash. |
| `parameters_hash` | content hash of the above |
| `solver_status` | mapped `SolveStatus` |
| `linearisation_used` | whether a non-linear response was linearised |
| `constraint_check` | `ConstraintReport.summary()` |
| `constraint_violations` | stringified violations, empty when clean |
| `input_hash` | hash over the four component hashes |
| `output_hash` | hash over scales, gross capital, discounts and raw risk reduction |

`input_hash` and `output_hash` together tie a result to the exact data,
parameters, model versions and seed that produced it, which is what the acquired
system's literal `sha256:8f4c99a...` badge was pretending to do.

The uncalibrated constants are the reason the version and hash fields matter.
`ParameterPowerResponse`, `AllocationConcaveResponse` and `LognormalShock` each
carry a `calibration_source` field, defaulting to `"uncalibrated-assumption"`.
`LinearResponse` carries `"not-applicable-no-free-parameters"`.
`UniformCorrelation` carries `source="uncalibrated-default"` alongside the
acquired system's unexplained `rho=0.35`. `PriceBook` carries `source`,
defaulting to `"unstated"`. Because these are ordinary dataclass fields, they are
inside the content hash of the object that holds them: changing a
`calibration_source` changes `risk_response_parameters_hash` or
`objective_parameters_hash`, and therefore `input_hash`. The point is that a
number like 0.85 or 0.35 is no longer a literal buried in an expression where it
reads as data. It is a named, versioned parameter that declares it has no
empirical basis, so the assumption shows up in the ledger, and a future
calibration exercise can replace it without touching the optimizer.

## 7. Testing

672 tests, all passing — 197 covering the engine and 475 covering the Streamlit app
that consumes it. Run both suites from the repository root:

```
$ python -m pytest -p no:warnings -q
........................................................................ [ 16%]
........................................................................ [ 32%]
........................................................................ [ 48%]
........................................................................ [ 64%]
........................................................................ [ 81%]
........................................................................ [ 97%]
............                                                             [100%]
444 passed in 35.49s
```

The root `pytest.ini` puts both `.` and `engine/src` on the path, so the app tests
import the real engine rather than a mock. Almost none of them import Streamlit: every
decision the interface makes lives in `app/adapters.py` and `app/presenters.py`,
which are ordinary functions over frames and result objects. `app/views/` and
`app/main.py` contain layout only, so there is nothing there a test would need to
assert. To run the engine suite alone:

```
$ cd engine && python -m pytest -p no:warnings -q
```

Shared fixtures live in `tests/conftest.py`: a `PriceBook` with round numbers so
expected values stay hand-checkable, and a five-node `small_network` with
deliberately asymmetric costs and benefits so the optimum is unique and no test
depends on how ties are broken.

**`tests/test_known_answer.py` — 12 tests.** Hand-derivable optima. Catches
regressions in constraint construction and objective assembly, and does so with
an answer a human can adjudicate: a binary knapsack, a budget-limited partial
funding, a budget below minimum economic scale funding nothing, dependency
cascade forcing equal funding, a bundle unlocking full funding and a bundle
unable to claim its discount without its nodes, target mode buying the cheapest
route to a floor and reporting an unreachable target as infeasible, the risk cap
preventing purchase of unusable reduction, and the concave response spreading
funding where the linear one does not. Without this anchor, a modelling change and
a bug look the same.

**`tests/test_objective_units.py` — 11 tests.** Unit coherence of the objective.
The headline case relabels lead-time savings from days to years — a pure unit
change that alters nothing about the business — and shows the legacy weighted
objective reverses its selected portfolio while the monetary objective is
invariant once the price is expressed in the matching unit. Also covers the
annuity factor against its closed form and its monotonicity in horizon and rate,
`PriceBook.from_annual_exposure`, strategic multipliers, the carbon price
penalising emitting interventions, the fact that the legacy objective always
spends the budget while the monetary one declines value-destroying portfolios, and
rejection of negative prices and out-of-range weights.

**`tests/test_risk_response.py` — 26 tests.** The response models and the
linearisation machinery. Proves `ParameterPowerResponse` is linear in funding and
only reweights between nodes, that `AllocationConcaveResponse` has genuine
diminishing returns with decreasing marginal returns and strict concavity, that
zero funding yields zero reduction and reduction is monotone for every model, that
negative funding is rejected, and that exponent bounds are enforced. The tangent
tests are the load-bearing ones for the optimizer: the envelope is a valid outer
approximation, tangent intercepts are positive (which is *why* the
`risk_ceiling_{n}` constraint is not redundant), slopes decrease along the curve,
and the envelope tightens as breakpoints are added. One test asserts the
uncalibrated exponents declare themselves.

**`tests/test_stochastic.py` — 25 tests.** Three forensic findings plus the
simulation contract. The 0.4 floor's upward bias is checked against its closed
form and shown to grow by orders of magnitude as sigma is calibrated up; the
legacy shock is shown to understate risk where the lognormal does not; the legacy
normal multiplier is shown to go negative without its floor. Ridge repair is shown
to inflate variances while eigenvalue repair preserves the unit diagonal and
reproduces the requested correlation far better. The rest is the contract: same
seed gives bit-identical samples, different seeds give similar statistics, the
global random state is untouched, P90 is never below P50, samples stay within a
valid percentage range, standard errors shrink with iterations, higher correlation
widens the tail, higher sigma widens the tail without moving the centre, an empty
portfolio returns the baseline with no dispersion, and invalid requests are
rejected.

**`tests/test_parity.py` — 35 tests.** Legacy equivalence against
`legacy/reference.py`. Marginal risk coefficients match exactly; commercial mode
matches across a weight-and-budget grid, under a macro shock and with a binding
budget; target mode matches at reachable targets; the legacy Monte Carlo is
reproducible in isolation and the new engine in legacy shock mode reproduces its
distribution. Two tests are findings rather than parity checks: target mode is
infeasible at the acquired system's own shipped default target, and the exponent
transform is what caps attainable reduction. Two more guard the boundary — the
legacy Monte Carlo pollutes the global RNG while the new one does not, the only
divergences are opt-in, and legacy results are labelled incommensurate.

**`tests/test_node_exposure.py` — 23 tests.** Per-node macro multipliers
(`OptimizationRequest.node_macro_multipliers`), the engine half of F11's
replacement. The multiplier reaches every one of the five places the scalar
`macro_multiplier` is applied — linear coefficient, structural ceiling, concave
chords, tangents, and the final evaluation — so a linearised concave response cannot
be adjusted in the objective but not in its own envelope. Also pinned: an unmapped
node receives the scalar exactly; an explicit `1.0` is indistinguishable from
omission, including in the input hash; insertion order does not change the hash;
exposure can change *which* portfolio is chosen, not merely its reported value (a
node unfunded at base becomes funded at 8x); unknown node ids raise and every one of
them is named; zero, negative, infinite and NaN multipliers are refused, because a
zero deletes an intervention's benefit while keeping its full cost; the budget sweep
inherits exposure (an F6 regression guard); and — deliberately — a per-node
multiplier does **not** move the reported baseline, while the scalar still does.

**`tests/test_properties.py` — 17 tests.** Hypothesis-generated networks,
budgets, dependency chains and bundles. Catches the invariant violations nobody
thought to write a case for: every solved portfolio satisfies every constraint,
the budget is never exceeded, minimum economic scale is all-or-nothing, a
dependent never outruns its prerequisite, a bundle discount requires all of its
nodes, capped risk reduction never exceeds the baseline, reported reduction equals
the ground-truth response, more budget is never worse, a zero budget funds
nothing, `solve` is deterministic, and residual slack is always reported and
within tolerance. Domain invariants are covered here too (invalid interventions,
dependency cycles, dangling references, duplicate ids), as are the simulation
output being a valid risk distribution and correlation repair always yielding a
usable matrix. This is the file that found the zero-budget bundle exploit and the
non-binding risk cap.

**`tests/test_elasticity_calibration.py` — 30 tests.** The estimator in
`calibration/elasticity.py`, and the interval it reports. Every refusal reason is
reached by a history that earns it, and each refusal's message is asserted to contain
**no number** — otherwise a user lifts `0.31` out of "only 4 periods" and types it into
the app. A negative estimate is asserted to be reported as a wrong-sign finding rather
than clamped to zero, because clamping would erase a real signal: a bad anchor, a
mismapped node, or a genuine hedge.

Three of these are not unit tests but measurements, and they are the reason the module
can make a claim about its own interval:

- The **false-positive rate** on pure-noise histories is measured over 200 replications
  and asserted to sit at or below twice nominal and strictly above zero. This test
  exists because an earlier version of it asserted that a single noise history would be
  refused, and that assertion failed at one seed. Measuring the rate showed the
  estimator was well calibrated and the *test premise* was wrong. The confidence level
  was raised from 0.90 to 0.95 in the same change, on the grounds that a spurious
  elasticity multiplies a dollar figure a CFO signs.
- **Coverage** of the reported interval is measured against a known elasticity, at about
  91% nominal-95% for independent prices and about 88% for an AR(1) market with
  persistence 0.85. Both are pinned with floors below the measurement, so a regression
  in the resampling — dropping the blocks, or substituting textbook standard errors —
  fails loudly while normal stochastic variation does not.
- The **point estimate is separately asserted unbiased** in both settings. Which of the
  two is wrong matters: a biased estimator would mean every adopted multiplier is
  systematically off, while an optimistic interval means the number is right on average
  and stated with slightly too much confidence. The first would be a reason not to ship
  the feature; the second is a documented caveat.

## 8. Known limitations and not yet built

**Price book values are user-supplied and uncalibrated.** `PriceBook` makes the
objective *coherent*, not *correct*. `value_per_risk_point`,
`value_per_lead_time_day` and `price_per_carbon_ton` are inputs someone has to
defend; the default `source` is `"unstated"`. `from_annual_exposure` divides an
annual exposure linearly across the risk scale, which is an explicit assumption
sitting in one place rather than a calibrated loss curve.

**The concave response's exponent is an uncalibrated assumption.**
`AllocationConcaveResponse(exponent=0.85)` inherits 0.85 from the acquired system,
where it had no stated empirical basis. Moving it from the parameter to the
allocation makes it mean what the documentation claimed, but it does not make it
measured. It is marked `calibration_source="uncalibrated-assumption"` for exactly
that reason. The same applies to `LognormalShock(sigma=0.12)` and
`UniformCorrelation(rho=0.35)`.

**Two baselines coexist, deliberately.** `network.baseline_risk_pts` is the raw
figure and is what the `enforce_risk_cap` constraint bounds.
`effective_baseline_risk_pts` is that figure times the macro multiplier, which is what
target-mode goals are set against and what every user-facing percentage is quoted
from. They differ whenever the macro multiplier exceeds 1.0. Unifying them would
change results, so the discrepancy is named and documented (ADR-001 F18) rather than
silently resolved. If you consume `optimized_risk_pts` directly, you probably want
`effective_optimized_risk_pts` instead.

**CBC only.** There is no solver abstraction. Larger networks with many tangent
breakpoints multiply binary big-M rows, and CBC will be the first thing to give
out. Nothing has been benchmarked at production scale.

**No persistence layer.** Nothing is stored. Results carry hashes and a
timestamp, but there is no ledger to write them to, no run history, no way to
retrieve the inputs behind a hash. `to_records()` gives flat rows for whatever a
caller wants to persist; that is the whole story today.

**Market-to-node mapping was not extracted.** It remains in the Streamlit app.
`macro_multiplier` here is a single scalar applied uniformly to every node's risk
reduction parameter — the acquired system's Brent-crude derived multiplier — and
its docstring says plainly that a single-commodity scalar stands in for a proper
macro state vector. `SupplyNetwork.with_interventions` exists to give a future
per-node adjustment layer somewhere to attach without mutating the network.

**The macro feed was not extracted.** The live market data feed stayed in the
Streamlit app, and it was carrying a placeholder API key. Anything downstream of
"what is the current macro state" is unbuilt here; the engine only accepts a
number.

**Target mode can be structurally infeasible at the acquired system's own
shipped default.** On the parity portfolio, the parameter-power transform caps
total attainable reduction at roughly 30.7 points against an effective baseline of
about 69.6, so the lowest reachable risk is roughly 38.9%. The acquired system
shipped `target_risk_goal` defaulting to 20.0, which is unreachable, and any macro
multiplier above 1.0 pushes it further out. Worse, target mode set the budget to
1e9, so the user cannot spend their way out of it — it is a structural ceiling
presented as a budget problem. Both engines return infeasible, so extraction is
faithful; the behaviour is recorded in `test_parity.py` so a rebuild does not
silently "fix" it and lose the evidence. The new engine at least reports why
rather than rendering a blank result.

**Also absent:** no CLI, no HTTP API, no packaging beyond an editable install, no
budget sensitivity sweep (the acquired system's second, divergent optimizer copy
was not carried over as a feature), and no scenario comparison — the monetary
objective makes results comparable across scenarios, but nothing in the engine
runs or diffs a set of them.
