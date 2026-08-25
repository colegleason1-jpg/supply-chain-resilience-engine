# ADR-001: Reformulating the allocation objective in money

- **Status:** Accepted, implemented
- **Date:** 2026-08-25
- **Scope:** `scrcae.optimization.objectives`, and the extraction of the engine out of the Streamlit application
- **Supersedes:** the acquired system's weighted-sum objective, now retained as `LegacyWeightedObjective`

## Summary

The acquired engine chose capital allocations by maximising

```
Σ  [ w · marginal_risk_n  +  (1 - w) · lead_time_saved_days_n ]  ·  x_n
```

This expression adds percentage points to days. It is not a wrong number; it is
not a number at all. Its argmax is a real decision that a real budget was going
to be spent against, so the incoherence is not academic.

We have replaced it with an explicit monetary objective — the net present value
of the portfolio, in currency — and preserved the original as a named, versioned,
opt-in class so that parity remains provable. The engine has also been lifted out
of Streamlit into a headless, tested package (126 tests). Four further defects
were found *by* that test suite, not by reading the code, and are documented
below.

**The recommendation this memo argues for: do the reformulation before any
calibration work, not after.** Every parameter tuned against the old objective
was tuned against the accident of the input data's units.

## Context

The objective sat at the centre of the acquired system. Everything else — the
0.85 exponent, the risk weight `w`, the bundle discounts, the Monte Carlo shock
model — was either an input to it or a presentation of its output. It was also
the one component nobody could test, because it only existed inside a Streamlit
callback that read from `st.session_state`.

That coupling matters more than it looks. The audit's conclusions about the math
were unfalsifiable while the math could not be executed outside the UI. So the
first move was extraction; the reformulation is what extraction made checkable.

## Decision

### 1. The unit problem, and why it is not fixable by tuning `w`

`w` is presented to the user as a preference dial: how much do we care about risk
versus speed? It cannot be that, because it is simultaneously doing unit
conversion. Squeezing two jobs into one scalar has four consequences.

**No interpretable value.** There is no answer to "should `w` be 0.6?", because
the question conflates a preference with an exchange rate.

**No portability.** `w = 0.65` tuned on one dataset means something different on
the next, since the implied exchange rate depends on the relative magnitudes of
whatever numbers happen to be in the two columns.

**It poisons everything calibrated beneath it.** The 0.85 exponent, the shock
sigma, the bundle discounts — anything fitted to make this objective's output
look reasonable inherited the defect.

**The dial mostly does nothing, then flips.** Because one coefficient family
typically carries much larger raw numbers than the other, the argmax is pinned by
whichever dominates. `w` can be swept across most of `[0, 1]` with no effect and
then jump discontinuously. A user tuning it sees a control that appears broken
and is in fact meaningless.

The fourth point is now a passing test rather than an assertion.
`test_legacy_objective_decision_flips_on_a_pure_unit_change` builds two
candidates — one risk-led, one lead-time-led — and solves twice. The only
difference between the runs is that the lead-time column is divided by 365,
relabelling the same savings from days into years. No business fact changes. At
`w = 0.5` the legacy objective funds the lead-time-led candidate when the column
is in days and the risk-led candidate when it is in years. Same preferences, same
data, opposite decision. The monetary objective, given a price expressed in the
matching unit, does not move.

### 2. The replacement

`MonetaryNPVObjective` maximises

```
Σ_n [ AF · ( λ_risk · v_risk · ΔR_n(x_n)
            + λ_lead · v_lead · L_n · x_n
            - p_carbon · K_n · x_n ) ]
   - Σ_n C_n x_n
   + Σ_b D_b b_v
```

where `AF` is the annuity factor `(1 - (1 + d)^-T) / d` — 3.7907867694 at a 10%
discount rate over 5 years, verified numerically — and prices come from an
explicit `PriceBook`:

```python
PriceBook(
    value_per_risk_point=...,      # currency per percentage point avoided, per year
    value_per_lead_time_day=...,   # currency per day saved, per year
    price_per_carbon_ton=...,
    benefit_horizon_years=5,
    discount_rate=0.10,
    source="...",                  # provenance is a required field, not a comment
)
```

The objective's unit is the string `"currency (net present value)"`, and it is
carried in the audit record.

Three properties follow, and each is worth more than the tidiness.

**The exchange rate becomes a claim someone can dispute.** `value_per_risk_point`
is a number a CFO can argue with, and `PriceBook.from_annual_exposure` derives it
from annual disruption exposure so the argument starts from something the business
already estimates. Preference is separated out into the `λ` multipliers, which
default to 1.0. A `λ_lead` of 1.5 says "we value speed at 150% of its computed
financial value" — a strategic statement one can agree or disagree with. `w = 0.65`
was not a statement about anything.

**The model can decline to act.** This is the single most valuable thing a capital
allocation model can say, and the legacy objective could never say it. Because
capital appears in the legacy objective only as a constraint, never as a cost, the
optimum always spends the entire budget; funding nothing scores zero and is always
dominated. Once capital is priced, a portfolio whose discounted benefits do not
cover its cost scores below zero, and the engine funds nothing and reports an
objective of 0.0. A tool that always recommends spending the budget is not
advising on capital allocation; it is ratifying a decision already made.

**Legacy results self-identify.** `LegacyWeightedObjective.unit` returns
`"incommensurate (percentage-points + days)"`. Any historic result carrying that
string can be found and reviewed rather than silently trusted.

### 3. Legacy behaviour is preserved, not deleted

Legacy mode is `LegacyWeightedObjective` + `ParameterPowerResponse` +
`LegacyTruncatedNormalShock`, with `enforce_risk_cap=False`,
`legacy_bundle_activation=True`, `min_funding_scale=0.3`. 35 parity tests cover
it, including a 20-cell weight × budget grid.

The reason is narrow and important: without provable parity, *"we rewrote the
engine"* and *"we changed the answers"* are indistinguishable claims. Anyone
reviewing a decision made on the old system needs to be able to reproduce it.

Legacy and new Monte Carlo cannot be made bit-identical, because the acquired
code drives a global `np.random.seed(42)` while the new code uses a local
`Generator`. Parity is therefore asserted distributionally, at 200k iterations,
within 6 standard errors.

## Findings from the forensic audit

These are properties of the acquired system, established by reading it and then
by pinning each one with a test.

### Objective and optimisation

**F1 — Unit incoherence.** As above. Proven by test.

**F2 — The 0.85 exponent is applied to the wrong argument.** This is the more
serious finding, and the audit conflated it with the separate observation that the
exponent is uncalibrated. The code computes `marginal_risk_n = (macro · risk_n) ^ 0.85`
and then multiplies by the funding scale `x_n` linearly. The exponent is applied
to the risk *parameter*, not to funding. The model is therefore perfectly linear in
funding: 50% funding delivers exactly half the reduction, always. There are no
diminishing returns anywhere in the acquired system, despite the equation ledger
stating that modelling them is the exponent's purpose. What the exponent actually
does is reweight nodes cross-sectionally — a node with 4× the raw reduction
delivers `4^0.85 = 3.28×`. That may even be defensible, but it is not what it was
documented to be. Now pinned by a constant-marginal-returns test.

**F3 — The risk cap was applied after solving.** `min(baseline, drop)` was
computed post-hoc, so the optimizer could spend capital on reduction beyond the
baseline, which cannot exist, and report it as productive. Now enforceable
in-model via `enforce_risk_cap`.

**F4 — Fake verification.** The UI displayed *"Verification: zero fractional
violations detected"* as a hard-coded string literal, and `sha256:8f4c99a...` as a
format-string literal that hashed nothing. Both are now real: constraints are
checked after solving and content hashes are computed over canonical JSON.

**F5 — Bundle economics were widget-order-dependent.** `required_nodes` defaulted
to `nodes[:2]` from a Streamlit `multiselect`, so which nodes a discount required
depended on render order. Requirements are now explicit data.

**F6 — The optimizer was duplicated.** The budget-sweep feature contained a
second, subtly divergent copy. There is now one solver.

### Stochastic model

**F7 — The 0.4 shock floor biases the mean upward.** `max(0.4, 1 + 0.12·Z)`
truncates only the left tail, so the multiplier's mean exceeds 1. Closed form:
`E[M] = f·Φ(a) + (1 - Φ(a)) + σ·φ(a)` with `a = (f - 1)/σ`. Verified magnitudes:
**6.4e-9 at σ = 0.12, 2.5e-3 at σ = 0.30, 1.9e-2 at σ = 0.45.** An earlier draft
of this analysis claimed the σ = 0.30 bias exceeded 1e-2; that was wrong by a
factor of four and the test now records the correct value. At the shipped σ = 0.12
the bias is negligible. The finding is a governance defect rather than a numerical
one: the error is strictly positive and monotone increasing in σ, so it grows
silently as someone calibrates σ toward realism, and it is multiplied by total
portfolio reduction — about 0.08 points on the test portfolio at σ = 0.30.

**F8 — The normal multiplier can go negative.** For `Z < -1/σ` the raw multiplier
is negative, meaning an intervention that increases risk. The floor conceals this
rather than fixing it. The replacement is lognormal, `exp(σZ - σ²/2)`: strictly
positive, exactly mean-one for any σ, and right-skewed, which is the correct shape
for disruption severity.

**F9 — Ridge repair destroys the unit diagonal.** The acquired code repaired a
non-PSD correlation matrix with `+= I·(-min_eig + 1e-5)`, which silently converts
it into a covariance matrix with inflated, unequal variances, and dilutes the
analyst's entered correlations toward zero. Verified: naive repair yields sample
variance above 1.05 while eigenvalue repair holds it at approximately 1.0.

**F10 — Global RNG pollution.** `np.random.seed(42)` mutates process-wide state,
which makes results depend on execution order elsewhere in the app and makes
seed-sensitivity testing impossible. Now a local `Generator`.

### Data and architecture

**F11 — Market→node mapping by exact string equality.** `MarketFeedbackBridge`
matched `Node Name == Sector`, which cannot work for a real supply chain. Not
extracted; still in the app.

**F12 — `st.session_state.local_market_table` was the system of record.** It is
not persistent, so the original design thesis is unsatisfied by the original
design. Not addressed here; it needs a storage layer.

**F13 — The macro feed shipped with a placeholder credential.** The Brent call
carried `'YOUR_FREE_API_KEY_HERE'` in source, plus a markdown-wrapped URL. Not
extracted.

### Found by testing, not by reading

These four were not in the original audit. They are the return on writing the
suite.

**F14 — Target mode is structurally infeasible at its own shipped default.** The
exponent transform (F2) caps attainable reduction at approximately 30.7 points
against 39.9 raw, so with an effective baseline of 69.6 the reachable risk floor
is about 38.9%. The shipped default `target_risk_goal = 20.0` is unreachable. And
because target mode sets the budget to 1e9, no amount of spending resolves it: a
structural ceiling presents itself as a budget problem. Both engines agree it is
infeasible, so extraction was faithful; the behaviour is preserved as a test.

**F15 — The verification tolerance was mis-shaped.** The engine's own budget
check used a relative factor of `1e-9`, which is tighter than CBC's primal
feasibility tolerance. Observed: a 5.1e-4 absolute overrun on an 88,224 budget —
5.8e-9 relative — flagged as a violation of a solution the solver considered
valid. Fixed by making the tolerance scale-aware via a `relative_tolerance`
request field, tightening CBC's options, and — the part that matters —
**recording realised slack in `ConstraintReport.residual_slack` rather than
discarding it.** A check that only ever says "passed" is F4 again in a different
costume. Whoever signs a capital plan is entitled to see the actual overshoot.

**F16 — The in-model risk cap was unsound under linearisation.** A concave risk
response is approximated by tangent lines, giving a variable bounded only from
*above*. It rises to its bound only because the objective pushes it there. When
risk carried little or no objective weight — a lead-time-focused mandate — nothing
pushed it, so the solver could report a low risk figure while funding a portfolio
whose true reduction exceeded the baseline. The cap stopped binding precisely when
it was needed. The fix is to recognise that the two scalar risk constraints need
opposite approximation errors: the objective and the required-reduction floor use
an inner (chord) approximation that never over-promises, while the cap uses an
outer (tangent) approximation that never under-reports, with a big-M term
releasing it for unfunded nodes. Found by a property test, not by inspection —
reading the code, the single shared variable looks obviously correct.

**F17 — Bundle discounts could manufacture spending capacity.** Discounts were
unbounded and gated on node *activation* (`b <= y`). A node funded at its 30%
minimum still counted as activated, so a portfolio could claim a discount priced
against the full package while paying a fraction of it. A property test found a
network that funded itself at a budget of zero. Two changes: the domain model now
rejects a discount greater than or equal to its bundle's gross cost at full
funding, and the discount is by default contingent on full funding (`b <= x`),
which is also what a volume discount actually is. The old rule is available as
`legacy_bundle_activation=True` for parity.

Also fixed in passing: breakpoint generation accumulated floating-point error via
repeated addition and produced a final breakpoint of 1.0000000000000002, outside
the response model's domain. Endpoints are now exact by construction.

## Consequences

**Positive.** The objective is unit-coherent and its parameters are individually
disputable. The engine runs headless and is covered by 126 tests. Legacy parity is
provable. Uncalibrated constants carry a `calibration_source` field, so
assumptions appear in the audit ledger instead of passing as data. Every reported
quantity is recomputed from `risk_response.evaluate()` ground truth rather than
read back from the linearised LP objective, so results are exact regardless of
approximation quality — itself asserted as a property test. Four real defects were
caught before shipping.

**Negative, and honestly.** The price book must be populated, and populating it is
a genuine business exercise that cannot be skipped or defaulted; the engine now
requires an input the acquired system let users avoid supplying. The concave
response's exponent remains an uncalibrated assumption — it is now labelled as
one, which is progress but not calibration. The objective's inner approximation
widens allocation indeterminacy in symmetric problems from roughly half a facet
width to a full facet, about 0.058 at the default 13 breakpoints; this is the
honest bound for an approximation that no longer over-states delivery, and it
narrows with more breakpoints at the cost of extra big-M rows. Legacy mode is
ongoing maintenance surface. And the UI, market bridge, macro feed, and
persistence layer are all still to be reconnected to this engine.

**Neutral.** Numbers will change relative to the acquired system. That is the
point, and parity mode is how we stay able to explain the difference.

**F18 — Two different baselines were both called "the baseline."** The acquired app
computed `effective_baseline_risk = min(100, baseline_risk * macro_multiplier)` and
derived target mode's required reduction from that inflated figure, while its
in-model constraints used the raw `baseline_risk`. The two agree only when the macro
multiplier is exactly 1.0, i.e. only when the macro feed is doing nothing. The
consequence was a reporting error in the optimistic direction: a user who set a 40%
target and got a feasible answer was shown a residual risk computed off the raw
baseline, which is lower than the one their target was defined against. Found while
building the frontier reporting, because the diagnosis quoted a target of 15.9% back
to a user who had typed 20%.

The fix names both quantities instead of merging them.
`OptimizationRequest.effective_baseline_risk_pts` is the macro-inflated figure and is
used for every number shown to a reader; `network.baseline_risk_pts` remains what the
`enforce_risk_cap` constraint bounds. Changing which baseline the *cap* uses would
alter results and is a modelling decision, not a presentation one, so it was left
alone and documented rather than quietly changed.

## The attainable frontier (resolves F14)

`SolveStatus.INFEASIBLE` is retained verbatim in the result and the audit ledger.
Softening the machine's answer is what produced F4, and we are not repeating it. But
`infeasible` is a solver word, and on its own it misdirects: because target mode set
the budget to 1e9, an unreachable target presented as a *budget* failure, so the
obvious remedy was to ask for more capital. On the shipped default that remedy is
useless — the ceiling is structural.

So the engine now computes, alongside the status, what the portfolio can actually
deliver:

- **`attainable_frontier(network, response, budget=None)`** maximises in-model risk
  reduction and returns the lowest reachable risk, its cost, and which constraint
  bound (`"structure"`, `"budget"`, or `"risk_cap"`). It is solved, not summed: a
  prerequisite capped at 0.2 scale drags its dependents down with it, which the
  naive sum in the old parity helper missed.
- **`diagnose(request)`** classifies a failure as `EXCEEDS_RISK_CAP` (more reduction
  requested than there is risk to remove), `EXCEEDS_STRUCTURAL_CEILING` (no budget
  helps), `EXCEEDS_BUDGET` (reachable; reports the exact minimum capital required),
  or `UNDIAGNOSED`.

Three properties are deliberate. **The frontier is conservative:** because the
objective uses the inner (chord) approximation, it can understate the ceiling by up
to one facet width but never overstate it — the safe direction for a number used to
tell someone what they cannot reach. **The target is never silently relaxed:** the
attainable figure is reported next to the request, never substituted for it, because
a reader who believes they hit 20% when the model reached 39% is worse off than one
who saw `Infeasible`. **`UNDIAGNOSED` is a real outcome:** when none of the checks
explains the failure, the engine says so instead of inferring a cause, which is the
F4 discipline applied to a new surface.

A non-optimal result also now presents no allocations. CBC leaves values in the
decision variables after reporting infeasible; those values violate the constraints
and mean nothing, yet they read as a fully funded portfolio. `active_allocations`
returns empty unless the solve succeeded, with the raw values kept on `allocations`
for debugging.

## Recommended next steps

1. ~~Populate the price book with defensible figures and record their provenance in
   the `source` field.~~ **Done, and the route taken matters more than the fact.**
   `app/services/pricing.py` derives `value_per_risk_point` from four figures a
   finance function already reports — annual revenue, the share of it carried by this
   network, gross margin on it, and expected days of interrupted supply — plus any
   fixed per-event cost:

   ```
   cost of one event = revenue x share x (days / 365) x margin + fixed cost
   value per risk point = cost of one event / 100
   ```

   The divisor is the load-bearing assumption and is stated on screen: the 0–100 scale
   is read as linear in expected loss, one point being one percentage point of annual
   probability, so at 100 points a disruption is certain and expected annual loss
   equals the cost of one event. `PriceBook.from_annual_exposure` already encoded that
   reading, so the app and the engine cannot drift apart on it.

   **An industry benchmark was deliberately rejected as the default.** The obvious
   move was to source a published figure for the average cost of a disruption as a
   share of revenue and ship it. That was investigated and abandoned, because such a
   figure is a *population* statistic: a median across firms of different size, margin,
   sector and network depth. Substituting one for this company's price of risk yields a
   number that *looks* sourced — it has a citation beside it — while being no more
   applicable to the business than the 0.85 exponent was. Borrowed authority is the
   specific defect this rebuild exists to remove, and it is worse in currency than in
   a dimensionless weight, because a dollar figure is more persuasive. Third-party
   benchmarks belong here only as a clearly labelled plausibility cross-check.

   Provenance now travels: `PriceBook.is_sourced` distinguishes a stated source from
   `"unstated"`, `MonetaryNPVObjective.provenance` exposes every price in force, and
   `build_audit_record` records it as `objective_provenance` — outside the hash inputs,
   so no historic hash moved. A derived book's `source` string contains the arithmetic
   itself, not a label, so the ledger can be checked rather than trusted.

   What remains is validation: back-testing the implied cost of an event against the
   company's own disruption post-mortems is what would turn this from an arguable
   assumption into a calibration.
2. ~~Make the Streamlit app a client of `scrcae` and delete the in-UI math~~ —
   **done.** See "The app as a client" below. The 1,146-line monolith is replaced
   by `app/`, which imports no solver and performs no arithmetic on model
   quantities. F6 is resolved by `optimization/sweep.py`: budget sensitivity is
   now repeated calls to the same `solve()` that produces the headline figures, so
   the curve and the number above it cannot disagree.
3. ~~Add a persistence layer so the system has a real system of record (F12).~~
   **Done.** `app/storage/` holds a SQLite store, owner-scoped in SQL rather than in
   the UI. Portfolios persist inputs and settings plus a `RunProvenance` record — the
   engine version and the input/output hashes of the run they were saved from — and
   deliberately **not** results. On reload the engine re-solves and the hashes are
   compared, so the app either confirms the saved result reproduces or names the
   divergence. Persisting the numbers instead would have aged them into fiction: a
   stored "optimised risk: 25.25%" attributed to an engine that no longer produces it
   is exactly F4's defect with a database behind it. Auth (`app/auth/`) scopes it, and
   its five-attempt lockout now lives in an attempts table rather than
   `st.session_state`, where a page refresh defeated it.
4. ~~Replace exact-string market→node mapping with an explicit mapping table (F11).~~
   **Done.** The acquired `MarketFeedbackBridge` was worse than first recorded. It
   built its node mask with `nodes['Node Name'] == sector`, comparing a market's sector
   name to a user-typed intervention name, so in the normal case it matched nothing —
   and when the `Node Name` column was absent the mask fell back to
   `[True] * len(nodes)`, applying one market's volatility to every node in the
   network. Its multiplier was `1 + volatility * 0.15 - min(daily_change, 0) * 0.5`,
   which subtracts a negative, so a market crash *increased* the reported risk
   reduction; both directions could only ratchet efficacy upward. And it assigned the
   result back into the user's own table (`st.session_state.nodes_df = ...`), so every
   30-minute sync compounded on the last one and the entered Risk Reduction figures
   drifted upward irrecoverably. The market table itself seeded every row at
   `Live Price 100.00 / 0.00%` and reported failures as "Holding Last Known", so a feed
   that had never once succeeded displayed eleven plausible prices — which is on top of
   the oil URL being stored as a markdown link string, malformed beyond repair.

   The replacement has four parts:

   * **An asserted mapping.** `Node ID`, `Market Symbol`, `Anchor Price`,
     `Risk Elasticity` — four columns a person fills in, parsed by
     `adapters.build_node_exposures`, which rejects an unknown node id, a missing
     anchor, a missing elasticity, a non-positive anchor, a duplicate pair, and a
     negative elasticity, naming each. A blank row means "no exposure", which is the
     right answer for most interventions; a *partly* filled row is an error, because
     silently skipping it leaves the user believing a mapping is live.
   * **Per-symbol quotes.** `MarketQuote.price` is `None` when unavailable. There is no
     default price anywhere in the module, and each symbol carries its own status, so
     three working markets and one broken one are reported as exactly that.
   * **A solve-time overlay, never a write-back.** `market.build_overlay` produces
     `OptimizationRequest.node_macro_multipliers`, a new per-node mapping the engine
     applies at every one of the five places the scalar `macro_multiplier` was already
     applied — the linear coefficient, the ceiling, the concave chords, the tangents,
     and the final evaluation — and which `sweep_budget` inherits automatically through
     `replace(request, budget=...)`. The intervention table is never modified, so
     nothing compounds between runs. `test_the_users_intervention_table_is_never_modified_by_the_feed`
     runs the real Streamlit script three times and asserts the entered values are
     byte-identical.
   * **Inertness reported as loudly as effect.** A configured exposure with no usable
     quote yields multiplier 1.0 *and* a warning naming the node, the market and the
     reason. An overlay with nothing to say hands the engine an empty mapping rather
     than a mapping of ones, so it is provably a no-op: the audit `input_hash` of a
     dead-feed run equals that of a run with no exposure configured at all.

   Two engine invariants were chosen deliberately. **Unknown node ids raise** rather
   than being ignored, because silent non-matching is precisely how the acquired bridge
   hid for as long as it did. And **a per-node multiplier does not move the reported
   baseline** — `effective_baseline_risk_pts` uses the scalar alone. Aggregating
   per-node multipliers into a baseline would need weights the network does not carry,
   and the obvious candidate (share of total risk reduction) makes the baseline shift
   when an intervention is merely *written down*, which is F17's defect in another
   costume. Exposure changes which interventions pay for themselves; it does not
   restate the risk being measured.

   The direction is one-sided by choice, matching `services/macro.py`: prices above
   anchor raise risk, prices below are not credited as reducing it. A `one_sided=False`
   switch exists and is tested, because that asymmetry is an assumption, not a finding.

   The copilot chat has also been ported (`app/copilot.py`), as a proposal engine
   rather than an actor. It parses an instruction into explicit `Change` records —
   field, current value, proposed value, rationale — and returns them for review;
   `apply_proposal` produces a new snapshot and raises `StaleProposal` if the inputs
   moved since the proposal was built. It imports no Streamlit, touches no session
   state, calls no model, and records anything it did not understand in `unresolved`
   rather than guessing. The acquired version wrote `val_total_budget` and
   `val_target_mode` straight into session state and reran, and switched target mode on
   for any sentence containing "risk" and a number — which is how a user reached the
   structurally unattainable 20% target of F14 by typing one line.
   **Addendum: the elasticity is no longer necessarily asserted.** The mapping above
   was designed as four columns a person fills in, and the elasticity was the weakest of
   the four — a number with the same standing as the 0.85 exponent this entire document
   exists to object to. It can now be measured from the company's own delivery history.
   `scrcae.calibration.elasticity` fits
   `disruption_rate = f0 · (1 + e · max((price − anchor)/anchor, 0))` per exposure
   against that node's monthly market series and reports `e` with a moving-block
   bootstrap interval; `app/services/calibration.py` decides which fits may be adopted.

   Three properties of the design are the point of it:

   * **A refusal is named and carries no number.** Seven distinct `FitQuality` outcomes,
     because "you have four periods" and "your data show nothing" require opposite
     responses. A refusal's summary text is asserted by test to contain no digits a user
     could copy, since anyone shown `0.31 (only 4 periods)` will type `0.31` into the
     table and the caveat will not survive the journey.
   * **Provenance is derived, never stored beside the number.** There is deliberately no
     "Elasticity Source" column, because a free-text provenance field is one a user can
     type `calibrated` into. A row counts as calibrated only while a stored fit for the
     same node and symbol records the same elasticity *and* the same anchor to within
     `1e-6`. Overtype the number or move the anchor and the label reverts to asserted on
     its own. The label cannot outlive its evidence, which is a structural guarantee
     rather than a discipline anyone has to maintain.
   * **The estimator's interval is measured, not claimed.** Coverage against a known
     elasticity is about 91% for a nominal 95% interval with independent prices and
     about 88% for an AR(1) market with persistence 0.85 — the regime a commodity is
     actually in. The point estimate is unbiased in both. Longer resampling blocks were
     tried as a remedy and were slightly worse, so the shortfall is documented and
     pinned by test rather than smoothed over: what is printed as 95% behaves like
     roughly 90%. Given the subject of this ADR, shipping an interval whose true
     coverage nobody had measured would have been the same error in new clothes.

   What this does **not** do is establish causation. The fit is an association: a
   hurricane that lifts crude and closes a port is attributed entirely to the price.
   And delivery records measure the risk a node *carries*, not the share an intervention
   *removes*, so using a fitted elasticity as an effectiveness multiplier assumes an
   intervention averts a constant fraction of node risk. Both caveats travel in every
   fit's own notes, including refused ones.

   The market feed was rebuilt underneath this at the same time. `app/services/feeds.py`
   introduces a `QuoteProvider` seam with three implementations, and **the default now
   requires no credential**, because F13's root cause was a provider that needed a key
   nobody had set — which is how "Offline (Holding Last Known)" over a hard-coded 100.00
   became the permanent state. A keyless default is only a partial answer: the endpoint
   it uses is public but undocumented and carries no service commitment, which is stated
   in the module and in the app README, and `SCRCAE_MARKET_PROVIDER=api-ninjas` is the
   production path. The provider that actually answered is named in brackets after every
   quote, so a silent fallback is still a legible one.

   One defect in this work was found by driving the app rather than by reading it, which is
   worth recording because it is the class of bug tests of the kind written here do not
   catch. The panel's comparison line printed the ratio of the entered elasticity to the
   fitted one — correct — and labelled it with the direction of the *inverse* ratio, so an
   assumption at 0.30 against a fit of 1.21 was reported as "0.25x above the fit". The
   number was right, the direction word was wrong, and it was wrong in the flattering
   direction: a badly understated risk elasticity read as though it were conservative. The
   existing test asserted that both numbers appeared in the sentence and said nothing about
   the word between them, so it passed throughout. It now pins the direction on both sides,
   which is the actual claim the sentence makes.

5. ~~Move the macro feed server-side with a real credential~~ — **done.**
   `app/services/macro.py` reads `SCRCAE_MACRO_API_KEY` from the environment and
   distinguishes "unconfigured" from "provider failed" from "live", so a held
   default can no longer be displayed as telemetry (F13).
6. Calibrate the concave exponent against historical delivery, or drop it in
   favour of a linear response and say so. **Still open, and now conspicuous.** The
   market elasticity has been moved from assertion to measurement (item 4's addendum);
   the response exponent has not. It remains marked
   `calibration_source="uncalibrated-assumption"`, and the machinery built for the
   elasticity does not transfer to it: delivery history reveals how a node's risk moves
   with a market price, not how its risk responds to being *funded*, which is the
   quantity the exponent governs. Measuring that needs variation in funding levels — an
   internal experiment or a staged rollout — not more market data.
7. ~~Decide the target-mode policy~~ — **done.** See "The attainable frontier"
   above. The engine distinguishes structural from budgetary infeasibility and
   reports the attainable floor. The UI now renders the diagnosis rather than the
   raw status string; see below.

## The app as a client

The replacement UI lives in `app/` and is organised so that the question "does the
interface do any modelling?" can be answered by reading one import list:

| Module | Responsibility |
| --- | --- |
| `adapters.py` | Spreadsheet frames → validated domain objects. Returns `(object, Rejections)`. |
| `engine_client.py` | The only module that calls `scrcae` solvers. |
| `presenters.py` | Engine results → display rows and strings. Formatting only. |
| `services/macro.py` | The macro feed, with provenance and explicit failure states. |
| `views/`, `main.py` | Streamlit layout. No arithmetic. |

Two consequences are worth stating because they were the point of the exercise.

**Infeasibility is never shown as "infeasible."** Target mode runs with no budget
ceiling or a very large one, so the solver's own word pointed every reader at a
funding shortfall when the truth was usually a portfolio gap. `status_banner`
reads `OptimizationResult.diagnosis` and styles the two cases differently: a budget
shortfall is a warning that names the capital required, and a structural ceiling is
an error that says plainly that additional capital will not change the outcome.
This is pinned by `test_unreachable_target_never_shows_the_word_infeasible` and
`test_the_two_failure_modes_are_visibly_different`.

**The audit tab can only report what the engine computed.** The acquired system
printed `sha256:8f4c99a...` and "Zero fractional violations detected" as string
literals with nothing behind either (F4). The replacement reads
`constraint_report.checks_performed`, names the constraint families that were
verified, and prints the real input and output hashes. It also discloses the
scale-relative tolerance actually used on money constraints, because showing only
the absolute `1e-6` would misrepresent the check (F15).

A failed solve renders as em-dashes rather than zeros. Zero capital against the
baseline risk is a legitimate-looking "do nothing" recommendation, and an empty
network solves cheaply and feasibly, so `build_network` returns `None` with reasons
rather than an empty network.

## Verification

```
cd engine
python -m pytest -p no:warnings
```

197 tests: 12 known-answer, 11 objective-unit, 26 risk-response, 25 stochastic,
35 parity, 23 node-exposure, 30 elasticity-calibration, 17 property-based,
18 diagnostics. Solver is CBC via PuLP; `pytest.ini` sets `pythonpath = src`, so run
from the `engine/` directory and invoke pytest as `python -m pytest`.

From the repository root, `python -m pytest -p no:warnings -q` runs 672 tests — those
197 plus the 475 covering the Streamlit client. Three of the calibration tests are
Monte Carlo measurements rather than assertions about a fixed case, and are the slowest
thing in the suite; they are what licenses the interval claim above.
