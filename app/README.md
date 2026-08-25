# app — the Streamlit client

This replaces `legacy/acquired-system-v1.py`, a 1,146-line single file that held the interface, the
optimizer, a second copy of the optimizer, the Monte Carlo, the market feed and the
auth flow. The rule here is that the interface collects inputs, hands them to
`scrcae`, and renders what comes back. When a number is wrong there is one place it
can be wrong.

## Running it

```bash
python -m pip install -e "engine[test]"
python -m pip install streamlit pandas
streamlit run app/main.py
```

`streamlit` and `pandas` are not engine dependencies and are not declared in
`engine/pyproject.toml` on purpose — the engine must stay importable from a notebook,
a script or a future API without pulling in a web framework.

Optional environment configuration:

| Variable | Effect if unset |
| --- | --- |
| `SCRCAE_MACRO_API_KEY` | The macro feed is reported as unconfigured and no macro adjustment is applied. |
| `SCRCAE_MARKET_PROVIDER` | Defaults to `yahoo`, which needs no credential. Set it to `api-ninjas` for the keyed provider or `static` for a fixed supplied price. An unrecognised name falls back to the default rather than failing to start, and the provider that actually answered is named in brackets after every quote. |
| `SCRCAE_MARKET_API_KEY` | Only read by the `api-ninjas` provider. When that provider is selected and the key is absent, every per-node quote says which variable is missing, each exposure is listed as inert, and no per-node adjustment is applied. The default provider ignores this variable. |
| `SCRCAE_DB_PATH` | Portfolios are saved to `portfolios.db` in the working directory. |
| `SUPABASE_URL`, `SUPABASE_ANON_KEY` | Supabase auth is skipped. |
| `SCRCAE_USERS` | No local accounts. With no Supabase either, the app runs unauthenticated as a single local user and says so on screen. |

`SCRCAE_USERS` is JSON mapping an email to a PBKDF2 hash. Generate one with:

```bash
python -m app.auth.hash_password
```

Plaintext passwords in `SCRCAE_USERS` are rejected rather than accepted with a warning,
because a development convenience that accepts plaintext is what ends up in production.

## Seeing the market exposure workflow

`tools/seed_demo_portfolio.py` writes a saved portfolio with two market exposures and 120
rows of delivery history, with both elasticities seeded at 0.30 — deliberately wrong, so the
calibration visibly disagrees with the stored assumption. `tools/capture_walkthrough.py`
then drives a running server through the whole path and writes the screenshots in
`app/docs/walkthrough/`. See `tools/README.md`.

The sequence those screenshots show, which is also the sequence to walk manually:

| Step | What changes |
| --- | --- |
| Live quotes arrive | `BZ=F` and `HG=F` read `Live [yahoo]`; the provider is named in the status, not implied |
| Exposures apply at 0.30 | N1 `+3.1%`, N3 `+20.3%` effect on effectiveness, from a number nobody measured |
| Calibrate | 1.21 (interval 1.03–1.44, R² 0.73) and 0.56 (0.39–1.82, R² 0.25) from 60 periods each |
| Read the caveats | Association not causation; risk carried, not risk removed; a moving-block interval |
| Adopt | The exposure table holds 1.2124 and 0.5559; the effects move to `+12.4%` and `+37.5%` |
| Verification tab | `Risk elasticities: 2 calibrated`, and the asserted-elasticity warning is gone |

The warning disappearing is the part to watch: it is counted from the exposures against the
stored fits, not set by the adoption code, so it cannot be made to disappear by asserting
that it should.

## Layout

| Module | Responsibility | Imports a solver? |
| --- | --- | --- |
| `adapters.py` | Editor frames → validated `scrcae` domain objects | no |
| `engine_client.py` | The single boundary that calls `solve`, `sweep_budget`, `attainable_frontier`, `montecarlo.run` | yes |
| `presenters.py` | Results → display rows and strings; formatting only | no |
| `services/macro.py` | Whole-network macro feed with explicit provenance and failure states | no |
| `services/market.py` | Per-symbol market quotes and monthly history, and per-node risk multipliers from an exposure table | no |
| `services/feeds.py` | The provider seam: one `QuoteProvider` protocol, three implementations, and the resolution rules between them | no |
| `services/calibration.py` | Fitting an elasticity per exposure from delivery history, and deciding which fits may be adopted | no |
| `services/pricing.py` | Deriving the price of risk from revenue, margin and disruption duration | no |
| `services/portfolios.py` | Save/load/list scoped to one principal | no |
| `services/copilot_state.py` | Copilot field names ↔ Streamlit widget keys | no |
| `storage/*` | `SavedPortfolio`, `RunProvenance`, serialization, the SQLite store | no |
| `auth/*` | `Principal`, password hashing, durable lockout, providers | no |
| `copilot.py` | A sentence → a reviewable `Proposal`; mutates nothing | no |
| `views/*`, `main.py` | Streamlit layout | no |

The dependency arrow only ever points one way: views depend on presenters and
adapters, and nothing depends on views. That is what lets most of the 399 app tests run
without Streamlit installed. `test_pipeline.py` enforces it structurally with an `ast`
walk over every module in the table above that is marked as not needing a solver — an
earlier version checked `sys.modules` instead and was really measuring the test
runner's import order.

## Decisions worth knowing before changing anything

**Adapters return `(object, Rejections)` instead of raising.** A sheet with one
unusable row among twenty must produce nineteen usable interventions *and* a visible
complaint. Raising hides the nineteen; coercing the bad row to zero hides the
complaint — and a zero-cost intervention is irresistible to the optimizer, funded to
the maximum at any budget.

**`build_network` returns `None` rather than an empty network.** An empty network
solves quickly and feasibly at zero cost, which renders as a considered
recommendation to do nothing.

**A failed solve renders em-dashes, not zeros,** and `portfolio_rows` is empty. CBC
leaves constraint-violating values in its decision variables after reporting
infeasibility; read back uncritically they look like a fully funded plan.

**The word "infeasible" never reaches the screen.** Target mode runs with no budget
ceiling, so the solver's own status pointed every reader at a funding shortfall when
the truth was usually that the portfolio itself cannot reach the target.
`status_banner` distinguishes the two and says which one applies.

**No widget is created inside a loop that computes anything.** The previous system
built `st.sidebar.multiselect` widgets inside its optimizer, with different keys per
mode, so flipping the mode switch silently reset bundle membership to the first two
rows.

**Nothing in `presenters.py` computes a model quantity.** If a number needs deriving,
it is derived in the engine and read from the result. The previous system's budget
chart and its headline figure were computed by two copies of the same arithmetic that
had drifted apart.

**Persistence stores inputs and provenance, never results.** A saved portfolio holds
its tables, its settings, and the engine version plus input/output hashes of the run it
was saved from. On reload the engine solves again and the hashes are compared, so the
app can say "this reproduces its saved result" or name the divergence. Storing the
numbers would have been storing a claim that decays: a persisted "optimised risk:
25.25%" attributed to an engine that no longer produces it is the acquired system's
whole class of defect with a database behind it.

**Owner scoping lives in SQL, not in the views.** `PortfolioService` holds the
principal and every query filters on it, so a view cannot pass the wrong owner id — it
has no owner id to pass. A request for someone else's portfolio returns `None`,
indistinguishable from one that does not exist, so ids cannot be probed.

**Queries are parameterised and nothing is sanitised.** The previous system stripped
characters from user strings before interpolating them into SQL, which is a filter that
must be perfect forever against an adversary who needs to be right once. Binding
parameters means the driver never parses user data as SQL, so an intervention can be
called `O'Brien's Port` and simply is.

**The sign-in lockout is in the database.** The previous system counted failed attempts
in `st.session_state`, so pressing refresh reset the counter and the control
communicated diligence without providing any. It is now an attempts table, and a test
proves the count survives a process restart.

**The copilot proposes; it never writes.** It parses a sentence into explicit `Change`
records — field, value now, value proposed, and why — and a human accepts or discards.
It solves nothing, calls no model, and never guesses: an unrecognised clause becomes an
`unresolved` entry shown as prominently as the changes. The previous system's copilot
wrote session-state keys directly and reran, so every number moved with no record of
what had been asked for and no distinction between understanding and guessing.

**Writes to widget state happen before the widgets exist.** Streamlit refuses to set a
widget's key after that widget has been instantiated in the current run, so loading a
portfolio is two-phase: the click parks a payload under a non-widget key and reruns,
and `apply_pending_load` performs the writes at the top of the next run.

### The price of risk is derived, not typed

`value_per_risk_point` decides every currency figure the app reports. Typing it into a
box is still permitted — a business may have its own model, or be exploring — but the
default path now builds it from four figures a finance function already reports: annual
revenue, the share of it carried by this network, gross margin on it, and expected days
of interrupted supply, plus any fixed per-event cost. The sidebar shows the arithmetic
line by line, so the argument is about the duration or the margin rather than about the
result.

An industry benchmark was considered as the default and rejected. A published average
cost of disruption is a population statistic across firms of different size, margin and
sector; using it here produces a number that looks sourced while being no more
applicable to this business than the 0.85 exponent was. A citation is not a calibration,
and in currency the illusion is stronger than in a dimensionless weight.

Out-of-range inputs raise rather than clamp. A gross margin of 1.4 is not a slightly
optimistic margin to be rounded down — it is 140% entered as a fraction, and accepting
it silently would inflate every figure in the app by 40%. When the derivation cannot
run, the app falls back to the *entered* price and says which numbers are in force,
rather than substituting a plausible default the user would read as derived.

Whichever path is used, `PriceBook.is_sourced` records whether a source was actually
stated, the Verification tab prints the source string in full, and the engine's audit
ledger carries it as `objective_provenance`.

### Market exposure is asserted, applied at solve time, and reported when inert

The exposure table maps `Node ID` → `Market Symbol` with an `Anchor Price` and a `Risk
Elasticity`. The relationship is stated by a person, because the acquired system
inferred it by comparing a market's sector name to an intervention's name with `==` —
which matched nothing in the normal case and matched *everything* when the column it
compared was missing.

Three properties are load-bearing:

- **No price is ever invented.** An unavailable market has no number, not a placeholder.
  The acquired table seeded every row at `100.00` and labelled failures "Holding Last
  Known", so a feed that had never succeeded looked healthy.
- **Nothing is written back.** Multipliers go to the engine as
  `node_macro_multipliers`, a solve-time overlay. The acquired bridge assigned its
  output into `st.session_state.nodes_df`, so each 30-minute sync compounded and the
  user's typed Risk Reduction values drifted upward with no way back.
- **An inert exposure is as visible as an active one.** A mapping the user configured,
  silently doing nothing because a symbol is down, is the condition that kept F11
  invisible for as long as it existed. It now produces a named warning per node.

The direction is one-sided — prices above anchor raise risk, prices below are not
credited as reducing it — matching `services/macro.py`, and switchable rather than
hardcoded.

An elasticity starts as a stated assumption. It can become a measurement, and the
Verification tab reports how many of each are in play.

### An elasticity is calibrated only while its evidence still matches

The **Delivery history** table takes `Node ID`, `Period` and `Disruption Rate %` — what
actually happened, month by month. The calibration panel fits

    disruption_rate = f0 · (1 + e · max((price − anchor)/anchor, 0))

per exposure against that node's market series, and reports `e` with a bootstrap
interval. Adopting a fit writes the number into the exposure table *and* writes the fit
itself into a stored calibration table.

The design decision worth understanding is that **provenance is derived, never stored
beside the number.** There is no "Elasticity Source" column, because a free-text source
field is a field somebody can type `calibrated` into. A row counts as calibrated only
when a stored fit for the same node and symbol records the same elasticity *and* the
same anchor, within `1e-6`. Overtype the elasticity, or move the anchor, and the row
silently reverts to asserted — the label cannot outlive the evidence for it. The
tolerance exists for one reason only: so a CSV round-trip does not downgrade a fit.

Four further rules, each with a failure mode behind it:

- **Refusals are named, not boolean.** "You have four periods" and "your data show
  nothing" call for opposite actions. A refusal's summary also contains **no number**,
  because a user who is shown `0.31 (only 4 periods)` will type `0.31` into the table.
- **Only adoptable fits are stored.** A refused fit sitting in a saved file reads as a
  calibration to whoever opens it next.
- **A negative estimate is reported, not clamped.** It surfaces as a wrong-sign refusal
  needing a mechanism. Clamping would erase a finding — a bad anchor, a mismapped node,
  or a genuine hedge.
- **Calibration runs only on an explicit button press.** It makes network calls; doing
  it on every Streamlit rerun would recreate exactly the compounding-side-effect class
  that F11 was.

Delivery history alignment deliberately **keeps** rows naming unknown nodes, unlike
exposure alignment which drops them: history is evidence of what happened, and renaming
an intervention must not destroy records. Unknown nodes are reported at parse time
instead. The join to market prices is strict — a delivery month with no price is dropped
by name and listed, never matched to the nearest month, because a mis-parsed period is
the kind of failure that otherwise has no symptom.

`sample_data/delivery_history_example.csv` is a runnable example. Its market prices are
real; **its disruption rates are synthetic**, generated from a known elasticity so the
fitting path can be exercised without anyone's operational data. It is a demonstration
of the mechanism, not evidence about any real supply chain.

### Pricing and exposure inputs are part of a saved portfolio

`copilot_state.PRICING_KEYS` round-trips the derivation inputs, the price source, and
both feed toggles; the exposure table is a fifth persisted frame alongside nodes,
bundles, dependencies and correlations. Without this, reopening a saved portfolio would
quietly revert the price of risk to the sidebar default and change every currency figure
while the plan, the budget and the reproducibility check all looked untouched. These
keys are persisted but **not** proposable: no copilot rule targets them, and a test
asserts that, because a sentence should not be able to move every number in the app by
rewriting the revenue figure.

## Still missing

- **A price of risk validated against realised losses.** The derivation is now built
  from figures a finance function owns, which makes it arguable; it does not make it
  right. Back-testing the implied cost of an event against the company's own
  post-mortems is the next thing that would turn an assumption into a calibration.
  Elasticities are now fitted from history (below); the price of risk still is not.
- **Causal identification for the fitted elasticities.** What the calibration measures
  is an association between a market price and a delivery outcome. A hurricane that
  lifts crude and closes a port is attributed entirely to the price. Every fit carries
  this caveat in its own notes; no amount of extra data removes it, only a design that
  exploits something closer to an experiment would.
- **An intervention-effectiveness link.** Delivery records measure the risk a node
  *carries*. Using a fitted elasticity as a multiplier on an intervention's benefit
  assumes the intervention averts a constant *fraction* of that node's risk. That
  assumption is stated, not tested.
- **Supabase row-level persistence.** Auth can use Supabase, but portfolios are always
  stored in SQLite; a `SupabasePortfolioStore` would need RLS policies to match the
  owner scoping enforced here.
- **Multi-user concurrency.** SQLite with one connection per server is right for a
  single-team deployment and wrong for many concurrent writers.
- **Copilot coverage of the price book and macro toggle,** deliberately excluded: a
  sentence should not be able to assert where a price came from.
