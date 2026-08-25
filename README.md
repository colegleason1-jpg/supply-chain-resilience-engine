# Supply Chain Resilience & Capital Allocation Engine

[![tests](https://github.com/colegleason1-jpg/supply-chain-resilience-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/colegleason1-jpg/supply-chain-resilience-engine/actions/workflows/tests.yml)

Decides how to spend a fixed resilience budget across supply chain interventions, and — the
part that matters — is able to show why. Every currency figure traces to a stated input,
every assumption that is an assumption says so on screen, and the numbers that were measured
are separated from the numbers somebody typed.

It replaces an acquired single-file system whose eighteen catalogued defects included a
solver that reported an infeasible problem as optimal, a correlation repair that silently
rescaled every simulated node, and a market mapping that matched everything when a column
was missing. All eighteen are now pinned by tests. The reasoning is in
[`engine/docs/ADR-001-objective-reformulation.md`](engine/docs/ADR-001-objective-reformulation.md).

## Two pieces

| | What it is | Depends on Streamlit? |
| --- | --- | --- |
| [`engine/`](engine/README.md) | `scrcae` — the headless optimiser, simulator, and calibration estimators. Importable, installable, testable without a UI. | no |
| [`app/`](app/README.md) | The Streamlit client. Editors, presenters, persistence, auth, and the audit ledger. A consumer of the engine. | yes |

The separation is not decoration. It is what makes the solver exercisable headlessly, and it
is why there are 672 tests instead of a manual checklist.

## Running it

```bash
pip install -r requirements-dev.txt
python -m pytest -q                     # 672 tests
python -m streamlit run app/main.py
```

Installing `requirements.txt` puts `engine/` on the path as a real package, so `import
scrcae` works without `PYTHONPATH`. Needs Python 3.12 or newer.

## Seeing the interesting part

Market exposure adjusts an intervention's effectiveness by how far a commodity has moved
from its anchor price. That adjustment needs a risk elasticity, and an elasticity somebody
typed into a box is an assumption wearing the costume of a measurement.

So the app fits it from delivery history, and refuses to when the history cannot support it:

```
Entered 0.30, fitted 1.21 — the entered value is 0.25× the fit, below it.
Elasticity 1.21 (95% interval 1.03 to 1.44), R² 0.73, from 60 periods of which 30 above anchor.
```

Adopting that fit moves the audit ledger's provenance row from `2 asserted` to
`2 calibrated`, and the count is derived by matching each exposure against a stored fit —
overtype the number and it reverts to asserted on the next run. You cannot type your way to
the word "calibrated".

Screenshots of the whole path are in [`app/docs/walkthrough/`](app/docs/walkthrough/), and
`tools/` holds the scripts that produced them, so it is reproducible rather than a one-off.

## Where the old system went

[`legacy/`](legacy/README.md) holds the acquired single-file system and the forensic audit of
it. Nothing there is imported by the app or the engine — it is kept because the F1–F18 defect
citations in ADR-001 point into it, and a claim about fixed defects is worth more when the
code that contained them is still readable.

## Deploying

See [`DEPLOYING.md`](DEPLOYING.md). Short version: it is a Streamlit server, so the host must
be able to upgrade WebSockets; GitHub → Streamlit Community Cloud with `app/main.py` as the
main file and Python 3.12 is the fast path. Read the three warnings about ephemeral storage,
default-off authentication, and the default market provider before sharing a public link.

## What it does not claim

Stated here rather than buried, because a tool that hides its own limits is the failure mode
this rebuild exists to correct:

- The risk response exponent (0.85) is inherited and uncalibrated. Delivery history cannot
  fix it — that needs variation in funding levels, not market data.
- Fitted elasticities are associations, not causal effects, and they measure risk a node
  *carries* rather than risk an intervention *removes*.
- The reported 95% interval measures closer to 90% under realistic price persistence. That
  measurement, and the Monte Carlo runs behind it, are in the estimator's module docstring.
