# tools

Two scripts that exist so the market exposure and calibration workflow can be shown, not
described. Neither is imported by the app or the engine, and nothing here runs in the test
suite.

## `seed_demo_portfolio.py`

Writes one saved portfolio — `Market exposure demo (synthetic delivery history)` — into the
local store, with two market exposures (`N1`/`BZ=F`, `N3`/`HG=F`) and 120 rows of delivery
history read from `app/sample_data/delivery_history_example.csv`.

The two elasticities are seeded at **0.30 deliberately, and both are wrong**. The history
supports roughly 1.21 and 0.56, so a demo that begins from this portfolio shows the
calibration disagreeing with the stored assumption instead of agreeing with it, which is
the only version of the demo worth watching.

```bash
PYTHONPATH=.:engine/src python tools/seed_demo_portfolio.py
```

Writes to `$SCRCAE_DB_PATH` (default `portfolios.db` in the working directory). Run it from
the same directory the Streamlit server was started in, or the server will read a different
file and show no saved portfolios.

## `capture_walkthrough.py`

Drives a running server with Playwright and writes the screenshots in
`app/docs/walkthrough/`. It performs the actual clicks: load the portfolio, open the
evidence tables, press **Calibrate elasticities from delivery history**, tick the adopt
boxes, press **Adopt selected elasticities**, then read the provenance row on the
Verification tab. No session state is written directly, so what it captures is the app
doing the thing rather than a mock of it.

```bash
PYTHONPATH=.:engine/src python -m streamlit run app/main.py --server.port 8501 &
WALKTHROUGH_OUT=app/docs/walkthrough python tools/capture_walkthrough.py
```

Two details worth keeping in mind if it breaks:

- `st.dataframe` renders to a **canvas**, so its cells have no DOM text. The provenance
  grid is located positionally and captured as an element, and it scrolls internally — the
  elasticity row needs a wheel event inside the grid, not a page scroll.
- `st.data_editor` is the same canvas widget, which is why the demo seeds a portfolio
  instead of typing into the exposure table.
