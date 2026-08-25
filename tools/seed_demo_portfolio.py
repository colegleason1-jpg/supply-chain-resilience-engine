"""Seed a saved portfolio so the calibration workflow can be walked in a browser.

Uses the real persistence path — `PortfolioService.save` writing to the same SQLite
file the app reads — rather than reaching into Streamlit session state. What the browser
then does is the ordinary "load a saved portfolio" journey a user would do.

The market prices behind the delivery history are real. The disruption rates are
synthetic, generated from a known elasticity. That distinction is loud in the portfolio
name and in the sample CSV's own documentation.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, "/home/user/workspace/projects/supply-chain-app-fP0HGOQmQ0qJtsvQhDx6xA/files")

from app import adapters  # noqa: E402
from app.auth.models import ANONYMOUS  # noqa: E402
from app.services.portfolios import PortfolioService  # noqa: E402
from app.storage.sqlite_store import SqlitePortfolioStore  # noqa: E402

DB = os.environ.get("SCRCAE_DB_PATH", "portfolios.db")

# Two of the five default interventions are given a market exposure. The elasticities
# entered here are deliberately *wrong* — 0.30 against fitted values near 1.2 and 0.55 —
# so the walkthrough shows the calibration disagreeing with an assumption, which is the
# entire point of the feature.
exposure = pd.DataFrame(
    [
        {
            adapters.COL_EXP_NODE_ID: "N1",
            adapters.COL_EXP_SYMBOL: "BZ=F",
            adapters.COL_EXP_ANCHOR: 79.0,
            adapters.COL_EXP_ELASTICITY: 0.30,
        },
        {
            adapters.COL_EXP_NODE_ID: "N3",
            adapters.COL_EXP_SYMBOL: "HG=F",
            adapters.COL_EXP_ANCHOR: 4.0,
            adapters.COL_EXP_ELASTICITY: 0.30,
        },
    ],
    columns=list(adapters.EXPOSURE_COLUMNS),
)

history = pd.read_csv(
    "/home/user/workspace/projects/supply-chain-app-fP0HGOQmQ0qJtsvQhDx6xA/files"
    "/app/sample_data/delivery_history_example.csv"
)
history[adapters.COL_HIST_NODE_ID] = history[adapters.COL_HIST_NODE_ID].map(
    {"SUP-DUAL": "N1", "LOG-REROUTE": "N3"}
)

nodes = adapters.default_nodes_frame()

service = PortfolioService(SqlitePortfolioStore(DB), ANONYMOUS)
saved = service.save(
    "Market exposure demo (synthetic delivery history)",
    tables={
        "nodes": nodes,
        "bundles": adapters.default_bundles_frame(),
        "dependencies": adapters.default_dependencies_frame(),
        "correlation": None,
        "exposure": exposure,
        "history": history,
        "calibration": adapters.default_calibration_frame(),
    },
    settings={"apply_market_exposure": True},
)
print("saved:", saved.name, saved.portfolio_id)
print("history rows:", len(history), "| exposure rows:", len(exposure))
