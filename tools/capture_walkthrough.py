"""Drive the market exposure + elasticity calibration workflow and capture it.

Runs against the local Streamlit server. Every step is an action a user would take:
load a saved portfolio, expand the evidence tables, press the calibrate button, adopt a
fit, read the provenance row. Nothing is stubbed and no session state is written
directly, so what the screenshots show is the real app doing the real thing.
"""

from __future__ import annotations

import os
import pathlib
import sys

from playwright.sync_api import sync_playwright

# Overridable so this runs somewhere other than the machine it was written on. CHROME is
# set explicitly because a pinned Playwright build and a cached browser build can disagree,
# and pointing at the cached binary is faster than re-downloading one.
CHROME = os.environ.get("WALKTHROUGH_CHROME", "")
OUT = pathlib.Path(os.environ.get("WALKTHROUGH_OUT", "walkthrough-shots"))
OUT.mkdir(parents=True, exist_ok=True)
URL = os.environ.get("WALKTHROUGH_URL", "http://localhost:8501/")


def settle(pg, timeout: int = 120_000) -> None:
    """Wait until Streamlit is not running a script.

    The status widget appears while a rerun is in flight. Waiting on it rather than on a
    fixed sleep is what keeps the calibration step (a real network call) from being
    screenshotted half-finished.
    """
    pg.wait_for_timeout(1200)
    try:
        pg.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=timeout)
    except Exception:
        pass
    pg.wait_for_timeout(2500)


def expander(pg, label: str, index: int = 0):
    """The expander whose summary contains `label`, expanded.

    Located via the summary and then its parent `details`, because "Market exposure"
    labels two different expanders — the input table and the market panel — and `index`
    is how they are told apart.
    """
    summary = pg.locator("summary", has_text=label).nth(index)
    summary.wait_for(timeout=60_000)
    node = summary.locator("xpath=..")
    if node.get_attribute("open") is None:
        summary.scroll_into_view_if_needed()
        summary.click()
        pg.wait_for_timeout(1500)
    return node


def shot(target, name: str) -> None:
    path = OUT / f"{name}.png"
    target.screenshot(path=str(path))
    print(f"  captured {path.name}")


def main() -> None:
    with sync_playwright() as p:
        launch = {"args": ["--no-sandbox"]}
        if CHROME:
            launch["executable_path"] = CHROME
        browser = p.chromium.launch(**launch)
        pg = browser.new_page(viewport={"width": 1600, "height": 1500}, device_scale_factor=2)
        pg.goto(URL, wait_until="domcontentloaded")
        pg.wait_for_selector('[data-testid="stSidebar"]', timeout=120_000)
        settle(pg)

        print("1. loading the saved portfolio")
        selected = pg.locator('[data-testid="stSelectbox"] input').first.get_attribute("value")
        print(f"  selected: {selected}")
        assert selected and "Market exposure demo" in selected, selected
        pg.get_by_role("button", name="Load", exact=True).click()
        settle(pg)

        print("2. market exposure panel")
        exposure = expander(pg, "Market exposure", 0)
        shot(exposure, "01-market-exposure-table")
        market = expander(pg, "Market exposure", 1)
        shot(market, "02-market-panel-live-quotes")

        print("3. delivery history")
        history = expander(pg, "Delivery history")
        shot(history, "03-delivery-history")

        print("4. running the calibration")
        panel = expander(pg, "Elasticity calibration")
        shot(panel, "04-calibration-before-running")
        pg.get_by_role("button", name="Calibrate elasticities from delivery history").click()
        settle(pg)
        panel = expander(pg, "Elasticity calibration")
        shot(panel, "05-calibration-results")
        # The caveats are the point of the panel, so open the first one rather than
        # photographing a collapsed control and calling it disclosure.
        caveats = expander(pg, "What this fit does and does not establish", 0)
        shot(caveats, "05c-what-the-fit-establishes")
        print("--- CAVEATS ---")
        print(caveats.inner_text())
        print("--- CALIBRATION PANEL TEXT ---")
        print(panel.inner_text())
        print("--- END ---")

        print("5. adopting the fits")
        boxes = panel.locator('[data-testid="stCheckbox"] input')
        count = boxes.count()
        print(f"  adopt checkboxes found: {count}")
        for index in range(count):
            boxes.nth(index).click(force=True)
            settle(pg)
        panel = expander(pg, "Elasticity calibration")
        shot(panel, "05b-fits-ticked-not-yet-adopted")
        # Ticking a box is not adopting. The write happens only on this button, so that a
        # rerun cannot adopt anything on its own.
        pg.get_by_role("button", name="Adopt selected elasticities").click()
        settle(pg)
        panel = expander(pg, "Elasticity calibration")
        shot(panel, "06-calibration-adopted")
        exposure = expander(pg, "Market exposure", 0)
        shot(exposure, "07-exposure-after-adoption")
        market = expander(pg, "Market exposure", 1)
        shot(market, "07b-market-panel-after-adoption")
        stored = expander(pg, "Stored calibrations")
        shot(stored, "08-stored-calibrations")

        print("6. verification tab")
        pg.get_by_role("tab", name="Verification").click()
        settle(pg)
        body = pg.inner_text("body")
        for line in body.splitlines():
            low = line.lower()
            if "calibrated" in low or "asserted" in low or "elasticit" in low:
                print("  PROVENANCE:", line.strip())
        # The provenance table is `st.dataframe`, which renders to a canvas: there is no
        # DOM text to match on, so it is located positionally (last grid on the tab) and
        # captured as an element. The numbers in it are checked separately in Python.
        grids = pg.locator('[data-testid="stDataFrame"]')
        print(f"  grids on tab: {grids.count()}")
        grid = grids.last
        grid.scroll_into_view_if_needed()
        pg.wait_for_timeout(2000)
        shot(grid, "09-verification-provenance-top")
        # The grid has a fixed height and scrolls internally, so the elasticity row sits
        # below the fold of its own viewport. Wheel inside it rather than the page.
        grid.hover()
        pg.mouse.wheel(0, 320)
        pg.wait_for_timeout(2000)
        shot(grid, "09-verification-provenance")
        assumptions = pg.get_by_role("heading", name="Assumptions in force").first
        assumptions.scroll_into_view_if_needed()
        pg.wait_for_timeout(1200)
        shot(pg, "09b-assumptions-in-force")

        pg.screenshot(path=str(OUT / "10-full-page.png"), full_page=True)
        print("  captured 10-full-page.png")
        browser.close()


if __name__ == "__main__":
    sys.exit(main())
