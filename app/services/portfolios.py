"""Saving, loading and listing portfolios, on behalf of one principal.

This service is the only thing the views talk to, and it holds the owner. Views
therefore cannot pass an owner id at all, let alone the wrong one — the scoping is
structural rather than a rule someone has to remember at every call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.auth.models import Principal
from app.storage.models import RunProvenance, SavedPortfolio
from app.storage.serialization import records_to_tables, tables_to_records

#: Column order for each editor, so an empty saved table reloads with usable headers.
TABLE_COLUMNS: dict[str, list[str]] = {}


def _columns() -> dict[str, list[str]]:
    global TABLE_COLUMNS
    if not TABLE_COLUMNS:
        from app import adapters

        TABLE_COLUMNS = {
            "nodes": list(adapters.default_nodes_frame().columns),
            "bundles": list(adapters.default_bundles_frame().columns),
            "dependencies": list(adapters.default_dependencies_frame().columns),
            "correlation": [],  # width depends on the node set
            "exposure": list(adapters.EXPOSURE_COLUMNS),
            # The evidence tables. Their columns are declared for the same reason as
            # every other table's: an empty saved calibration set must reload with
            # headers, because a headerless frame cannot be parsed back into fits and
            # every elasticity it vouched for would revert to asserted.
            "history": list(adapters.HISTORY_COLUMNS),
            "calibration": list(adapters.CALIBRATION_COLUMNS),
        }
    return TABLE_COLUMNS


@dataclass(frozen=True, slots=True)
class LoadedPortfolio:
    """A portfolio brought back from storage, ready to populate the editors."""

    portfolio: SavedPortfolio
    tables: dict[str, pd.DataFrame]
    settings: dict[str, Any]

    @property
    def name(self) -> str:
        return self.portfolio.name


class PortfolioService:
    def __init__(self, store, principal: Principal):
        self._store = store
        self._principal = principal

    @property
    def owner_label(self) -> str:
        return self._principal.label

    def list(self) -> list[SavedPortfolio]:
        return self._store.list_for_owner(self._principal.user_id)

    def save(
        self,
        name: str,
        *,
        tables: dict[str, pd.DataFrame | None],
        settings: dict[str, Any],
        result=None,
    ) -> SavedPortfolio:
        """Persist inputs, plus the provenance of the run being looked at.

        ``result`` is used only for its audit hashes. Its allocations and risk figures
        are deliberately not stored: they are derived, and a stored derived figure
        becomes a false claim the moment the engine changes. See
        `storage/models.py`.
        """
        clean_name = name.strip()
        existing = self._store.get_by_name(clean_name, owner_id=self._principal.user_id)

        # Saving under an existing name updates that portfolio rather than creating a
        # second one, so the id and the original creation time are carried over.
        identity: dict[str, Any] = {}
        if existing is not None:
            identity = {
                "portfolio_id": existing.portfolio_id,
                "created_at": existing.created_at,
            }

        portfolio = SavedPortfolio(
            name=clean_name,
            owner_id=self._principal.user_id,
            tables=tables_to_records(tables),
            settings=dict(settings),
            provenance=(
                RunProvenance.from_result(result) if result is not None else RunProvenance()
            ),
            **identity,
        )
        return self._store.save(portfolio)

    def load(self, portfolio_id: str) -> LoadedPortfolio | None:
        portfolio = self._store.get(portfolio_id, owner_id=self._principal.user_id)
        if portfolio is None:
            return None
        return LoadedPortfolio(
            portfolio=portfolio,
            tables=records_to_tables(portfolio.tables, _columns()),
            settings=dict(portfolio.settings),
        )

    def delete(self, portfolio_id: str) -> bool:
        return self._store.delete(portfolio_id, owner_id=self._principal.user_id)
