"""SQLite-backed persistence. The default store, and the one the tests exercise.

Chosen as the default deliberately. The acquired system's only store was
`st.session_state`, which meant closing the tab discarded the portfolio, and its
intended replacement was a hosted Supabase instance nobody could run locally. A file
on disk with no credentials and no network makes persistence testable, which is what
turns it from a claim into a property.

Every query is parameterised. The acquired system had a `sanitize_input()` function
that stripped characters from user strings before interpolating them into queries,
which is not how injection is prevented — it is a filter that has to be perfect
forever against an adversary who only needs to be right once. Parameter binding makes
the question moot: the driver never parses user data as SQL. Intervention names are
therefore stored exactly as typed, apostrophes and all.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.storage.models import RunProvenance, SavedPortfolio, StorageError
from app.storage.serialization import isoformat, parse_datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id   TEXT PRIMARY KEY,
    owner_id       TEXT NOT NULL,
    name           TEXT NOT NULL,
    tables_json    TEXT NOT NULL,
    settings_json  TEXT NOT NULL,
    engine_version TEXT NOT NULL DEFAULT '',
    input_hash     TEXT NOT NULL DEFAULT '',
    output_hash    TEXT NOT NULL DEFAULT '',
    solver_status  TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    -- One name per owner, so "save" is idempotent under a name the user reuses and
    -- they cannot end up with two different portfolios both called "Q3 plan".
    UNIQUE (owner_id, name)
);
CREATE INDEX IF NOT EXISTS portfolios_by_owner
    ON portfolios (owner_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS sign_in_attempts (
    email        TEXT NOT NULL,
    attempted_at TEXT NOT NULL,
    succeeded    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS attempts_by_email
    ON sign_in_attempts (email, attempted_at DESC);
"""


class SqlitePortfolioStore:
    """Portfolios on disk, scoped by owner at the query level."""

    def __init__(self, path: str | Path = "portfolios.db") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # `check_same_thread=False` because Streamlit runs script reruns on worker
        # threads; access is serialised by SQLite's own locking.
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    # ---------------------------------------------------------------- helpers

    def close(self) -> None:
        self._connection.close()

    @staticmethod
    def _to_portfolio(row: sqlite3.Row) -> SavedPortfolio:
        return SavedPortfolio(
            portfolio_id=row["portfolio_id"],
            owner_id=row["owner_id"],
            name=row["name"],
            tables=json.loads(row["tables_json"]),
            settings=json.loads(row["settings_json"]),
            provenance=RunProvenance(
                engine_version=row["engine_version"],
                input_hash=row["input_hash"],
                output_hash=row["output_hash"],
                solver_status=row["solver_status"],
                saved_at=parse_datetime(row["updated_at"]),
            ),
            created_at=parse_datetime(row["created_at"]),
            updated_at=parse_datetime(row["updated_at"]),
        )

    # ------------------------------------------------------------ portfolios

    def save(self, portfolio: SavedPortfolio) -> SavedPortfolio:
        """Insert or update. Reusing a name overwrites that portfolio in place.

        The upsert targets ``(owner_id, name)`` rather than the primary key, because
        the user's mental model of "save" is the name they typed, not a uuid they never
        see. Without this, saving twice under one name would either fail or silently
        create a second portfolio indistinguishable from the first in the list.
        """
        stored = portfolio.touched()
        try:
            self._connection.execute(
                """
                INSERT INTO portfolios (
                    portfolio_id, owner_id, name, tables_json, settings_json,
                    engine_version, input_hash, output_hash, solver_status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (owner_id, name) DO UPDATE SET
                    tables_json    = excluded.tables_json,
                    settings_json  = excluded.settings_json,
                    engine_version = excluded.engine_version,
                    input_hash     = excluded.input_hash,
                    output_hash    = excluded.output_hash,
                    solver_status  = excluded.solver_status,
                    updated_at     = excluded.updated_at
                """,
                (
                    stored.portfolio_id,
                    stored.owner_id,
                    stored.name,
                    json.dumps(stored.tables),
                    json.dumps(stored.settings),
                    stored.provenance.engine_version,
                    stored.provenance.input_hash,
                    stored.provenance.output_hash,
                    stored.provenance.solver_status,
                    isoformat(stored.created_at),
                    isoformat(stored.updated_at),
                ),
            )
            self._connection.commit()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            raise StorageError(f"could not save portfolio {stored.name!r}: {exc}") from exc

        reloaded = self.get(stored.portfolio_id, owner_id=stored.owner_id)
        return reloaded or self.get_by_name(stored.name, owner_id=stored.owner_id) or stored

    def get(self, portfolio_id: str, *, owner_id: str) -> SavedPortfolio | None:
        """Fetch by id, scoped to the owner.

        The owner predicate is in the WHERE clause, not checked afterwards. A caller
        holding someone else's id gets ``None``, which is indistinguishable from the
        portfolio not existing — no oracle for probing.
        """
        row = self._connection.execute(
            "SELECT * FROM portfolios WHERE portfolio_id = ? AND owner_id = ?",
            (portfolio_id, owner_id),
        ).fetchone()
        return self._to_portfolio(row) if row else None

    def get_by_name(self, name: str, *, owner_id: str) -> SavedPortfolio | None:
        row = self._connection.execute(
            "SELECT * FROM portfolios WHERE name = ? AND owner_id = ?",
            (name, owner_id),
        ).fetchone()
        return self._to_portfolio(row) if row else None

    def list_for_owner(self, owner_id: str) -> list[SavedPortfolio]:
        rows = self._connection.execute(
            "SELECT * FROM portfolios WHERE owner_id = ? ORDER BY updated_at DESC",
            (owner_id,),
        ).fetchall()
        return [self._to_portfolio(row) for row in rows]

    def delete(self, portfolio_id: str, *, owner_id: str) -> bool:
        cursor = self._connection.execute(
            "DELETE FROM portfolios WHERE portfolio_id = ? AND owner_id = ?",
            (portfolio_id, owner_id),
        )
        self._connection.commit()
        return cursor.rowcount > 0

    # -------------------------------------------------------- sign-in attempts

    def record_attempt(self, email: str, *, succeeded: bool, at: Any = None) -> None:
        """Persist a sign-in attempt.

        This is in the database rather than session state on purpose. The acquired
        system counted failed attempts in `st.session_state`, so its five-attempt
        lockout was defeated by pressing refresh.
        """
        from datetime import UTC, datetime

        moment = at or datetime.now(UTC)
        self._connection.execute(
            "INSERT INTO sign_in_attempts (email, attempted_at, succeeded) VALUES (?, ?, ?)",
            (email.strip().lower(), isoformat(moment), int(succeeded)),
        )
        self._connection.commit()

    def recent_failures(self, email: str, *, since) -> int:
        row = self._connection.execute(
            """
            SELECT COUNT(*) AS failures FROM sign_in_attempts
            WHERE email = ? AND succeeded = 0 AND attempted_at >= ?
            """,
            (email.strip().lower(), isoformat(since)),
        ).fetchone()
        return int(row["failures"])

    def clear_attempts(self, email: str) -> None:
        self._connection.execute(
            "DELETE FROM sign_in_attempts WHERE email = ?", (email.strip().lower(),)
        )
        self._connection.commit()
