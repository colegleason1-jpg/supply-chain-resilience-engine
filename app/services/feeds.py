"""Where market prices come from, stated explicitly and swappable.

The acquired system had one hard-coded provider requiring an API key that was never
set in any environment it ran in (F13), so the feed reported "Offline (Holding Last
Known)" over a hard-coded 100.00 forever. Two things follow from that, and both are
design constraints rather than preferences.

**The provider is a seam, not a call site.** Fetching lived inline inside the
Streamlit callback, which is why it could not be tested and why nobody noticed it had
never returned a number. Here a provider is an object with two methods, and the tests
use a third implementation that never opens a socket.

**The default provider needs no credential.** A feed that only works once somebody
finds a key is a feed that does not work. The default reads a public endpoint, so a
fresh checkout has a live market on first run and an inert exposure means something
is genuinely wrong rather than merely unconfigured.

A caveat that belongs in the code rather than only in a README: the default provider
reads Yahoo Finance's chart endpoint, which is public but undocumented and carries no
service commitment. It is right for evaluation and wrong for a production
installation, where ``SCRCAE_MARKET_PROVIDER=api-ninjas`` with a contracted key — or a
provider written against whatever the customer already licenses — is the answer. The
seam exists so that swap is a configuration change.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence, runtime_checkable

__all__ = [
    "DEFAULT_PROVIDER",
    "PROVIDER_ENV_VAR",
    "PricePoint",
    "PriceHistory",
    "QuoteProvider",
    "SymbolReading",
    "ApiNinjasProvider",
    "StaticProvider",
    "YahooChartProvider",
    "resolve_provider",
]

PROVIDER_ENV_VAR = "SCRCAE_MARKET_PROVIDER"
DEFAULT_PROVIDER = "yahoo"


@dataclass(frozen=True, slots=True)
class SymbolReading:
    """A provider's answer for one symbol: a price, or the reason there isn't one.

    Deliberately not a ``MarketQuote``: providers report what they saw and the market
    service decides what that means. Keeping the two apart is also what stops the
    import cycle between this module and ``services.market``.
    """

    symbol: str
    price: float | None
    status: str

    @property
    def ok(self) -> bool:
        return self.price is not None


@dataclass(frozen=True, slots=True)
class PricePoint:
    """One historical close, labelled with the period it belongs to.

    ``period`` is a ``YYYY-MM`` month key. Monthly is the granularity a company's
    delivery records actually come in, and matching a daily close to a quarter of
    delivery performance would be a false precision.
    """

    period: str
    price: float


@dataclass(frozen=True, slots=True)
class PriceHistory:
    """A symbol's monthly price history, or an explanation of its absence."""

    symbol: str
    points: tuple[PricePoint, ...] = field(default_factory=tuple)
    status: str = "Loaded"

    @property
    def ok(self) -> bool:
        return bool(self.points)

    def by_period(self) -> dict[str, float]:
        return {point.period: point.price for point in self.points}


@runtime_checkable
class QuoteProvider(Protocol):
    """What the app requires of a market data source.

    Neither method may raise. A provider that cannot answer says so in a status
    string, because a traceback in a Streamlit callback takes the whole page down and
    the user loses the portfolio they were editing.
    """

    name: str
    symbol_help: str

    def latest(self, symbols: Sequence[str], *, timeout: float = 4.0) -> tuple[SymbolReading, ...]: ...

    def history(self, symbol: str, *, months: int = 60, timeout: float = 8.0) -> PriceHistory: ...


def _month_key(epoch_seconds: float) -> str:
    import datetime as _dt

    moment = _dt.datetime.fromtimestamp(float(epoch_seconds), tz=_dt.timezone.utc)
    return f"{moment.year:04d}-{moment.month:02d}"


def _usable(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


class YahooChartProvider:
    """Keyless public quotes and monthly history from Yahoo Finance's chart endpoint.

    One endpoint serves both purposes, which is why it is the default: the price used
    to build today's overlay and the history used to calibrate the elasticity come
    from the same series. If they came from different sources, an elasticity measured
    against one definition of "the price of copper" would be applied to another, and
    nothing in the app would notice.
    """

    name = "yahoo"
    symbol_help = (
        "Yahoo Finance ticker, e.g. BZ=F Brent crude, CL=F WTI, HG=F copper, "
        "ZC=F corn, ^SPX S&P 500."
    )

    _CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    # Yahoo rejects requests without a browser-ish agent. Stated plainly rather than
    # hidden in a constant called HEADERS: relying on this is part of why this
    # provider is fine for evaluation and not for production.
    _HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; scrcae/1.0)"}

    def _get(self, symbol: str, *, params: dict[str, str], timeout: float) -> tuple[dict | None, str]:
        try:
            import requests

            response = requests.get(
                self._CHART.format(symbol=symbol),
                params=params,
                headers=self._HEADERS,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in the status line
            return None, f"Unavailable ({type(exc).__name__})"
        if response.status_code != 200:
            return None, f"Feed returned HTTP {response.status_code}"
        try:
            payload = response.json()
            result = payload["chart"]["result"][0]
        except Exception:  # noqa: BLE001
            return None, "Feed returned an unrecognised response"
        if not isinstance(result, dict):
            return None, "Feed returned an unrecognised response"
        return result, "Live"

    def latest(self, symbols: Sequence[str], *, timeout: float = 4.0) -> tuple[SymbolReading, ...]:
        readings: list[SymbolReading] = []
        for symbol in symbols:
            result, status = self._get(
                symbol, params={"interval": "1d", "range": "5d"}, timeout=timeout
            )
            if result is None:
                readings.append(SymbolReading(symbol=symbol, price=None, status=status))
                continue
            price = _usable((result.get("meta") or {}).get("regularMarketPrice"))
            if price is None:
                readings.append(
                    SymbolReading(
                        symbol=symbol,
                        price=None,
                        status="Feed returned no usable price for this symbol",
                    )
                )
                continue
            readings.append(SymbolReading(symbol=symbol, price=price, status="Live"))
        return tuple(readings)

    def history(self, symbol: str, *, months: int = 60, timeout: float = 8.0) -> PriceHistory:
        span = max(1, min(120, int(months)))
        result, status = self._get(
            symbol,
            params={"interval": "1mo", "range": f"{max(1, span // 12 + 1)}y"},
            timeout=timeout,
        )
        if result is None:
            return PriceHistory(symbol=symbol, status=status)

        stamps = result.get("timestamp") or []
        try:
            closes = result["indicators"]["quote"][0]["close"]
        except Exception:  # noqa: BLE001
            return PriceHistory(symbol=symbol, status="Feed returned no closing prices")

        points: list[PricePoint] = []
        for stamp, close in zip(stamps, closes):
            price = _usable(close)
            if price is None:
                # A gap, not a zero. Skipped rather than interpolated: an invented
                # price would become an invented observation in the calibration.
                continue
            points.append(PricePoint(period=_month_key(stamp), price=price))

        # Later readings win, so a partial current month is represented by its latest
        # available close rather than by whichever row the feed happened to emit first.
        deduped = {point.period: point.price for point in points}
        ordered = tuple(
            PricePoint(period=period, price=price)
            for period, price in sorted(deduped.items())
        )[-span:]
        if not ordered:
            return PriceHistory(symbol=symbol, status="Feed returned no usable history")
        return PriceHistory(symbol=symbol, points=ordered, status="Loaded")


class ApiNinjasProvider:
    """The keyed commodity endpoint the acquired system used.

    Kept because a contracted feed is the right answer for a real installation, and
    because deleting it would make the seam theoretical. It offers spot prices only,
    so ``history`` refuses rather than fabricating a series — which means an
    installation on this provider can run the overlay but cannot calibrate, and the
    UI says exactly that instead of showing an empty chart.
    """

    name = "api-ninjas"
    symbol_help = "API Ninjas commodity name, e.g. brent_crude_oil, copper, corn."
    key_env_var = "SCRCAE_MARKET_API_KEY"

    _ENDPOINT = "https://api.api-ninjas.com/v1/commodityprice"

    def latest(self, symbols: Sequence[str], *, timeout: float = 4.0) -> tuple[SymbolReading, ...]:
        api_key = os.environ.get(self.key_env_var, "").strip()
        if not api_key:
            return tuple(
                SymbolReading(
                    symbol=symbol,
                    price=None,
                    status=f"No credential for this feed (set {self.key_env_var})",
                )
                for symbol in symbols
            )

        readings: list[SymbolReading] = []
        for symbol in symbols:
            try:
                import requests

                response = requests.get(
                    self._ENDPOINT,
                    params={"name": symbol.lower()},
                    headers={"X-Api-Key": api_key},
                    timeout=timeout,
                )
                if response.status_code != 200:
                    readings.append(
                        SymbolReading(
                            symbol=symbol,
                            price=None,
                            status=f"Feed returned HTTP {response.status_code}",
                        )
                    )
                    continue
                price = _usable(response.json().get("price"))
            except Exception as exc:  # noqa: BLE001
                readings.append(
                    SymbolReading(
                        symbol=symbol, price=None, status=f"Unavailable ({type(exc).__name__})"
                    )
                )
                continue
            if price is None:
                readings.append(
                    SymbolReading(
                        symbol=symbol,
                        price=None,
                        status="Feed returned an unusable price",
                    )
                )
                continue
            readings.append(SymbolReading(symbol=symbol, price=price, status="Live"))
        return tuple(readings)

    def history(self, symbol: str, *, months: int = 60, timeout: float = 8.0) -> PriceHistory:
        return PriceHistory(
            symbol=symbol,
            status=(
                "This feed provides spot prices only, so elasticities cannot be "
                "calibrated against it. Switch the provider or supply history directly."
            ),
        )


class StaticProvider:
    """Prices supplied in the call, for tests and for reproducible demonstrations.

    Named for what it is. The point of the acquired 100.00 placeholder was not that a
    fixed price is always wrong — it is that a fixed price presented as a live one is.
    Anything reading this provider knows the numbers were handed to it.
    """

    name = "static"
    symbol_help = "Any symbol present in the supplied price map."

    def __init__(
        self,
        prices: dict[str, float | None] | None = None,
        histories: dict[str, Iterable[tuple[str, float]]] | None = None,
    ) -> None:
        self._prices = {str(k).upper(): v for k, v in (prices or {}).items()}
        self._histories = {
            str(symbol).upper(): tuple(
                PricePoint(period=str(period), price=float(price))
                for period, price in series
            )
            for symbol, series in (histories or {}).items()
        }

    def latest(self, symbols: Sequence[str], *, timeout: float = 4.0) -> tuple[SymbolReading, ...]:
        readings = []
        for symbol in symbols:
            key = str(symbol).upper()
            if key not in self._prices:
                readings.append(
                    SymbolReading(
                        symbol=symbol, price=None, status="No price supplied for this symbol"
                    )
                )
                continue
            price = _usable(self._prices[key])
            readings.append(
                SymbolReading(
                    symbol=symbol,
                    price=price,
                    status="Supplied" if price is not None else "Supplied as unavailable",
                )
            )
        return tuple(readings)

    def history(self, symbol: str, *, months: int = 60, timeout: float = 8.0) -> PriceHistory:
        points = self._histories.get(str(symbol).upper())
        if not points:
            return PriceHistory(symbol=symbol, status="No history supplied for this symbol")
        ordered = tuple(sorted(points, key=lambda p: p.period))[-max(1, int(months)):]
        return PriceHistory(symbol=symbol, points=ordered, status="Supplied")


_REGISTRY: dict[str, type] = {
    YahooChartProvider.name: YahooChartProvider,
    ApiNinjasProvider.name: ApiNinjasProvider,
}


def resolve_provider(name: str | None = None) -> QuoteProvider:
    """The configured provider, defaulting to the keyless one.

    An unknown name falls back to the default rather than raising, and the fallback is
    visible because the returned provider's ``name`` is displayed in the market panel.
    A typo in an environment variable should not stop a CFO opening the app.
    """
    requested = (name or os.environ.get(PROVIDER_ENV_VAR) or DEFAULT_PROVIDER).strip().lower()
    factory = _REGISTRY.get(requested, _REGISTRY[DEFAULT_PROVIDER])
    return factory()  # type: ignore[return-value]
