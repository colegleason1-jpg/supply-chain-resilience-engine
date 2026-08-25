"""Where prices come from, and how each provider fails.

No test here opens a socket. Every provider is driven through a stubbed ``requests``
module, because a test suite that needs the internet is a test suite that goes red
when a commodity exchange has a bad afternoon — and because the behaviour worth
pinning is precisely what happens when the feed misbehaves.
"""

from __future__ import annotations

import sys
import types

import pytest

from app.services import feeds


class FakeResponse:
    def __init__(self, status_code=200, payload=None, raises=None):
        self.status_code = status_code
        self._payload = payload
        self._raises = raises

    def json(self):
        if self._raises is not None:
            raise self._raises
        return self._payload


@pytest.fixture
def stub_requests(monkeypatch):
    """Install a fake ``requests`` for the duration of a test.

    The providers import ``requests`` inside the call rather than at module scope,
    which is what makes this possible — and that local import exists for exactly this
    reason: the app must start and its tests must run without the library present.
    """
    calls: list[dict] = []
    responses: list[object] = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params or {}, "headers": headers or {}})
        outcome = responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    module = types.ModuleType("requests")
    module.get = fake_get  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", module)
    return types.SimpleNamespace(calls=calls, responses=responses)


def chart_payload(price=None, stamps=None, closes=None):
    meta = {} if price is None else {"regularMarketPrice": price}
    result = {"meta": meta}
    if stamps is not None:
        result["timestamp"] = stamps
        result["indicators"] = {"quote": [{"close": closes}]}
    return {"chart": {"result": [result]}}


# Two mid-month instants in 2026-01 and 2026-02, UTC.
JAN = 1768000000
FEB = 1770500000


class TestTheDefaultProviderNeedsNoCredential:
    def test_the_configured_default_is_the_keyless_one(self, monkeypatch) -> None:
        """A feed that only works once somebody finds a key is a feed that does not work.

        The acquired system's provider required a key that was never set anywhere it
        ran, so the market panel reported "Offline (Holding Last Known)" over a
        hard-coded 100.00 for the whole life of the product (F13).
        """
        monkeypatch.delenv(feeds.PROVIDER_ENV_VAR, raising=False)

        assert isinstance(feeds.resolve_provider(), feeds.YahooChartProvider)

    def test_an_unknown_provider_name_falls_back_rather_than_raising(self, monkeypatch) -> None:
        """A typo in an environment variable must not stop a CFO opening the app.

        The fallback is visible because the market panel prints the provider's name
        beside every status, so a silent substitution is still a legible one.
        """
        monkeypatch.setenv(feeds.PROVIDER_ENV_VAR, "not-a-provider")

        assert feeds.resolve_provider().name == feeds.DEFAULT_PROVIDER

    def test_an_explicit_name_overrides_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv(feeds.PROVIDER_ENV_VAR, "yahoo")

        assert feeds.resolve_provider("api-ninjas").name == "api-ninjas"


class TestTheKeylessProviderReportsWhatItSaw:
    def test_a_good_response_yields_a_price(self, stub_requests) -> None:
        stub_requests.responses.append(FakeResponse(payload=chart_payload(price=87.35)))

        reading = feeds.YahooChartProvider().latest(["BZ=F"])[0]

        assert reading.price == pytest.approx(87.35)
        assert reading.ok

    @pytest.mark.parametrize(
        "outcome,expected",
        [
            (FakeResponse(status_code=404), "HTTP 404"),
            (FakeResponse(status_code=429), "HTTP 429"),
            (FakeResponse(payload={"chart": {"result": []}}), "unrecognised"),
            (FakeResponse(raises=ValueError("not json")), "unrecognised"),
            (TimeoutError("slow"), "TimeoutError"),
        ],
    )
    def test_every_failure_mode_is_named_and_none_produces_a_price(
        self, stub_requests, outcome, expected
    ) -> None:
        """"The symbol is wrong" and "the provider is down" need different actions.

        The acquired system collapsed both into one status string over a price it had
        never fetched, so neither action was ever taken.
        """
        stub_requests.responses.append(outcome)

        reading = feeds.YahooChartProvider().latest(["BZ=F"])[0]

        assert reading.price is None
        assert expected in reading.status

    @pytest.mark.parametrize("price", [0.0, -5.0, None, "n/a", float("inf")])
    def test_an_unusable_price_is_no_price(self, stub_requests, price) -> None:
        """Zero is the dangerous one: it is a number, and it would divide into a
        deviation of minus one hundred percent."""
        stub_requests.responses.append(FakeResponse(payload=chart_payload(price=price)))

        assert feeds.YahooChartProvider().latest(["BZ=F"])[0].price is None

    def test_one_broken_symbol_does_not_cost_the_others_their_readings(
        self, stub_requests
    ) -> None:
        stub_requests.responses.extend(
            [
                FakeResponse(payload=chart_payload(price=87.0)),
                FakeResponse(status_code=404),
                FakeResponse(payload=chart_payload(price=6.7)),
            ]
        )

        readings = feeds.YahooChartProvider().latest(["BZ=F", "NOPE", "HG=F"])

        assert [r.ok for r in readings] == [True, False, True]


class TestHistoryIsMonthlyAndHasNoInventedPoints:
    def test_closes_become_month_keyed_points_in_order(self, stub_requests) -> None:
        stub_requests.responses.append(
            FakeResponse(payload=chart_payload(stamps=[JAN, FEB], closes=[80.0, 90.0]))
        )

        history = feeds.YahooChartProvider().history("BZ=F")

        assert history.ok
        assert [p.period for p in history.points] == ["2026-01", "2026-02"]
        assert history.by_period() == {"2026-01": 80.0, "2026-02": 90.0}

    def test_a_missing_close_is_a_gap_not_a_zero(self, stub_requests) -> None:
        """Interpolating would invent an observation and then calibrate against it.

        A fabricated price in the history is worse than a fabricated quote: a quote
        skews today's overlay, while a fabricated history skews the elasticity that
        will be applied to every future run.
        """
        stub_requests.responses.append(
            FakeResponse(payload=chart_payload(stamps=[JAN, FEB], closes=[None, 90.0]))
        )

        history = feeds.YahooChartProvider().history("BZ=F")

        assert [p.period for p in history.points] == ["2026-02"]

    def test_a_response_with_no_closes_at_all_is_refused_with_a_reason(
        self, stub_requests
    ) -> None:
        stub_requests.responses.append(
            FakeResponse(payload=chart_payload(stamps=[JAN], closes=[None]))
        )

        history = feeds.YahooChartProvider().history("BZ=F")

        assert history.ok is False
        assert "no usable history" in history.status

    def test_a_repeated_month_keeps_the_later_reading(self, stub_requests) -> None:
        """So a part-finished current month is represented by its latest close."""
        stub_requests.responses.append(
            FakeResponse(
                payload=chart_payload(stamps=[JAN, JAN + 100, FEB], closes=[80.0, 82.0, 90.0])
            )
        )

        assert feeds.YahooChartProvider().history("BZ=F").by_period()["2026-01"] == 82.0

    def test_a_transport_failure_returns_an_empty_history_and_says_why(
        self, stub_requests
    ) -> None:
        stub_requests.responses.append(ConnectionError("no route"))

        history = feeds.YahooChartProvider().history("BZ=F")

        assert history.points == ()
        assert "ConnectionError" in history.status


class TestTheKeyedProviderIsHonestAboutWhatItCannotDo:
    def test_without_its_key_it_says_which_variable_is_missing(self, monkeypatch) -> None:
        monkeypatch.delenv("SCRCAE_MARKET_API_KEY", raising=False)

        reading = feeds.ApiNinjasProvider().latest(["copper"])[0]

        assert reading.price is None
        assert "SCRCAE_MARKET_API_KEY" in reading.status

    def test_it_sends_the_key_as_a_header_and_never_as_a_query_parameter(
        self, monkeypatch, stub_requests
    ) -> None:
        """A key in a query string ends up in access logs and in shared screenshots."""
        monkeypatch.setenv("SCRCAE_MARKET_API_KEY", "secret-key")
        stub_requests.responses.append(FakeResponse(payload={"price": 4.2}))

        feeds.ApiNinjasProvider().latest(["copper"])

        call = stub_requests.calls[0]
        assert call["headers"]["X-Api-Key"] == "secret-key"
        assert "secret-key" not in str(call["params"])

    def test_a_spot_only_feed_refuses_history_instead_of_faking_a_series(self) -> None:
        """An installation on this feed can run the overlay and cannot calibrate.

        Saying so is the whole point. Returning an empty series with a cheerful status
        would show a user an empty chart and let them conclude their history was the
        problem.
        """
        history = feeds.ApiNinjasProvider().history("copper")

        assert history.ok is False
        assert "spot prices only" in history.status


class TestTheStaticProviderIsNamedForWhatItIs:
    def test_supplied_prices_are_labelled_supplied_not_live(self) -> None:
        """A fixed price is not the sin; a fixed price presented as a live one is."""
        reading = feeds.StaticProvider(prices={"BZ=F": 90.0}).latest(["BZ=F"])[0]

        assert reading.price == 90.0
        assert reading.status == "Supplied"

    def test_a_symbol_with_no_supplied_price_is_not_given_one(self) -> None:
        reading = feeds.StaticProvider(prices={"BZ=F": 90.0}).latest(["HG=F"])[0]

        assert reading.price is None
        assert "No price supplied" in reading.status

    def test_supplied_history_is_returned_in_period_order(self) -> None:
        provider = feeds.StaticProvider(
            histories={"BZ=F": [("2026-02", 90.0), ("2026-01", 80.0)]}
        )

        assert [p.period for p in provider.history("BZ=F").points] == ["2026-01", "2026-02"]

    def test_it_satisfies_the_provider_protocol(self) -> None:
        """The seam is checkable, so a new provider cannot be half-written."""
        for provider in (
            feeds.StaticProvider(),
            feeds.YahooChartProvider(),
            feeds.ApiNinjasProvider(),
        ):
            assert isinstance(provider, feeds.QuoteProvider)
