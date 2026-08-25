"""Turning delivery records plus market history into elasticities the app can use.

The engine's ``scrcae.calibration`` does the estimation and owns every statistical
judgement. This module does the unglamorous part that decides whether the estimate
means anything: joining a company's delivery months to a market's price months, and
reporting what failed to join.

The join is where a calibration quietly goes wrong. A node's delivery record for
``2026-07`` has to meet the same market's ``2026-07`` price, and if a period label is
misread the pairing is off by a month and the estimator faithfully fits the wrong
relationship. So nothing is joined loosely: a record whose month has no market price
is dropped *by name* and counted, never matched to the nearest available month.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from scrcae.calibration import (
    CALIBRATION_VERSION,
    MIN_OBSERVATIONS,
    DeliveryObservation,
    ElasticityFit,
    calibrate_elasticity,
)

from app.adapters import CalibrationRecord, NodeExposure
from app.services.feeds import PriceHistory, QuoteProvider
from app.services.market import fetch_history

__all__ = [
    "CALIBRATION_VERSION",
    "NodeCalibration",
    "CalibrationRun",
    "calibrate_exposures",
    "records_from_run",
]


@dataclass(frozen=True, slots=True)
class NodeCalibration:
    """One exposure's calibration attempt, joined data and all."""

    node_id: str
    symbol: str
    anchor: float
    current_elasticity: float
    fit: ElasticityFit | None
    matched_periods: int
    unmatched_periods: tuple[str, ...] = field(default_factory=tuple)
    history_status: str = ""
    blocked: str = ""

    @property
    def can_adopt(self) -> bool:
        """Whether this fit is fit to replace the number in the exposure table."""
        return self.fit is not None and self.fit.is_usable

    @property
    def fitted_elasticity(self) -> float | None:
        return self.fit.elasticity if self.fit is not None else None

    def headline(self) -> str:
        if self.blocked:
            return self.blocked
        if self.fit is None:
            return "Not calibrated."
        text = self.fit.summary()
        if self.unmatched_periods:
            shown = ", ".join(self.unmatched_periods[:6])
            more = (
                f" and {len(self.unmatched_periods) - 6} more"
                if len(self.unmatched_periods) > 6
                else ""
            )
            text += (
                f" {len(self.unmatched_periods)} delivery period(s) had no market price "
                f"and were left out: {shown}{more}."
            )
        return text

    def delta_note(self) -> str:
        """How far the entered elasticity sits from the fitted one.

        Shown because the interesting outcome of a calibration is rarely the number
        itself — it is discovering that the figure someone typed a quarter ago is
        three times what the records support.
        """
        fitted = self.fitted_elasticity
        if fitted is None:
            return ""
        entered = self.current_elasticity
        if abs(fitted - entered) <= 1e-9:
            return "The entered elasticity already matches the fit."
        if entered == 0.0:
            return f"Entered 0, fitted {fitted:.2f}."
        if fitted == 0.0:
            return f"Entered {entered:.2f}, fitted 0."
        # The subject of the sentence is the entered value, so the multiplier has to be
        # entered-over-fitted and the direction has to agree with it. An earlier version
        # printed entered/fitted while labelling it with the direction of fitted/entered,
        # which read "0.30 is 0.25x above 1.21" — wrong word, and wrong in the direction
        # that flatters a stale assumption.
        ratio = entered / fitted
        direction = "above" if ratio > 1 else "below"
        return (
            f"Entered {entered:.2f}, fitted {fitted:.2f} \u2014 the entered value is "
            f"{ratio:.2f}\u00d7 the fit, {direction} it."
        )


@dataclass(frozen=True, slots=True)
class CalibrationRun:
    """Every exposure's attempt, plus what the run as a whole managed to do."""

    calibrations: tuple[NodeCalibration, ...] = field(default_factory=tuple)
    histories: Mapping[str, PriceHistory] = field(default_factory=dict)
    provider_name: str = ""

    @property
    def adoptable(self) -> tuple[NodeCalibration, ...]:
        return tuple(c for c in self.calibrations if c.can_adopt)

    @property
    def refused(self) -> tuple[NodeCalibration, ...]:
        return tuple(c for c in self.calibrations if not c.can_adopt)

    def summary(self) -> str:
        if not self.calibrations:
            return (
                "Nothing to calibrate: no market exposure has been configured, so "
                "there is no elasticity to estimate."
            )
        usable = len(self.adoptable)
        total = len(self.calibrations)
        if usable == 0:
            return (
                f"None of the {total} configured exposure(s) could be calibrated from "
                f"the delivery history supplied. Each reason is given below; the "
                f"elasticities remain asserted."
            )
        return (
            f"{usable} of {total} exposure(s) produced a usable elasticity. "
            f"The rest are listed with the reason they did not."
        )


def calibrate_exposures(
    exposures: Sequence[NodeExposure],
    records: Sequence[object],
    *,
    provider: QuoteProvider | None = None,
    histories: Mapping[str, PriceHistory] | None = None,
    months: int = 60,
    one_sided: bool = True,
) -> CalibrationRun:
    """Calibrate every configured exposure, or say why each one could not be.

    ``histories`` lets a caller supply price series directly, which is how the tests
    run without a network and how an installation on a spot-only feed can still
    calibrate from a series it exports itself. When absent, each distinct symbol is
    fetched once — once, not once per exposure, because two nodes exposed to the same
    market must be calibrated against the same series or their elasticities are not
    comparable.
    """
    if not exposures:
        return CalibrationRun(provider_name=getattr(provider, "name", "") or "")

    by_node: dict[str, list[object]] = {}
    for record in records:
        by_node.setdefault(getattr(record, "node_id", ""), []).append(record)

    resolved: dict[str, PriceHistory] = dict(histories or {})
    provider_name = getattr(provider, "name", "") or ""
    for exposure in exposures:
        if exposure.symbol in resolved:
            continue
        fetched = fetch_history(
            exposure.symbol, provider=provider, months=months
        )
        resolved[exposure.symbol] = fetched

    calibrations: list[NodeCalibration] = []
    for exposure in exposures:
        history = resolved.get(exposure.symbol)
        node_records = by_node.get(exposure.node_id, [])

        if history is None or not history.ok:
            calibrations.append(
                NodeCalibration(
                    node_id=exposure.node_id,
                    symbol=exposure.symbol,
                    anchor=exposure.anchor,
                    current_elasticity=exposure.elasticity,
                    fit=None,
                    matched_periods=0,
                    history_status=history.status if history else "No history",
                    blocked=(
                        f"No price history for {exposure.symbol}: "
                        f"{history.status if history else 'not fetched'}."
                    ),
                )
            )
            continue

        if not node_records:
            calibrations.append(
                NodeCalibration(
                    node_id=exposure.node_id,
                    symbol=exposure.symbol,
                    anchor=exposure.anchor,
                    current_elasticity=exposure.elasticity,
                    fit=None,
                    matched_periods=0,
                    history_status=history.status,
                    blocked=(
                        f"No delivery history recorded for {exposure.node_id}. "
                        f"At least {MIN_OBSERVATIONS} months are needed."
                    ),
                )
            )
            continue

        prices = history.by_period()
        observations: list[DeliveryObservation] = []
        unmatched: list[str] = []
        for record in sorted(node_records, key=lambda r: getattr(r, "period", "")):
            period = getattr(record, "period", "")
            price = prices.get(period)
            if price is None:
                unmatched.append(period)
                continue
            observations.append(
                DeliveryObservation(
                    period=period,
                    market_price=float(price),
                    # Percent in, percent out. The estimator is scale-free in this
                    # column, so no conversion is applied — converting would invite a
                    # future edit to convert twice.
                    disruption_rate=float(getattr(record, "disruption_rate_pct", 0.0)),
                )
            )

        fit = calibrate_elasticity(
            observations,
            anchor=exposure.anchor,
            one_sided=one_sided,
        )
        calibrations.append(
            NodeCalibration(
                node_id=exposure.node_id,
                symbol=exposure.symbol,
                anchor=exposure.anchor,
                current_elasticity=exposure.elasticity,
                fit=fit,
                matched_periods=len(observations),
                unmatched_periods=tuple(unmatched),
                history_status=history.status,
            )
        )

    return CalibrationRun(
        calibrations=tuple(calibrations),
        histories=resolved,
        provider_name=provider_name,
    )


def records_from_run(run: CalibrationRun) -> tuple[CalibrationRecord, ...]:
    """The storable evidence for the fits worth adopting.

    Only adoptable fits are recorded. A refused fit is not evidence of anything and
    storing it would leave a row in the saved portfolio that looks like a calibration
    to anyone reading the file.
    """
    stored: list[CalibrationRecord] = []
    for calibration in run.adoptable:
        fit = calibration.fit
        if fit is None or fit.elasticity is None:
            continue
        stored.append(
            CalibrationRecord(
                node_id=calibration.node_id,
                symbol=calibration.symbol,
                elasticity=float(fit.elasticity),
                anchor=float(calibration.anchor),
                interval_low=fit.interval_low,
                interval_high=fit.interval_high,
                r_squared=fit.r_squared,
                observations=fit.observations,
                periods_above_anchor=fit.periods_above_anchor,
                method=fit.method,
            )
        )
    return tuple(stored)
