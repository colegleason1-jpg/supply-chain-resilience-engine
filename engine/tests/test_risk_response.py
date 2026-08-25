"""Tests for the risk response models.

The forensic finding under test: the acquired system computed
``(macro * risk_pts) ** 0.85`` and then multiplied by the funding scale ``x``.
Raising the *parameter* to a power and scaling *linearly* by funding leaves the
model exactly linear in funding. Funding a node at 50% delivers exactly half the
reduction of funding it fully. There are no diminishing returns to funding
anywhere in the acquired model, which is what the exponent was documented to
represent.

``ParameterPowerResponse`` reproduces that behaviour faithfully and is honest
about it. ``AllocationConcaveResponse`` applies the exponent to the funding
allocation, which is the model the documentation described.
"""

from __future__ import annotations

import math

import pytest

from scrcae.risk import (
    AllocationConcaveResponse,
    LinearResponse,
    ParameterPowerResponse,
)

SCALES = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)


# --------------------------------------------------------------------------- #
# The central finding
# --------------------------------------------------------------------------- #


def test_parameter_power_response_is_linear():
    """Half the funding gives exactly half the reduction. No curvature at all."""
    response = ParameterPowerResponse(exponent=0.85)
    full = response.evaluate(20.0, 1.0)
    for scale in SCALES:
        assert response.evaluate(20.0, scale) == pytest.approx(full * scale, rel=1e-12)
    assert response.is_linear is True


def test_parameter_power_response_only_reweights_between_nodes():
    """What the exponent actually does: compress large risk reducers relative to
    small ones. A real effect, but a cross-sectional reweighting, not a returns
    curve. Here a node with 4x the raw reduction delivers 4**0.85 = 3.28x."""
    response = ParameterPowerResponse(exponent=0.85)
    small = response.evaluate(5.0, 1.0)
    large = response.evaluate(20.0, 1.0)
    assert large / small == pytest.approx(math.pow(4.0, 0.85), rel=1e-12)


def test_allocation_concave_response_has_real_diminishing_returns():
    """Half the funding gives strictly more than half the reduction."""
    response = AllocationConcaveResponse(exponent=0.85)
    full = response.evaluate(20.0, 1.0)
    assert response.evaluate(20.0, 0.5) > 0.5 * full
    assert response.is_linear is False
    # At exponent 1 it degenerates to the linear model and says so, so the
    # optimizer can drop the linearisation machinery.
    assert AllocationConcaveResponse(exponent=1.0).is_linear is True


def test_allocation_concave_response_is_strictly_concave():
    """Midpoint value exceeds the chord: f((a+b)/2) > (f(a)+f(b))/2."""
    response = AllocationConcaveResponse(exponent=0.85)
    for a, b in ((0.1, 0.9), (0.2, 0.4), (0.3, 1.0), (0.05, 0.5)):
        mid = response.evaluate(15.0, (a + b) / 2)
        chord = (response.evaluate(15.0, a) + response.evaluate(15.0, b)) / 2
        assert mid > chord


def test_allocation_concave_marginal_returns_decrease():
    """Successive equal funding increments buy successively less reduction."""
    response = AllocationConcaveResponse(exponent=0.85)
    step = 0.1
    increments = [
        response.evaluate(15.0, x + step) - response.evaluate(15.0, x)
        for x in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    ]
    assert all(a > b for a, b in zip(increments, increments[1:]))
    assert all(inc > 0 for inc in increments)


def test_all_responses_agree_at_full_funding_only_when_exponents_align():
    """A useful sanity anchor for interpreting the two models against each other.

    At x = 1 the concave model returns the raw coefficient, because 1**alpha = 1.
    The parameter-power model returns coefficient**alpha. So the two models make
    identical predictions nowhere except where the coefficient is 1, and the
    divergence grows with the coefficient. Anyone reading historic results needs
    to know which model produced them.
    """
    concave = AllocationConcaveResponse(exponent=0.85)
    power = ParameterPowerResponse(exponent=0.85)
    assert concave.evaluate(20.0, 1.0) == pytest.approx(20.0)
    assert power.evaluate(20.0, 1.0) == pytest.approx(math.pow(20.0, 0.85))
    assert concave.evaluate(1.0, 1.0) == pytest.approx(power.evaluate(1.0, 1.0))


# --------------------------------------------------------------------------- #
# Shared behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "response",
    [LinearResponse(), ParameterPowerResponse(), AllocationConcaveResponse()],
)
def test_zero_funding_yields_zero_reduction(response):
    assert response.evaluate(20.0, 0.0) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "response",
    [LinearResponse(), ParameterPowerResponse(), AllocationConcaveResponse()],
)
def test_reduction_is_monotone_in_funding(response):
    values = [response.evaluate(12.0, s) for s in SCALES]
    assert all(a < b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize(
    "response",
    [LinearResponse(), ParameterPowerResponse(), AllocationConcaveResponse()],
)
def test_macro_multiplier_scales_exposure_upwards(response):
    calm = response.evaluate(12.0, 1.0, macro=1.0)
    stressed = response.evaluate(12.0, 1.0, macro=1.3)
    assert stressed > calm


@pytest.mark.parametrize(
    "response",
    [LinearResponse(), ParameterPowerResponse(), AllocationConcaveResponse()],
)
def test_negative_funding_is_rejected(response):
    with pytest.raises(ValueError):
        response.evaluate(12.0, -0.1)


def test_linear_response_is_the_identity_scaling():
    assert LinearResponse().evaluate(12.0, 0.4) == pytest.approx(4.8)
    assert LinearResponse().is_linear is True
    assert LinearResponse().linear_coefficient(12.0, 1.25) == pytest.approx(15.0)


def test_exponent_bounds_are_enforced():
    with pytest.raises(ValueError):
        AllocationConcaveResponse(exponent=0.0)
    with pytest.raises(ValueError):
        AllocationConcaveResponse(exponent=1.5)
    with pytest.raises(ValueError):
        ParameterPowerResponse(exponent=-0.2)


def test_uncalibrated_exponents_declare_themselves():
    """The exponent is an assumption inherited from the acquired system, not a
    measurement. It must say so in the audit record rather than looking like data.
    """
    for response in (AllocationConcaveResponse(), ParameterPowerResponse()):
        assert response.calibration_source == "uncalibrated-assumption"
        assert response.version


# --------------------------------------------------------------------------- #
# Tangent envelope
# --------------------------------------------------------------------------- #


def test_tangents_form_a_valid_outer_approximation():
    """Each tangent lies on or above the concave function everywhere, and the
    envelope touches it exactly at the tangent points."""
    response = AllocationConcaveResponse(exponent=0.85)
    coefficient = 18.0
    breakpoints = (0.1, 0.3, 0.5, 0.75, 1.0)
    tangents = response.tangents(coefficient, breakpoints=breakpoints)

    grid = [i / 200 for i in range(201)]
    for tangent in tangents:
        for x in grid:
            assert tangent.at(x) >= response.evaluate(coefficient, x) - 1e-9

    for point in breakpoints:
        envelope = min(t.at(point) for t in tangents)
        assert envelope == pytest.approx(response.evaluate(coefficient, point), abs=1e-9)


def test_tangent_intercepts_are_positive_which_is_why_the_ceiling_matters():
    """Every tangent to a strictly concave function through the origin has a
    positive intercept. Without a ``v_n <= ceiling * y_n`` constraint, an
    *unfunded* node could claim that intercept as free risk reduction. This test
    pins the reason that constraint exists so it is not removed as redundant.
    """
    response = AllocationConcaveResponse(exponent=0.85)
    tangents = response.tangents(20.0, breakpoints=(0.3, 0.6, 1.0))
    assert all(t.intercept > 0 for t in tangents)
    assert all(t.slope > 0 for t in tangents)
    # At x = 0 the envelope would hand out spurious value.
    assert min(t.at(0.0) for t in tangents) > 0.0
    assert response.evaluate(20.0, 0.0) == pytest.approx(0.0)


def test_tangent_slopes_decrease_along_the_curve():
    response = AllocationConcaveResponse(exponent=0.85)
    tangents = response.tangents(20.0, breakpoints=(0.1, 0.3, 0.6, 1.0))
    slopes = [t.slope for t in tangents]
    assert all(a > b for a, b in zip(slopes, slopes[1:]))


def test_envelope_tightens_as_breakpoints_are_added():
    response = AllocationConcaveResponse(exponent=0.85)
    coefficient = 20.0
    coarse = response.tangents(coefficient, breakpoints=(0.3, 1.0))
    fine = response.tangents(
        coefficient, breakpoints=tuple(0.3 + 0.07 * i for i in range(11))
    )
    x = 0.62
    truth = response.evaluate(coefficient, x)
    coarse_gap = min(t.at(x) for t in coarse) - truth
    fine_gap = min(t.at(x) for t in fine) - truth
    assert 0 <= fine_gap < coarse_gap


def test_linear_responses_expose_a_single_exact_tangent():
    """A linear response is exactly representable, so its "envelope" is the
    function itself: one tangent, zero intercept, no approximation error. The
    optimizer checks ``is_linear`` and skips auxiliary variables entirely."""
    for response in (LinearResponse(), ParameterPowerResponse()):
        tangents = response.tangents(20.0)
        assert len(tangents) == 1
        assert tangents[0].intercept == pytest.approx(0.0)
        assert tangents[0].slope == pytest.approx(
            response.linear_coefficient(20.0)
        )
        assert response.is_linear
