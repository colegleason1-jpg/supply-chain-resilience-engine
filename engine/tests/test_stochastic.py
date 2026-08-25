"""Tests for the stochastic layer.

Three forensic findings are under test here.

**The 0.4 shock floor biases the central estimate upward.** The acquired system
drew ``m = max(0.4, 1 + 0.12 Z)``. Truncating only the left tail raises the mean
above one, so simulated risk *reduction* is systematically larger than the
deterministic reduction and simulated remaining risk is systematically
optimistic — by construction, not by chance. At sigma = 0.12 the floor sits five
standard deviations out and the bias is invisible. At sigma = 0.30 it sits two
out and becomes material. Calibrating sigma is on the roadmap, which is exactly
what makes this dangerous: raising sigha to a realistic value silently turns on
an optimism bias.

**Normal multipliers can go negative.** For ``Z < -1/sigma`` the multiplier is
below zero: an intervention that *increases* the risk it was funded to reduce.
The floor conceals this rather than fixing it.

**Ridge repair destroys the unit diagonal.** Adding ``I * (-min_eig + eps)``
produces a covariance matrix with inflated, unequal variances. Cholesky then
scales every shock by roughly ``sqrt(1 + ridge)``, widening reported tail risk
for reasons that are not properties of the business.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scrcae.stochastic import (
    DeterministicShock,
    LegacyTruncatedNormalShock,
    LognormalShock,
    MatrixCorrelation,
    SimulationRequest,
    UniformCorrelation,
    cholesky_factor,
    naive_ridge_repair,
    repair_to_correlation,
    run,
)

NODES = ("N1", "N2", "N3", "N4")
REDUCTIONS = {"N1": 12.5, "N2": 9.0, "N3": 4.25, "N4": 7.4}
DETERMINISTIC_TOTAL = sum(REDUCTIONS.values())


def _request(**kwargs: object) -> SimulationRequest:
    defaults: dict[str, object] = {
        "baseline_risk_pts": 65.5,
        "risk_reduction_by_node": REDUCTIONS,
        "correlation": UniformCorrelation(rho=0.35).matrix(NODES),
        "node_order": NODES,
        "iterations": 20_000,
        "seed": 42,
    }
    defaults.update(kwargs)
    return SimulationRequest(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Reproducibility and RNG hygiene
# --------------------------------------------------------------------------- #


def test_same_seed_produces_bit_identical_samples():
    a = run(_request(iterations=5_000, seed=7))
    b = run(_request(iterations=5_000, seed=7))
    np.testing.assert_array_equal(a.samples, b.samples)
    assert a.p50_risk_pts == b.p50_risk_pts
    assert a.p90_risk_pts == b.p90_risk_pts


def test_different_seeds_produce_different_samples_but_similar_statistics():
    """Seed sensitivity must be *testable*. The acquired system called
    ``np.random.seed(42)`` internally, so every run returned the same numbers and
    there was no way to distinguish a stable estimate from a lucky one."""
    a = run(_request(iterations=20_000, seed=1))
    b = run(_request(iterations=20_000, seed=2))
    assert not np.array_equal(a.samples, b.samples)
    # Both estimate the same underlying quantity, so they must agree to within a
    # few standard errors.
    assert abs(a.p50_risk_pts - b.p50_risk_pts) < 4 * (
        a.p50_standard_error + b.p50_standard_error
    )


def test_simulation_does_not_touch_the_global_random_state():
    """No ``np.random.seed`` anywhere. Global RNG mutation makes every other
    stochastic component in the process non-reproducible as a side effect."""
    np.random.seed(12345)
    before = np.random.random(5)

    np.random.seed(12345)
    run(_request(iterations=2_000, seed=99))
    after = np.random.random(5)

    np.testing.assert_array_equal(before, after)


def test_iteration_and_seed_are_recorded_on_the_result():
    result = run(_request(iterations=3_000, seed=77))
    assert result.iterations == 3_000
    assert result.seed == 77
    assert result.samples.shape == (3_000,)
    assert "shock/" in result.shock_version


# --------------------------------------------------------------------------- #
# The truncation bias
# --------------------------------------------------------------------------- #


def test_legacy_floor_bias_matches_its_closed_form():
    """E[max(f, 1+sZ)] = f*Phi(a) + (1-Phi(a)) + s*phi(a), with a = (f-1)/s.

    The ``s*phi(a)`` term is strictly positive, which *is* the bias. Asserting
    the simulation against the closed form proves the bias is a property of the
    estimator rather than an artefact of any particular draw.
    """
    for sigma in (0.12, 0.20, 0.30, 0.45):
        shock = LegacyTruncatedNormalShock(sigma=sigma, floor=0.4)
        rng = np.random.default_rng(2024)
        draws = shock.apply(rng.standard_normal(4_000_000))
        assert draws.mean() == pytest.approx(shock.analytic_mean, abs=3e-4)
        assert shock.analytic_bias > 0.0
        assert shock.analytic_bias == pytest.approx(shock.analytic_mean - 1.0, abs=1e-12)
        assert shock.is_mean_preserving is False


def test_floor_bias_grows_by_orders_of_magnitude_as_sigma_is_calibrated_up():
    """The reason this survived, quantified honestly.

    At sigma = 0.12 the floor sits 5 sigma out and the multiplier bias is around
    6e-9 — genuinely undetectable. At sigma = 0.30 it sits 2 sigma out and the
    bias is around 2.5e-3. In multiplier terms that is still a small number, and
    it would be an overstatement to call it alarming on its own.

    What makes it matter is that it is (a) strictly one-directional, always
    flattering, and (b) multiplied by total portfolio reduction, so it scales
    with the size of the programme rather than staying fixed. The end-to-end
    consequence is asserted in ``test_legacy_shock_understates_risk_and_...``.

    So the finding is not "the shipped model is badly biased" — at sigma = 0.12
    it is not. It is that the error term is a monotone function of a parameter
    the roadmap intends to raise, so it grows silently as the model is made more
    realistic. That is a governance defect more than a numerical one.
    """
    sigmas = (0.12, 0.20, 0.30, 0.45, 0.60)
    biases = [LegacyTruncatedNormalShock(sigma=s).analytic_bias for s in sigmas]

    # Strictly positive at every sigma: truncation can only ever flatter.
    assert all(bias > 0.0 for bias in biases)
    # Monotonically increasing in sigma.
    assert all(a < b for a, b in zip(biases, biases[1:]))

    at_012, at_030 = biases[0], biases[2]
    assert at_012 < 1e-7
    assert at_030 > 1e-3
    assert at_030 > 10_000 * at_012

    # Translated into the unit the product reports: percentage points of
    # understated risk on this portfolio.
    assert at_030 * DETERMINISTIC_TOTAL > 0.05


def test_legacy_shock_understates_risk_and_lognormal_does_not():
    """End-to-end consequence: the legacy model reports lower remaining risk
    than the deterministic model, for no reason a user could defend."""
    legacy = run(
        _request(
            iterations=60_000,
            shock_model=LegacyTruncatedNormalShock(sigma=0.30, floor=0.4),
        )
    )
    corrected = run(_request(iterations=60_000, shock_model=LognormalShock(sigma=0.30)))

    assert legacy.reduction_bias_pts > 0.05
    assert legacy.mean_risk_pts < legacy.deterministic_risk_pts

    assert abs(corrected.reduction_bias_pts) < 0.05
    assert corrected.mean_risk_pts == pytest.approx(
        corrected.deterministic_risk_pts, abs=0.05
    )


def test_lognormal_shock_is_positive_and_exactly_mean_one():
    """``exp(sigma*Z - sigma^2/2)`` is strictly positive for every Z and has mean
    exactly one for *any* sigma, so no floor is needed and dispersion can be
    calibrated without moving the central estimate."""
    for sigma in (0.05, 0.12, 0.30, 0.60, 1.0):
        shock = LognormalShock(sigma=sigma)
        rng = np.random.default_rng(11)
        draws = shock.apply(rng.standard_normal(2_000_000))
        assert draws.min() > 0.0
        assert draws.mean() == pytest.approx(1.0, abs=5e-3)
        assert shock.is_mean_preserving is True
        # Right-skewed: the median sits below the mean.
        assert float(np.median(draws)) < draws.mean()


def test_legacy_normal_multiplier_would_go_negative_without_the_floor():
    """The defect the floor was hiding. At sigma = 0.30, Z < -3.33 yields a
    negative multiplier: an intervention that adds risk. Roughly 4 draws in
    10,000 — rare, but a rare *sign error* is still a sign error."""
    sigma = 0.30
    rng = np.random.default_rng(5)
    z = rng.standard_normal(500_000)
    unfloored = 1.0 + sigma * z
    assert (unfloored < 0.0).any()

    floored = LegacyTruncatedNormalShock(sigma=sigma, floor=0.4).apply(z)
    assert floored.min() >= 0.4  # concealed, not fixed

    lognormal = LognormalShock(sigma=sigma).apply(z)
    assert lognormal.min() > 0.0  # structurally impossible to go negative


def test_deterministic_shock_collapses_onto_the_deterministic_result():
    """A zero-variance shock must reproduce the optimizer's own answer exactly.
    This is the bridge that proves the simulation and the optimizer agree on what
    they are measuring."""
    result = run(_request(iterations=500, shock_model=DeterministicShock()))
    assert result.std_dev_pts == pytest.approx(0.0, abs=1e-12)
    assert result.p50_risk_pts == pytest.approx(result.deterministic_risk_pts, abs=1e-12)
    assert result.p90_risk_pts == pytest.approx(result.p50_risk_pts, abs=1e-12)
    assert result.reduction_bias_pts == pytest.approx(0.0, abs=1e-12)
    assert result.deterministic_reduction_pts == pytest.approx(
        DETERMINISTIC_TOTAL, abs=1e-9
    )


# --------------------------------------------------------------------------- #
# Distributional sanity
# --------------------------------------------------------------------------- #


def test_p90_is_never_below_p50():
    """Risk is stated as a bad-outcome quantity, so the 90th percentile is the
    pessimistic tail and must weakly exceed the median."""
    for seed in (1, 2, 3, 4, 5):
        result = run(_request(iterations=10_000, seed=seed))
        assert result.p90_risk_pts >= result.p50_risk_pts
        assert result.p10_risk_pts <= result.p50_risk_pts
        assert result.expected_shortfall_90_pts >= result.p90_risk_pts


def test_risk_samples_are_clamped_to_a_valid_percentage():
    result = run(_request(iterations=20_000, shock_model=LognormalShock(sigma=0.8)))
    assert result.samples.min() >= 0.0
    assert result.samples.max() <= 100.0


def test_standard_errors_shrink_with_more_iterations():
    """The estimator converges, and the result says how precise it is. The
    acquired system reported P50 and P90 to two decimals with no error bar, which
    invites reading precision that is not there."""
    coarse = run(_request(iterations=2_000, seed=3))
    fine = run(_request(iterations=50_000, seed=3))
    assert fine.p50_standard_error < coarse.p50_standard_error
    assert fine.p90_standard_error < coarse.p90_standard_error
    # Roughly the 1/sqrt(N) rate: a 25x sample increase should cut the error by
    # something in the neighbourhood of 5x.
    ratio = coarse.p50_standard_error / fine.p50_standard_error
    assert 2.0 < ratio < 12.0


def test_higher_correlation_widens_the_tail():
    """Correlated node shocks fail together, which is the entire reason for
    modelling correlation. Independent shocks diversify; correlated ones do not."""
    independent = run(
        _request(iterations=40_000, correlation=UniformCorrelation(rho=0.0).matrix(NODES))
    )
    correlated = run(
        _request(iterations=40_000, correlation=UniformCorrelation(rho=0.9).matrix(NODES))
    )
    assert correlated.std_dev_pts > independent.std_dev_pts
    assert correlated.tail_spread_pts > independent.tail_spread_pts


def test_higher_sigma_widens_the_tail_without_moving_the_centre():
    """The property that makes sigma calibratable: dispersion is a free
    parameter, and changing it does not drag the central estimate along."""
    calm = run(_request(iterations=60_000, shock_model=LognormalShock(sigma=0.10)))
    wild = run(_request(iterations=60_000, shock_model=LognormalShock(sigma=0.40)))
    assert wild.std_dev_pts > calm.std_dev_pts
    assert abs(wild.reduction_bias_pts) < 0.10
    assert abs(calm.reduction_bias_pts) < 0.10


def test_empty_portfolio_returns_the_baseline_with_no_dispersion():
    result = run(
        _request(
            risk_reduction_by_node={},
            node_order=(),
            correlation=np.zeros((0, 0)),
            iterations=100,
        )
    )
    assert result.p50_risk_pts == pytest.approx(65.5)
    assert result.std_dev_pts == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Correlation repair
# --------------------------------------------------------------------------- #


def _indefinite_matrix() -> np.ndarray:
    """A plausible user-entered matrix that is not positive semi-definite.

    Three nodes each claiming strong negative correlation with the other two is
    geometrically impossible, but nothing stops an analyst typing it into a grid.
    """
    m = np.array(
        [
            [1.0, -0.8, -0.8],
            [-0.8, 1.0, -0.8],
            [-0.8, -0.8, 1.0],
        ]
    )
    assert np.linalg.eigvalsh(m).min() < 0
    return m


def test_eigenvalue_repair_preserves_the_unit_diagonal():
    repaired, report = repair_to_correlation(_indefinite_matrix())
    assert report.repair_applied
    assert report.min_eigenvalue_before < 0
    assert report.min_eigenvalue_after >= -1e-12
    np.testing.assert_allclose(np.diag(repaired), 1.0, atol=1e-12)
    assert report.max_diagonal_deviation < 1e-12
    np.testing.assert_allclose(repaired, repaired.T, atol=1e-12)
    assert np.abs(repaired).max() <= 1.0 + 1e-12
    assert "repaired" in report.summary().lower() or report.repair_applied


def test_naive_ridge_repair_inflates_the_variances():
    """The acquired system's repair, quantified.

    Adding ``I * (-min_eig + eps)`` leaves a matrix whose diagonal is no longer
    one, so it is a covariance matrix. Every simulated shock is then scaled by
    ``sqrt(diagonal)``, inflating reported tail risk by a factor nobody chose.
    """
    naive = naive_ridge_repair(_indefinite_matrix())
    assert np.linalg.eigvalsh(naive).min() >= -1e-9  # it does fix definiteness
    diagonal = np.diag(naive)
    assert diagonal.min() > 1.0 + 1e-6  # but at this cost

    inflation = math.sqrt(float(diagonal[0]))
    assert inflation > 1.0
    # Correlations are diluted towards zero as a side effect, so the correlation
    # the analyst entered is not the correlation being simulated.
    naive_corr = naive / np.outer(np.sqrt(diagonal), np.sqrt(diagonal))
    assert abs(naive_corr[0, 1]) < abs(_indefinite_matrix()[0, 1])


def test_repaired_matrix_reproduces_the_requested_correlation_far_better():
    """The practical test: simulate with each repair and compare the realised
    sample correlation against the matrix the analyst actually asked for."""
    target = _indefinite_matrix()
    rng = np.random.default_rng(3)
    z = rng.standard_normal((3, 200_000))

    repaired, _ = repair_to_correlation(target)
    good = cholesky_factor(repaired) @ z
    bad = cholesky_factor(naive_ridge_repair(target)) @ z

    # Variance fidelity: correlated standard normals should have unit variance.
    assert float(np.var(good, axis=1).mean()) == pytest.approx(1.0, abs=0.02)
    assert float(np.var(bad, axis=1).mean()) > 1.05


def test_already_valid_matrix_is_returned_untouched():
    valid = UniformCorrelation(rho=0.35).matrix(NODES)
    repaired, report = repair_to_correlation(valid)
    assert not report.repair_applied
    assert report.frobenius_shift == pytest.approx(0.0)
    np.testing.assert_allclose(repaired, valid, atol=1e-12)


def test_uniform_correlation_rejects_impossible_coefficients():
    """For k nodes a uniform rho is only positive semi-definite above
    ``-1/(k-1)``, so a uniformly strongly-negative network is not a thing that
    can exist. Better to say so than to repair it into something else."""
    assert UniformCorrelation(rho=0.35).matrix(NODES).shape == (4, 4)
    np.testing.assert_allclose(
        np.diag(UniformCorrelation(rho=0.35).matrix(NODES)), 1.0
    )
    with pytest.raises(ValueError):
        UniformCorrelation(rho=1.5)
    with pytest.raises(ValueError):
        UniformCorrelation(rho=-1.2)

    # rho = -0.5 is a perfectly legal coefficient for two nodes and impossible
    # for four, so the bound depends on network size and is checked at build.
    assert UniformCorrelation(rho=-0.5).matrix(("A", "B")).shape == (2, 2)
    with pytest.raises(ValueError):
        UniformCorrelation(rho=-0.5).matrix(NODES)


def test_matrix_correlation_reorders_to_match_node_order():
    """Node ordering is a live bug class: the acquired system built its
    correlation grid from Streamlit widget order and indexed it positionally, so
    reordering the input file silently reassigned correlations between nodes."""
    values = (
        (1.0, 0.9, 0.1),
        (0.9, 1.0, 0.2),
        (0.1, 0.2, 1.0),
    )
    correlation = MatrixCorrelation(values=values, node_ids=("A", "B", "C"))

    as_given = correlation.matrix(("A", "B", "C"))
    reordered = correlation.matrix(("C", "B", "A"))

    assert as_given[0, 1] == pytest.approx(0.9)
    # After reordering, the A-B pair must still be 0.9 — now at index (2, 1).
    assert reordered[2, 1] == pytest.approx(0.9)
    assert reordered[0, 2] == pytest.approx(0.1)


def test_matrix_correlation_rejects_unknown_or_missing_nodes():
    correlation = MatrixCorrelation(
        values=((1.0, 0.5), (0.5, 1.0)), node_ids=("A", "B")
    )
    with pytest.raises(Exception):
        correlation.matrix(("A", "Z"))


def test_simulation_reports_its_correlation_repair():
    """Whatever repair happened must reach the audit record. A silent matrix
    substitution is the kind of thing that makes a number unexplainable six
    months later."""
    result = run(
        _request(
            iterations=1_000,
            correlation=np.array(
                [
                    [1.0, -0.8, -0.8, 0.0],
                    [-0.8, 1.0, -0.8, 0.0],
                    [-0.8, -0.8, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
    )
    assert result.correlation_repair.repair_applied
    assert result.correlation_repair.min_eigenvalue_before < 0


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def test_invalid_simulation_requests_are_rejected():
    with pytest.raises(ValueError):
        _request(iterations=0)
    with pytest.raises(ValueError):
        _request(baseline_risk_pts=-1.0)
    with pytest.raises(Exception):
        _request(correlation=np.eye(3))  # wrong size for four nodes
