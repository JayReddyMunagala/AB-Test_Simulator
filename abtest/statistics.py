from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    upper: float


@dataclass(frozen=True)
class ABTestResult:
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    p_value: float
    z_score: float
    confidence_interval: ConfidenceInterval
    is_significant: bool
    sample_size: int
    control_conversions: int
    treatment_conversions: int


@dataclass(frozen=True)
class PowerAnalysisResult:
    power: float
    recommended_sample_size: int
    detection_probability: float


def standard_normal_cdf(z: float) -> float:
    """Standard normal CDF via erf (stable and dependency-free)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def inverse_normal_cdf(p: float) -> float:
    """Inverse standard normal CDF.

    Ported from the TS implementation (Peter J. Acklam's approximation).
    Accurate enough for typical A/B testing use.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")

    # Coefficients
    a1 = -39.6968302866538
    a2 = 220.946098424521
    a3 = -275.928510446969
    a4 = 138.357751867269
    a5 = -30.6647980661472
    a6 = 2.50662827745924

    b1 = -54.4760987982241
    b2 = 161.585836858041
    b3 = -155.698979859887
    b4 = 66.8013118877197
    b5 = -13.2806815528857

    c1 = -0.00778489400243029
    c2 = -0.322396458041136
    c3 = -2.40075827716184
    c4 = -2.54973253934373
    c5 = 4.37466414146497
    c6 = 2.93816398269878

    d1 = 0.00778469570904146
    d2 = 0.32246712907004
    d3 = 2.445134137143
    d4 = 3.75440866190742

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        z = (
            (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
        )
        return z

    if p <= p_high:
        q = p - 0.5
        r = q * q
        z = (
            ((((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q)
            / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
        )
        return z

    # upper tail
    q = math.sqrt(-2 * math.log(1 - p))
    z = -(
        (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    )
    return z


def z_test_proportions(
    control_conversions: int,
    control_total: int,
    treatment_conversions: int,
    treatment_total: int,
    confidence_level: float = 0.95,
) -> ABTestResult:
    """Two-proportion z-test (pooled) + CI for difference (unpooled SE)."""
    if control_total <= 0 or treatment_total <= 0:
        raise ValueError("Totals must be > 0")
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be in (0, 1)")

    p1 = control_conversions / control_total
    p2 = treatment_conversions / treatment_total

    pooled_p = (control_conversions + treatment_conversions) / (control_total + treatment_total)
    se_pooled = math.sqrt(pooled_p * (1 - pooled_p) * (1 / control_total + 1 / treatment_total))

    z_score = 0.0 if se_pooled == 0 else (p2 - p1) / se_pooled
    p_value = 2 * (1 - standard_normal_cdf(abs(z_score)))

    alpha = 1 - confidence_level
    z_crit = inverse_normal_cdf(1 - alpha / 2)

    se_ci = math.sqrt((p1 * (1 - p1)) / control_total + (p2 * (1 - p2)) / treatment_total)
    diff = p2 - p1
    moe = z_crit * se_ci

    relative_lift = (diff / p1) * 100 if p1 != 0 else 0.0

    return ABTestResult(
        control_rate=p1,
        treatment_rate=p2,
        absolute_lift=diff,
        relative_lift=relative_lift,
        p_value=p_value,
        z_score=z_score,
        confidence_interval=ConfidenceInterval(lower=diff - moe, upper=diff + moe),
        is_significant=p_value < alpha,
        sample_size=control_total,
        control_conversions=control_conversions,
        treatment_conversions=treatment_conversions,
    )


def calculate_power(
    baseline_conversion: float,
    expected_uplift: float,
    sample_size: int,
    alpha: float = 0.05,
) -> float:
    """Approximate power for detecting expected_uplift at given sample_size per group."""
    p1 = baseline_conversion
    p2 = baseline_conversion + expected_uplift
    p2 = max(0.0, min(1.0, p2))

    pooled_p = (p1 + p2) / 2
    se = math.sqrt(2 * pooled_p * (1 - pooled_p) / sample_size)

    z_alpha = inverse_normal_cdf(1 - alpha / 2)
    effect = abs(p2 - p1)

    z_beta = (effect - z_alpha * se) / se if se != 0 else 0.0
    return standard_normal_cdf(z_beta)


def calculate_required_sample_size(
    baseline_conversion: float,
    expected_uplift: float,
    power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """Sample size per group needed to achieve desired power for expected uplift."""
    p1 = baseline_conversion
    p2 = baseline_conversion + expected_uplift
    p2 = max(0.0, min(1.0, p2))

    z_alpha = inverse_normal_cdf(1 - alpha / 2)
    z_beta = inverse_normal_cdf(power)

    pooled_p = (p1 + p2) / 2
    denom = (p2 - p1) ** 2
    if denom == 0:
        return 0

    n = math.ceil((2 * pooled_p * (1 - pooled_p) * (z_alpha + z_beta) ** 2) / denom)
    return int(n)


def simulate_power_analysis(
    baseline_conversion: float,
    expected_uplift: float,
    sample_size: int,
    simulations: int = 1000,
    alpha: float = 0.05,
) -> PowerAnalysisResult:
    """Monte Carlo estimate of power by simulating binomial draws and running the z-test."""
    significant = 0

    p_treat = max(0.0, min(1.0, baseline_conversion + expected_uplift))

    for _ in range(simulations):
        control_conv = _generate_binomial(sample_size, baseline_conversion)
        treat_conv = _generate_binomial(sample_size, p_treat)

        res = z_test_proportions(
            control_conv,
            sample_size,
            treat_conv,
            sample_size,
            confidence_level=1 - alpha,
        )

        if res.is_significant and res.absolute_lift > 0:
            significant += 1

    power_est = significant / simulations if simulations > 0 else 0.0
    recommended = calculate_required_sample_size(baseline_conversion, expected_uplift, power=0.8, alpha=alpha)

    return PowerAnalysisResult(
        power=power_est,
        recommended_sample_size=recommended,
        detection_probability=power_est * 100,
    )


def _generate_binomial(n: int, p: float) -> int:
    successes = 0
    for _ in range(n):
        if random.random() < p:
            successes += 1
    return successes


def generate_recommendation(result: ABTestResult) -> str:
    is_significant = result.is_significant
    relative_lift = result.relative_lift
    absolute_lift = result.absolute_lift
    p_value = result.p_value

    if not is_significant:
        return (
            "Do not ship variant B. The test did not reach statistical significance "
            f"(p = {p_value:.4f}). The observed difference could be due to random chance."
        )

    if relative_lift < 0:
        return (
            "Do not ship variant B. While statistically significant "
            f"(p = {p_value:.4f}), variant B performs worse than control "
            f"with a {abs(relative_lift):.2f}% decrease in conversion rate."
        )

    if relative_lift < 5:
        return (
            "Consider business context before shipping variant B. The test is "
            f"statistically significant (p = {p_value:.4f}), but the relative lift "
            f"of {relative_lift:.2f}% is small. Evaluate if the absolute lift of "
            f"{absolute_lift * 100:.2f} percentage points justifies implementation costs."
        )

    return (
        "Ship variant B. The test shows statistical significance "
        f"(p = {p_value:.4f}) with a meaningful relative lift of {relative_lift:.2f}% "
        f"(absolute lift: {absolute_lift * 100:.2f} percentage points)."
    )
