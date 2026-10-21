"""Exact rational sign certificate for the explicit parity counterexample."""

import csv
import json
from fractions import Fraction
from pathlib import Path

import sympy as sp


def main():
    x, p, s, c = sp.symbols("x p s c")
    a = x * (1 - x * x)
    b = (1 - x * x) ** 12
    ax = sp.diff(a, x)
    axx = sp.diff(a, x, 2)
    bx = sp.diff(b, x)
    bxx = sp.diff(b, x, 2)
    nu = sp.Rational(1, 100)
    linear = -s * bx - p * c * b - nu * bxx
    quadratic = a * bx + ax * b
    r0 = a + p * s * c - nu * p * p * s
    r1 = -s * ax - p * c * a - nu * axx
    r2 = a * ax
    integrand = sp.expand(
        b * b
        + b * linear
        + (linear * linear + 2 * b * quadratic) / 3
        + linear * quadratic / 2
        + quadratic * quadratic / 5
        + 2 * b * bx * (r0 / 3 + r1 / 4 + r2 / 5)
    )
    trig = sp.Poly(integrand, s, c)
    assert set(trig.monoms()) <= {(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)}
    maximum = max(sp.Poly(coefficient, x).degree() for coefficient in trig.coeffs())
    rules = {}
    for k in (1, 2):
        cosine = [sp.Integer(0)]
        sine = [sp.Integer(0)]
        for n in range(1, maximum + 1):
            cosine.append(sp.expand(-n * sine[n - 1] / (k * p)))
            sine.append(
                sp.expand(-(1 - (-1) ** n) * (-1) ** k / (k * p) + n * cosine[n - 1] / (k * p))
            )
        rules[k] = (cosine, sine)

    def spatial(coefficient, kind=0, k=1):
        value = 0
        for (n,), factor in sp.Poly(coefficient, x).terms():
            norm = sp.Rational(2, n + 1) if n % 2 == 0 else sp.Integer(0)
            integral = norm if kind == 0 else rules[k][0 if kind == 1 else 1][n]
            value += factor * integral
        return sp.expand(value)

    q = 0
    for powers, coefficient in trig.terms():
        if powers == (0, 0):
            value = spatial(coefficient)
        elif powers == (1, 0):
            value = spatial(coefficient, 2)
        elif powers == (0, 1):
            value = spatial(coefficient, 1)
        elif powers == (2, 0):
            value = (spatial(coefficient) - spatial(coefficient, 1, 2)) / 2
        elif powers == (0, 2):
            value = (spatial(coefficient) + spatial(coefficient, 1, 2)) / 2
        else:
            value = spatial(coefficient, 2, 2) / 2
        q += value
    q = sp.expand(q)
    fourth = spatial(sp.expand((b * bx) ** 2)) / 5
    assert fourth.is_Rational and fourth > 0
    terms = []
    for term in q.as_ordered_terms():
        coefficient, power = term.as_coeff_exponent(p)
        assert coefficient.is_Rational and power.is_Integer
        terms.append((int(power), Fraction(int(coefficient.p), int(coefficient.q))))

    def atan_bounds(denominator, count=100):
        total = sum(
            (Fraction((-1) ** n, (2 * n + 1) * denominator ** (2 * n + 1)) for n in range(count)),
            Fraction(),
        )
        error = Fraction(1, (2 * count + 1) * denominator ** (2 * count + 1))
        return total, total + error  # even count: alternating partial sum is a lower bound

    alo, ahi = atan_bounds(5)
    blo, bhi = atan_bounds(239)
    plo, phi = 16 * alo - 4 * bhi, 16 * ahi - 4 * blo
    assert plo > 0 and phi > plo
    qlo = qhi = Fraction()
    for exponent, coefficient in terms:
        lo, hi = (
            (plo**exponent, phi**exponent) if exponent >= 0 else (phi**exponent, plo**exponent)
        )
        if coefficient < 0:
            lo, hi = hi, lo
        qlo += coefficient * lo
        qhi += coefficient * hi
    assert qhi < 0
    tf = Fraction(int(fourth.p), int(fourth.q))
    epsilon = Fraction(2, 5)
    dlo, dhi = epsilon**2 * qlo + epsilon**4 * tf, epsilon**2 * qhi + epsilon**4 * tf
    assert dhi < 0 and dlo <= dhi
    scale = 10**12

    def outward(lo, hi):
        lower = (lo * scale).numerator // (lo * scale).denominator
        upper = -(-(hi * scale).numerator // (hi * scale).denominator)
        return dict(lower_numerator=lower, upper_numerator=upper, denominator=scale)

    base = Path(__file__).with_name("2026-10-21-unforced-burgers-parity-sign-certificate")
    assert not base.with_suffix(".json").exists() and not base.with_suffix(".csv").exists()
    with base.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["pi_exponent", "coefficient_numerator", "coefficient_denominator"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            dict(
                pi_exponent=n,
                coefficient_numerator=v.numerator,
                coefficient_denominator=v.denominator,
            )
            for n, v in terms
        )
    result = dict(
        method="Exact polynomial integration and rational outward bounds; Machin identity with alternating arctangent series",
        arctangent_terms=100,
        machin_denominators=[5, 239],
        q_bounds=outward(qlo, qhi),
        raw_minus_projected_squared_residual_bounds=outward(dlo, dhi),
        fourth_coefficient_numerator=tf.numerator,
        fourth_coefficient_denominator=tf.denominator,
        laurent_terms=len(terms),
        projection_worsens_proved=True,
        optimizer_steps=0,
        scope="Exact sign for this explicit analytic example only; not a certificate for trained PINNs, quadrature routines or registered quality.",
    )
    base.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
