"""Independent exact verifier: polynomial differential algebra, no SymPy or quadrature."""

import csv
import json
from fractions import Fraction
from math import factorial
from pathlib import Path

# Powers of x, t, sin(pi*x), cos(pi*x), pi; coefficients are exact fractions.
ZERO = (0, 0, 0, 0, 0)


def constant(value):
    return {ZERO: Fraction(value)} if value else {}


def add(*polynomials):
    result = {}
    for polynomial in polynomials:
        for powers, value in polynomial.items():
            result[powers] = result.get(powers, Fraction()) + value
    return {k: v for k, v in result.items() if v}


def scale(polynomial, value):
    return {k: v * value for k, v in polynomial.items() if v * value}


def multiply(left, right):
    result = {}
    for a, va in left.items():
        for b, vb in right.items():
            key = tuple(x + y for x, y in zip(a, b))
            result[key] = result.get(key, Fraction()) + va * vb
    return {k: v for k, v in result.items() if v}


def power(polynomial, exponent):
    result = constant(1)
    for _ in range(exponent):
        result = multiply(result, polynomial)
    return result


def derivative(polynomial, axis):
    result = {}

    def insert(powers, value):
        key = tuple(powers)
        result[key] = result.get(key, Fraction()) + value

    for key, value in polynomial.items():
        a = list(key)
        if axis == "t":
            if a[1]:
                a[1] -= 1
                insert(a, value * key[1])
        else:
            if key[0]:
                a = list(key)
                a[0] -= 1
                insert(a, value * key[0])
            if key[2]:
                a = list(key)
                a[2] -= 1
                a[3] += 1
                a[4] += 1
                insert(a, value * key[2])
            if key[3]:
                a = list(key)
                a[3] -= 1
                a[2] += 1
                a[4] += 1
                insert(a, -value * key[3])
    return {k: v for k, v in result.items() if v}


def residual(v):
    return add(
        derivative(v, "t"),
        multiply(v, derivative(v, "x")),
        scale(derivative(derivative(v, "x"), "x"), Fraction(-1, 100)),
    )


def moment(n, kind, k=1):
    if kind == "constant":
        return {0: Fraction(2, n + 1)} if n % 2 == 0 else {}
    result = {}
    # Endpoint formula for integral x**n exp(i*k*pi*x); i**(-(j+1)) cycles.
    cycle = ((1, 0), (0, 1), (-1, 0), (0, -1))
    for j in range(n + 1):
        if (n - j) % 2 == 0:
            continue
        real, imaginary = cycle[(-(j + 1)) % 4]
        component = real if kind == "cosine" else imaginary
        if component:
            coefficient = Fraction(
                2 * (-1) ** (k + j) * factorial(n) * component, factorial(n - j) * k ** (j + 1)
            )
            result[-(j + 1)] = result.get(-(j + 1), Fraction()) + coefficient
    return result


def integrate(polynomial):
    result = {}
    for (n, time, sine, cosine, pi_power), value in polynomial.items():
        basis = {
            (0, 0): [(1, "constant", 1)],
            (1, 0): [(1, "sine", 1)],
            (0, 1): [(1, "cosine", 1)],
            (2, 0): [(Fraction(1, 2), "constant", 1), (Fraction(-1, 2), "cosine", 2)],
            (0, 2): [(Fraction(1, 2), "constant", 1), (Fraction(1, 2), "cosine", 2)],
            (1, 1): [(Fraction(1, 2), "sine", 2)],
        }[(sine, cosine)]
        for factor, kind, k in basis:
            for exponent, coefficient in moment(n, kind, k).items():
                key = exponent + pi_power
                result[key] = result.get(
                    key, Fraction()
                ) + value * factor * coefficient / Fraction(time + 1)
    return {k: v for k, v in result.items() if v}


def main():
    x = {(1, 0, 0, 0, 0): Fraction(1)}
    t = {(0, 1, 0, 0, 0): Fraction(1)}
    s = {(0, 0, 1, 0, 0): Fraction(1)}
    wall = add(constant(1), scale(power(x, 2), -1))
    z = add(scale(s, -1), multiply(t, multiply(x, wall)))
    a = multiply(t, power(wall, 12))
    ax = derivative(a, "x")
    linear = add(
        derivative(a, "t"),
        multiply(z, ax),
        multiply(derivative(z, "x"), a),
        scale(derivative(ax, "x"), Fraction(-1, 100)),
    )
    q = integrate(add(power(linear, 2), scale(multiply(residual(z), multiply(a, ax)), 2)))
    fourth = integrate(power(multiply(a, ax), 2))
    assert set(fourth) == {0}
    base = Path(__file__).with_name("2026-10-21-unforced-burgers-parity-sign-certificate")
    with base.with_suffix(".csv").open() as stream:
        declared = {
            int(r["pi_exponent"]): Fraction(
                int(r["coefficient_numerator"]), int(r["coefficient_denominator"])
            )
            for r in csv.DictReader(stream)
        }
    assert q == declared
    record = json.loads(base.with_suffix(".json").read_text())
    assert fourth[0] == Fraction(
        record["fourth_coefficient_numerator"], record["fourth_coefficient_denominator"]
    )
    count = record["arctangent_terms"]
    assert count == 100 and count % 2 == 0

    def arctangent(d):
        term = Fraction(1, d)
        total = Fraction()
        for n in range(count):
            total += term / Fraction(2 * n + 1)
            term /= -(d * d)
        return total, total + abs(term) / Fraction(2 * count + 1)

    alo, ahi = arctangent(5)
    blo, bhi = arctangent(239)
    low, high = 16 * alo - 4 * bhi, 16 * ahi - 4 * blo
    qlow = qhigh = Fraction()
    for exponent, coefficient in q.items():
        v = [low**exponent, high**exponent]
        endpoints = [coefficient * w for w in v]
        qlow += min(endpoints)
        qhigh += max(endpoints)
    delta_low, delta_high = (
        Fraction(4, 25) * qlow + Fraction(16, 625) * fourth[0],
        Fraction(4, 25) * qhigh + Fraction(16, 625) * fourth[0],
    )

    def enclosed(lo, hi, declared_bounds):
        denominator = declared_bounds["denominator"]
        return (
            Fraction(declared_bounds["lower_numerator"], denominator)
            <= lo
            <= hi
            <= Fraction(declared_bounds["upper_numerator"], denominator)
        )

    assert enclosed(qlow, qhigh, record["q_bounds"])
    assert enclosed(delta_low, delta_high, record["raw_minus_projected_squared_residual_bounds"])
    assert delta_high < 0
    result = dict(
        independent_polynomial_operator=True,
        exact_laurent_coefficients_matched=len(q),
        fourth_coefficient_matched=True,
        machin_bounds_recomputed=True,
        outward_bounds_verified=True,
        negative_difference_verified=True,
        uses_sympy=False,
        uses_quadrature=False,
        optimizer_steps=0,
        scope="Independent exact verification of the explicit counterexample only.",
    )
    target = Path(__file__).with_name(
        "2026-10-21-unforced-burgers-parity-certificate-verification.json"
    )
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
