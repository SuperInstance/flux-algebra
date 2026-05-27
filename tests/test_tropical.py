"""Tests for flux_algebra.tropical module."""

import pytest
from flux_algebra.tropical import (
    tropical_add,
    tropical_multiply,
    tropical_power,
    TROPICAL_INF,
    TropicalPolynomial,
    TropicalHarmony,
    TropicalVoiceLeading,
)


class TestTropicalArithmetic:
    def test_add_min(self):
        assert tropical_add(3.0, 5.0) == 3.0
        assert tropical_add(-1.0, 2.0) == -1.0

    def test_add_commutative(self):
        assert tropical_add(3.0, 5.0) == tropical_add(5.0, 3.0)

    def test_add_idempotent(self):
        assert tropical_add(3.0, 3.0) == 3.0

    def test_add_inf(self):
        assert tropical_add(3.0, TROPICAL_INF) == 3.0

    def test_multiply_sum(self):
        assert tropical_multiply(3.0, 5.0) == 8.0

    def test_multiply_zero(self):
        """Tropical 'zero' (multiplicative identity) is 0."""
        assert tropical_multiply(5.0, 0.0) == 5.0

    def test_multiply_commutative(self):
        assert tropical_multiply(3.0, 5.0) == tropical_multiply(5.0, 3.0)

    def test_power(self):
        """Tropical power: a^n = n*a."""
        assert tropical_power(3.0, 4) == 12.0

    def test_power_zero(self):
        assert tropical_power(5.0, 0) == 0.0

    def test_power_one(self):
        assert tropical_power(5.0, 1) == 5.0


class TestTropicalPolynomial:
    def test_evaluate_simple(self):
        """min(0, 4+x, 7+2x) at x=0 → min(0,4,7) = 0."""
        tp = TropicalPolynomial([0.0, 4.0, 7.0])
        assert tp.evaluate(0.0) == 0.0

    def test_evaluate_at_point(self):
        tp = TropicalPolynomial([0.0, 4.0, 7.0])
        # At x=3: min(0, 4+3, 7+6) = min(0, 7, 13) = 0
        assert tp.evaluate(3.0) == 0.0

    def test_degree(self):
        tp = TropicalPolynomial([1.0, 2.0, 3.0, 4.0])
        assert tp.degree == 3

    def test_callable(self):
        tp = TropicalPolynomial([0.0, 4.0])
        assert tp(0.0) == 0.0

    def test_roots(self):
        """Roots at a_i - a_{i+1}."""
        tp = TropicalPolynomial([4.0, 2.0])
        roots = tp.roots()
        assert len(roots) == 1
        assert roots[0] == pytest.approx(2.0)  # 4 - 2

    def test_inf_coefficient(self):
        """Infinity coefficients are skipped."""
        tp = TropicalPolynomial([0.0, TROPICAL_INF, 5.0])
        # At x=1: min(0, 5+2) = 0
        assert tp.evaluate(1.0) == 0.0

    def test_coefficients_copy(self):
        tp = TropicalPolynomial([1.0, 2.0])
        coeffs = tp.coefficients
        coeffs[0] = 999.0
        assert tp.coefficients[0] == 1.0  # original unchanged


class TestTropicalHarmony:
    def test_chord_cost_member(self):
        th = TropicalHarmony()
        cost = th.chord_cost([0, 4, 7])
        # C is in C major → distance 0
        assert cost(0) == 0.0
        assert cost(4) == 0.0
        assert cost(7) == 0.0

    def test_chord_cost_nonmember(self):
        th = TropicalHarmony()
        cost = th.chord_cost([0, 4, 7])
        # E♭ (3) is 1 semitone from E (4)
        assert cost(3) == 1.0
        # G♭ (6) is 1 semitone from G (7)
        assert cost(6) == 1.0

    def test_chord_cost_wraps(self):
        th = TropicalHarmony()
        cost = th.chord_cost([0])
        # B (11) is 1 semitone from C (0) via wrap
        assert cost(11) == 1.0

    def test_tropical_polynomial(self):
        th = TropicalHarmony()
        tp = th.chord_tropical_polynomial([0, 4, 7])
        assert tp.degree == 2

    def test_voice_leading(self):
        th = TropicalHarmony()
        vl = th.tropical_voice_leading([0, 4, 7], [5, 9, 0])
        assert len(vl) == 3
        sources = {p[0] for p in vl}
        targets = {p[1] for p in vl}
        assert sources == {0, 4, 7}
        assert targets == {5, 9, 0}

    def test_tropical_distance(self):
        th = TropicalHarmony()
        d = th.tropical_distance([0, 4, 7], [0, 4, 7])
        assert d == 0.0

    def test_harmonic_tension_diatonic(self):
        """Diatonic chord has 0 tension against its scale."""
        th = TropicalHarmony()
        c_major_scale = [0, 2, 4, 5, 7, 9, 11]
        tension = th.harmonic_tension([0, 4, 7], c_major_scale)
        assert tension == 0.0

    def test_harmonic_tension_chromatic(self):
        """Non-scale tones have positive tension."""
        th = TropicalHarmony()
        c_major_scale = [0, 2, 4, 5, 7, 9, 11]
        # C♯ major chord contains non-scale tones
        tension = th.harmonic_tension([1, 5, 8], c_major_scale)
        assert tension > 0.0


class TestTropicalVoiceLeading:
    def test_compute(self):
        tvl = TropicalVoiceLeading()
        result = tvl.compute([0, 4, 7], [5, 9, 0])
        assert result.distance >= 0
        assert len(result.pairs) == 3

    def test_compute_same(self):
        tvl = TropicalVoiceLeading()
        result = tvl.compute([0, 4, 7], [0, 4, 7])
        assert result.distance == 0.0

    def test_result_has_polynomials(self):
        tvl = TropicalVoiceLeading()
        result = tvl.compute([0, 4, 7], [5, 9, 0])
        assert result.source_polynomial is not None
        assert result.target_polynomial is not None

    def test_repr(self):
        tvl = TropicalVoiceLeading()
        assert "TropicalVoiceLeading" in repr(tvl)
        result = tvl.compute([0, 4, 7], [5, 9, 0])
        assert "→" in repr(result)

    def test_tropical_polynomial_repr(self):
        tp = TropicalPolynomial([0.0, 4.0, 7.0])
        r = repr(tp)
        assert "TropicalPolynomial" in r

    def test_tropical_harmony_repr(self):
        th = TropicalHarmony()
        assert "TropicalHarmony" in repr(th)

    def test_non12_modulus(self):
        th = TropicalHarmony(modulus=6)
        cost = th.chord_cost([0])
        assert cost(0) == 0.0
        assert cost(3) == 3.0  # 3 away from 0 in mod 6

    def test_voice_leading_empty(self):
        th = TropicalHarmony()
        assert th.tropical_voice_leading([], []) == []

    def test_voice_leading_mismatch(self):
        th = TropicalHarmony()
        with pytest.raises(ValueError):
            th.tropical_voice_leading([0], [0, 4])

    def test_tropical_polynomial_all_inf(self):
        tp = TropicalPolynomial([TROPICAL_INF, TROPICAL_INF])
        assert tp.evaluate(0.0) == TROPICAL_INF

    def test_tropical_vl_non12(self):
        tvl = TropicalVoiceLeading(modulus=6)
        result = tvl.compute([0, 2, 4], [1, 3, 5])
        assert result.distance >= 0
