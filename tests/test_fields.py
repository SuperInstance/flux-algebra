"""Tests for flux_algebra.fields module."""

import pytest
from fractions import Fraction
from flux_algebra.fields import TuningField, AlgebraicTone


class TestTuningField:
    def test_et12(self):
        tf = TuningField.ET(12)
        assert tf.name == "ET-12"
        assert tf.degree == 12

    def test_et_step_cents(self):
        tf = TuningField.ET(12)
        assert abs(tf.step_cents() - 100.0) < 0.01

    def test_et24(self):
        tf = TuningField.ET(24)
        assert tf.degree == 24
        assert abs(tf.step_cents() - 50.0) < 0.01

    def test_meantone(self):
        tf = TuningField.meantone()
        assert "meantone" in tf.name
        assert tf.degree == 4

    def test_just(self):
        tf = TuningField.just()
        assert "just" in tf.name

    def test_pythagorean(self):
        tf = TuningField.pythagorean()
        assert tf.primes == (2, 3)

    def test_ratio_to_cents(self):
        tf = TuningField.ET(12)
        assert abs(tf.ratio_to_cents(2.0) - 1200.0) < 0.01

    def test_cents_to_ratio(self):
        tf = TuningField.ET(12)
        assert abs(tf.cents_to_ratio(1200.0) - 2.0) < 0.01

    def test_approximate_ratio(self):
        tf = TuningField.just()
        result = tf.approximate_ratio(1.5)
        assert isinstance(result, Fraction)

    def test_repr(self):
        assert "ET-12" in repr(TuningField.ET(12))


class TestAlgebraicTone:
    def test_from_fraction(self):
        tone = AlgebraicTone.from_fraction(3, 2, name="P5")
        assert tone.name == "P5"
        assert abs(tone.cents - 701.955) < 1.0

    def test_from_cents(self):
        tone = AlgebraicTone.from_cents(700.0, name="ET P5")
        assert abs(tone.cents - 700.0) < 0.01

    def test_compose(self):
        p5 = AlgebraicTone.from_fraction(3, 2)
        m3 = AlgebraicTone.from_fraction(6, 5)
        composed = p5.compose(m3)
        assert abs(composed.ratio - (3/2 * 6/5)) < 0.01

    def test_invert(self):
        p5 = AlgebraicTone.from_fraction(3, 2)
        inv = p5.invert()
        assert abs(inv.ratio - 2/3) < 0.01

    def test_stack(self):
        p5 = AlgebraicTone.from_fraction(3, 2)
        stacked = p5.stack(2)
        assert abs(stacked.ratio - 9/4) < 0.01

    def test_semitones(self):
        tone = AlgebraicTone.from_cents(700.0)
        assert abs(tone.semitones - 7.0) < 0.01

    def test_degree(self):
        tone = AlgebraicTone(ratio=1.5, minimal_polynomial=(1, 0, -2))
        assert tone.degree == 2

    def test_repr(self):
        tone = AlgebraicTone.from_fraction(3, 2, name="P5")
        assert "P5" in repr(tone)
