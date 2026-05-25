"""Tests for flux_algebra.oscar_compat module — Oscar.jl-compatible API surface."""

import pytest
from flux_algebra.oscar_compat import (
    polynomial_ring,
    integer_ring,
    ideal,
    gens,
    number_field,
    dihedral_group,
    permutation_group,
    order,
    elements,
    convex_hull,
    voice_leading,
    tropical_semiring,
    free_module,
    major_triad,
    minor_triad,
)
from flux_algebra.rings import HarmonicRing, ChordIdeal
from flux_algebra.fields import TuningField
from flux_algebra.groups import TranspositionInversionGroup, PLRGroup, Triad
from flux_algebra.geometry import DialPolytope
from flux_algebra.tropical import TropicalHarmony
from flux_algebra.modules import VoiceModule


class TestRingConstructors:
    def test_polynomial_ring(self):
        R = polynomial_ring("ZZ", 12)
        assert isinstance(R, HarmonicRing)
        assert R.modulus == 12

    def test_integer_ring(self):
        R = integer_ring(12)
        assert isinstance(R, HarmonicRing)

    def test_ideal(self):
        R = integer_ring(12)
        I = ideal(R, [0, 4, 7])
        assert isinstance(I, ChordIdeal)

    def test_gens(self):
        R = integer_ring(12)
        assert gens(R) == [1]


class TestNumberFieldConstructors:
    def test_et12(self):
        K = number_field("ET-12")
        assert isinstance(K, TuningField)
        assert K.name == "ET-12"

    def test_et24(self):
        K = number_field("ET-24")
        assert K.degree == 24

    def test_meantone(self):
        K = number_field("meantone")
        assert "meantone" in K.name

    def test_just(self):
        K = number_field("just")
        assert "just" in K.name

    def test_pythagorean(self):
        K = number_field("pythagorean")
        assert K.primes == (2, 3)


class TestGroupConstructors:
    def test_dihedral_group(self):
        G = dihedral_group(12)
        assert isinstance(G, TranspositionInversionGroup)
        assert G.order == 24

    def test_permutation_group(self):
        G = permutation_group(12)
        assert isinstance(G, PLRGroup)

    def test_order(self):
        G = dihedral_group(12)
        assert order(G) == 24

    def test_plr_order(self):
        G = permutation_group(12)
        assert order(G) == 24

    def test_elements(self):
        G = dihedral_group(12)
        elems = elements(G)
        assert len(elems) == 24

    def test_elements_wrong_type(self):
        with pytest.raises(TypeError):
            elements(PLRGroup())


class TestGeometryConstructors:
    def test_convex_hull(self):
        points = [(1, 2, 3), (4, 5, 6), (0, 1, 2), (3, 2, 1)]
        dp = convex_hull(points)
        assert isinstance(dp, DialPolytope)

    def test_convex_hull_with_names(self):
        points = [(1, 2, 3), (4, 5, 6), (0, 1, 2), (3, 2, 1)]
        dp = convex_hull(points, names=["a", "b", "c", "d"])
        names = [t.name for t in dp.traditions]
        assert names == ["a", "b", "c", "d"]


class TestVoiceLeadingCompat:
    def test_voice_leading(self):
        vl = voice_leading([0, 4, 7], [5, 9, 0])
        assert len(vl) == 3


class TestTropicalCompat:
    def test_tropical_semiring(self):
        th = tropical_semiring()
        assert isinstance(th, TropicalHarmony)


class TestModuleConstructors:
    def test_free_module(self):
        vm = free_module(rank=4, modulus=12)
        assert isinstance(vm, VoiceModule)
        assert vm.rank == 4


class TestTriadConstructors:
    def test_major_triad(self):
        t = major_triad(0)
        assert isinstance(t, Triad)
        assert t.quality == "major"
        assert t.pitch_classes == (0, 4, 7)

    def test_minor_triad(self):
        t = minor_triad(0)
        assert isinstance(t, Triad)
        assert t.quality == "minor"
        assert t.pitch_classes == (0, 3, 7)

    def test_triad_various_roots(self):
        for root in range(12):
            maj = major_triad(root)
            assert maj.is_major
            m = minor_triad(root)
            assert m.is_minor
