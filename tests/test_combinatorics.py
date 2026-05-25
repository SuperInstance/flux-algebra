"""Tests for flux_algebra.combinatorics module."""

import pytest
from flux_algebra.combinatorics import (
    minimal_voice_leading,
    all_voice_leadings,
    smoothness,
    efficiency,
    voice_leading_distance,
    sort_by_smoothness,
)


class TestMinimalVoiceLeading:
    def test_C_to_F(self):
        """C major {0,4,7} → F major {5,9,0}: optimal assignment."""
        vl = minimal_voice_leading([0, 4, 7], [5, 9, 0])
        sources = {p[0] for p in vl}
        targets = {p[1] for p in vl}
        assert sources == {0, 4, 7}
        assert targets == {5, 9, 0}
        # Hungarian algorithm finds the minimal total distance
        s = smoothness(vl)
        assert s <= 5  # at most 5 (naive assignment)
        assert s == 3  # optimal: 0→0(0), 4→5(1), 7→9(2) = 3

    def test_same_chord(self):
        vl = minimal_voice_leading([0, 4, 7], [0, 4, 7])
        assert smoothness(vl) == 0

    def test_tritone(self):
        """C → G♭: {0,4,7} → {6,10,1}."""
        vl = minimal_voice_leading([0, 4, 7], [6, 10, 1])
        s = smoothness(vl)
        assert s <= 3 * 5  # at most 5 semitones per voice (wrapping)

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            minimal_voice_leading([0, 4], [5, 9, 0])

    def test_empty(self):
        assert minimal_voice_leading([], []) == []

    def test_single_voice(self):
        vl = minimal_voice_leading([0], [7])
        assert vl == [(0, 7)]
        assert smoothness(vl) == 5

    def test_wrapping_distance(self):
        """Voice leading should prefer wrapping distance when shorter."""
        # E (4) to F (5): distance 1, not 11
        vl = minimal_voice_leading([4], [5])
        assert smoothness(vl) == 1

    def test_two_voices(self):
        vl = minimal_voice_leading([0, 7], [5, 0])
        assert len(vl) == 2


class TestAllVoiceLeadings:
    def test_count(self):
        """3! = 6 voice leadings for 3-voice chords."""
        vls = all_voice_leadings([0, 4, 7], [5, 9, 0])
        assert len(vls) == 6

    def test_count_2voice(self):
        vls = all_voice_leadings([0, 4], [5, 9])
        assert len(vls) == 2

    def test_each_is_bijection(self):
        """Each VL should be a proper bijection."""
        vls = all_voice_leadings([0, 4, 7], [5, 9, 0])
        for vl in vls:
            sources = [p[0] for p in vl]
            targets = [p[1] for p in vl]
            assert sorted(sources) == [0, 4, 7]
            assert sorted(targets) == [0, 5, 9]

    def test_empty(self):
        assert all_voice_leadings([], []) == [[]]


class TestSmoothness:
    def test_zero_movement(self):
        assert smoothness([(0, 0), (4, 4), (7, 7)]) == 0

    def test_simple(self):
        # (0→0)=0, (4→5)=1, (7→9)=2 → total 3
        assert smoothness([(0, 0), (4, 5), (7, 9)]) == 3

    def test_wrapping(self):
        """B (11) to C (0): wrapping distance = 1."""
        assert smoothness([(11, 0)]) == 1

    def test_nonwrapping(self):
        """C (0) to F# (6): distance = 6 (not 6 the other way)."""
        assert smoothness([(0, 6)]) == 6


class TestEfficiency:
    def test_zero_movement(self):
        assert efficiency([(0, 0), (4, 4)]) == 0.0

    def test_all_moved(self):
        eff = efficiency([(0, 1), (4, 5), (7, 9)])
        # smoothness = 1+1+2 = 4, voices moved = 3 → 4/3
        assert abs(eff - 4/3) < 0.01

    def test_some_moved(self):
        eff = efficiency([(0, 0), (4, 5), (7, 9)])
        # smoothness = 0+1+2 = 3, voices moved = 2 → 1.5
        assert abs(eff - 1.5) < 0.01


class TestVoiceLeadingDistance:
    def test_same(self):
        assert voice_leading_distance([0, 4, 7], [0, 4, 7]) == 0

    def test_different(self):
        d = voice_leading_distance([0, 4, 7], [5, 9, 0])
        assert d == 3  # Hungarian optimal


class TestSortBySmoothness:
    def test_sort(self):
        targets = [[0, 4, 7], [5, 9, 0], [1, 5, 8]]
        result = sort_by_smoothness([0, 4, 7], targets)
        assert len(result) == 3
        # Same chord should be first (distance 0)
        assert result[0][0] == 0
        # Sorted ascending
        distances = [r[0] for r in result]
        assert distances == sorted(distances)
