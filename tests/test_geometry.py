"""Tests for flux_algebra.geometry module."""

import pytest
import numpy as np
from flux_algebra.geometry import (
    DialPolytope,
    TraditionRegion,
    VoiceLeadingGeodesic,
    GeodesicPath,
)


class TestTraditionRegion:
    def test_contains_center(self):
        r = TraditionRegion("jazz", center=(3.0, 2.0, 4.0), radius=1.0)
        assert r.contains((3.0, 2.0, 4.0))

    def test_contains_within_radius(self):
        r = TraditionRegion("blues", center=(3.0, 2.0, 4.0), radius=1.0)
        assert r.contains((3.5, 2.5, 4.5))

    def test_not_contains_outside(self):
        r = TraditionRegion("blues", center=(3.0, 2.0, 4.0), radius=0.1)
        assert not r.contains((0.0, 0.0, 0.0))

    def test_distance(self):
        r = TraditionRegion("test", center=(0.0, 0.0, 0.0), radius=1.0)
        assert abs(r.distance((3.0, 4.0, 0.0)) - 5.0) < 1e-10

    def test_repr(self):
        r = TraditionRegion("jazz", center=(1.0, 2.0, 3.0))
        assert "jazz" in repr(r)


class TestDialPolytope:
    def _make_traditions(self):
        """Create 4+ traditions for valid convex hull."""
        return [
            TraditionRegion("jazz", center=(3.5, 2.0, 4.0), radius=0.8),
            TraditionRegion("blues", center=(4.0, 1.5, 3.0), radius=0.7),
            TraditionRegion("classical", center=(1.5, 3.0, 2.0), radius=0.9),
            TraditionRegion("rock", center=(4.2, 2.5, 3.5), radius=0.6),
            TraditionRegion("folk", center=(2.0, 1.0, 1.5), radius=0.5),
        ]

    def test_create_with_traditions(self):
        traditions = self._make_traditions()
        dp = DialPolytope(traditions)
        assert len(dp.traditions) == 5

    def test_vertices(self):
        traditions = self._make_traditions()
        dp = DialPolytope(traditions)
        verts = dp.vertices
        assert verts.shape[1] == 3
        assert len(verts) >= 4

    def test_add_tradition(self):
        dp = DialPolytope([self._make_traditions()[0]])
        dp.add_tradition(self._make_traditions()[1])
        assert len(dp.traditions) == 2

    def test_contains_inside(self):
        traditions = self._make_traditions()
        dp = DialPolytope(traditions)
        # Center of first tradition should be inside hull
        assert dp.contains((3.0, 2.0, 3.0))

    def test_volume_positive(self):
        traditions = self._make_traditions()
        dp = DialPolytope(traditions)
        assert dp.volume() > 0

    def test_surface_area_positive(self):
        traditions = self._make_traditions()
        dp = DialPolytope(traditions)
        assert dp.surface_area() > 0

    def test_nearest_tradition(self):
        traditions = self._make_traditions()
        dp = DialPolytope(traditions)
        nearest = dp.nearest_tradition((4.0, 2.0, 3.5))
        assert nearest is not None
        # Should be rock or jazz (both near)
        assert nearest.name in ("jazz", "rock", "blues")

    def test_nearest_tradition_empty(self):
        dp = DialPolytope()
        assert dp.nearest_tradition((0, 0, 0)) is None

    def test_unexplored_regions(self):
        traditions = self._make_traditions()
        dp = DialPolytope(traditions)
        unexplored = dp.unexplored_regions(resolution=5)
        # Should find some unexplored points near edges of [0,5]^3
        assert len(unexplored) > 0

    def test_few_traditions_contains(self):
        """With <4 traditions, falls back to radius check."""
        t1 = TraditionRegion("t1", center=(2.5, 2.5, 2.5), radius=1.0)
        dp = DialPolytope([t1])
        assert dp.contains((2.5, 2.5, 2.5))
        assert not dp.contains((0.0, 0.0, 0.0))

    def test_voronoi(self):
        traditions = self._make_traditions()
        dp = DialPolytope(traditions)
        vor = dp.voronoi()
        assert vor is not None

    def test_voronoi_too_few(self):
        dp = DialPolytope([TraditionRegion("a", center=(1, 1, 1))])
        assert dp.voronoi() is None


class TestVoiceLeadingGeodesic:
    def test_geodesic_C_to_F(self):
        """C major {0,4,7} → F major {5,9,0}: minimal VL distance."""
        vlg = VoiceLeadingGeodesic(modulus=12)
        path = vlg.geodesic([0, 4, 7], [5, 9, 0])
        assert isinstance(path, GeodesicPath)
        assert path.distance == 3  # optimal via Hungarian

    def test_geodesic_same_chord(self):
        vlg = VoiceLeadingGeodesic()
        path = vlg.geodesic([0, 4, 7], [0, 4, 7])
        assert path.distance == 0

    def test_geodesic_length_mismatch(self):
        vlg = VoiceLeadingGeodesic()
        with pytest.raises(ValueError):
            vlg.geodesic([0, 4], [0, 4, 7])

    def test_all_geodesics(self):
        """There may be multiple minimal paths."""
        vlg = VoiceLeadingGeodesic()
        paths = vlg.all_geodesics([0, 4, 7], [0, 4, 7])
        assert len(paths) >= 1
        assert all(p.distance == 0 for p in paths)

    def test_geodesic_path_repr(self):
        vlg = VoiceLeadingGeodesic()
        path = vlg.geodesic([0, 4, 7], [5, 9, 0])
        r = repr(path)
        assert "→" in r
        assert "dist=" in r
