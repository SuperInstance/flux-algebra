"""
Dial Geometry — Polyhedral geometry for tradition/dial space.

Analogous to Oscar.jl's PolyhedralGeometry module (convex hulls, polytopes).

Maps music traditions to points in [0,5]^3 "dial space" where axes represent
continuous parameters (e.g., dissonance tolerance, rhythmic complexity,
harmonic density). Each tradition occupies a convex region. Unexplored music
lies outside the union of tradition polytopes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import ConvexHull, Voronoi


@dataclass
class TraditionRegion:
    """A region in dial space associated with a musical tradition.

    Parameters
    ----------
    name : str
        Name of the tradition (e.g., "jazz", "blues", "classical").
    center : tuple of float
        Center point in [0,5]^3 dial space.
    radius : float
        Radius of the spherical approximation.
    description : str, optional
        Human-readable description.

    Examples
    --------
    >>> jazz = TraditionRegion("jazz", center=(3.5, 2.0, 4.0), radius=0.8)
    """

    name: str
    center: Tuple[float, float, float]
    radius: float = 1.0
    description: Optional[str] = None

    def contains(self, point: Tuple[float, ...]) -> bool:
        """Check if a point is within this tradition's region.

        Parameters
        ----------
        point : tuple of float
            Point in dial space.

        Returns
        -------
        bool
        """
        dist = np.linalg.norm(np.array(point) - np.array(self.center))
        return dist <= self.radius

    def distance(self, point: Tuple[float, ...]) -> float:
        """Euclidean distance from point to center.

        Parameters
        ----------
        point : tuple of float

        Returns
        -------
        float
        """
        return float(np.linalg.norm(np.array(point) - np.array(self.center)))

    def __repr__(self) -> str:
        return f"TraditionRegion('{self.name}', center={self.center}, r={self.radius})"


class DialPolytope:
    """Convex hull of tradition centers in [0,5]^3 dial space.

    The convex hull defines the "explored" region of musical space.
    Points inside the hull are connected to known traditions; points
    outside represent genuinely novel musical territory.

    Parameters
    ----------
    traditions : list of TraditionRegion
        Musical traditions defining the polytope.

    Examples
    --------
    >>> jazz = TraditionRegion("jazz", center=(3.5, 2.0, 4.0))
    >>> blues = TraditionRegion("blues", center=(4.0, 1.5, 3.0))
    >>> hull = DialPolytope(traditions=[jazz, blues])
    """

    def __init__(self, traditions: Optional[List[TraditionRegion]] = None) -> None:
        self._traditions = traditions or []
        self._hull: Optional[ConvexHull] = None
        if len(self._traditions) >= 4:
            points = np.array([t.center for t in self._traditions])
            try:
                self._hull = ConvexHull(points)
            except Exception:
                self._hull = None

    @property
    def traditions(self) -> List[TraditionRegion]:
        """Traditions defining this polytope."""
        return self._traditions

    @property
    def vertices(self) -> np.ndarray:
        """Vertices of the convex hull."""
        if self._hull is not None:
            return self._hull.points[self._hull.vertices]
        pts = np.array([t.center for t in self._traditions]) if self._traditions else np.empty((0, 3))
        return pts

    def add_tradition(self, tradition: TraditionRegion) -> None:
        """Add a tradition and recompute the convex hull.

        Parameters
        ----------
        tradition : TraditionRegion
        """
        self._traditions.append(tradition)
        if len(self._traditions) >= 4:
            points = np.array([t.center for t in self._traditions])
            try:
                self._hull = ConvexHull(points)
            except Exception:
                self._hull = None

    def contains(self, point: Tuple[float, ...]) -> bool:
        """Check if a point lies inside the convex hull.

        For fewer than 4 traditions, falls back to checking if the point
        is within any tradition's radius.

        Parameters
        ----------
        point : tuple of float
            Point in [0,5]^3 space.

        Returns
        -------
        bool
        """
        if self._hull is not None:
            # Use half-plane test: point is inside iff it's on the correct
            # side of all facet hyperplanes.
            p = np.array(point)
            for equation in self._hull.equations:
                normal = equation[:3]
                offset = equation[3]
                if np.dot(normal, p) + offset > 1e-10:
                    return False
            return True
        else:
            # Fallback: check distance to any tradition center
            return any(t.contains(point) for t in self._traditions)

    def volume(self) -> float:
        """Volume of the convex hull in [0,5]^3 space.

        Returns 0 if fewer than 4 points.

        Returns
        -------
        float
        """
        if self._hull is not None:
            return float(self._hull.volume)
        return 0.0

    def surface_area(self) -> float:
        """Surface area of the convex hull.

        Returns 0 if fewer than 4 points.

        Returns
        -------
        float
        """
        if self._hull is not None:
            return float(self._hull.area)
        return 0.0

    def nearest_tradition(self, point: Tuple[float, ...]) -> Optional[TraditionRegion]:
        """Find the nearest tradition to a point.

        Parameters
        ----------
        point : tuple of float

        Returns
        -------
        TraditionRegion or None
        """
        if not self._traditions:
            return None
        return min(self._traditions, key=lambda t: t.distance(point))

    def voronoi(self) -> Optional[Voronoi]:
        """Compute Voronoi tessellation of tradition centers.

        Requires at least 4 traditions in 3D space.

        Returns
        -------
        scipy.spatial.Voronoi or None
        """
        if len(self._traditions) < 4:
            return None
        points = np.array([t.center for t in self._traditions])
        try:
            return Voronoi(points)
        except Exception:
            return None

    def unexplored_regions(self, resolution: int = 10) -> np.ndarray:
        """Find grid points in [0,5]^3 outside all tradition regions.

        Parameters
        ----------
        resolution : int
            Grid points per axis.

        Returns
        -------
        np.ndarray
            Array of (x, y, z) points outside all traditions.
        """
        xs = np.linspace(0, 5, resolution)
        ys = np.linspace(0, 5, resolution)
        zs = np.linspace(0, 5, resolution)
        grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3)

        mask = np.ones(len(grid), dtype=bool)
        for t in self._traditions:
            dists = np.linalg.norm(grid - np.array(t.center), axis=1)
            mask &= dists > t.radius
        return grid[mask]

    def __repr__(self) -> str:
        return f"DialPolytope({len(self._traditions)} traditions)"


class VoiceLeadingGeodesic:
    """Shortest path between chord points in harmonic space.

    Uses voice-leading distance (sum of semitone movements) to find
    geodesics on the orbifold quotient of R^n by transposition/inversion.

    Parameters
    ----------
    modulus : int
        Number of pitch divisions (default 12).

    Examples
    --------
    >>> vlg = VoiceLeadingGeodesic(modulus=12)
    >>> path = vlg.geodesic([0, 4, 7], [5, 9, 0])
    >>> print(path.distance)
    5
    """

    def __init__(self, modulus: int = 12) -> None:
        self._modulus = modulus

    def geodesic(
        self, source: Sequence[int], target: Sequence[int]
    ) -> "GeodesicPath":
        """Find shortest voice-leading path between two chords.

        Uses the Hungarian algorithm to find the optimal assignment
        of source voices to target voices minimizing total distance.

        Parameters
        ----------
        source, target : sequence of int
            Pitch classes of source and target chords.

        Returns
        -------
        GeodesicPath
        """
        from scipy.optimize import linear_sum_assignment

        n = len(source)
        if n != len(target):
            raise ValueError("Source and target must have same number of voices")

        # Build cost matrix: cost[i][j] = minimal distance from source[i] to target[j]
        cost = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                diff = abs(source[i] - target[j])
                # Wrap-around distance
                cost[i][j] = min(diff, self._modulus - diff)

        row_ind, col_ind = linear_sum_assignment(cost)
        total_distance = int(cost[row_ind, col_ind].sum())

        # Build the path as pairs
        pairs = [(source[row_ind[i]], target[col_ind[i]]) for i in range(n)]

        return GeodesicPath(
            source=tuple(source),
            target=tuple(target),
            pairs=pairs,
            distance=total_distance,
            assignment=(tuple(row_ind), tuple(col_ind)),
        )

    def all_geodesics(
        self, source: Sequence[int], target: Sequence[int]
    ) -> List["GeodesicPath"]:
        """Find all minimal voice-leading paths.

        Parameters
        ----------
        source, target : sequence of int
            Pitch classes.

        Returns
        -------
        list of GeodesicPath
        """
        from flux_algebra.combinatorics import all_voice_leadings, smoothness

        all_vl = all_voice_leadings(list(source), list(target))
        if not all_vl:
            return []

        costs = [smoothness(vl) for vl in all_vl]
        min_cost = min(costs)

        return [
            GeodesicPath(
                source=tuple(source),
                target=tuple(target),
                pairs=vl,
                distance=min_cost,
                assignment=None,
            )
            for vl, c in zip(all_vl, costs)
            if c == min_cost
        ]

    def __repr__(self) -> str:
        return f"VoiceLeadingGeodesic(modulus={self._modulus})"


@dataclass(frozen=True)
class GeodesicPath:
    """A voice-leading geodesic between two chords.

    Parameters
    ----------
    source : tuple of int
        Source chord.
    target : tuple of int
        Target chord.
    pairs : list of (int, int)
        Voice-leading pairs.
    distance : int
        Total voice-leading distance.
    assignment : tuple or None
        Hungarian algorithm assignment indices.
    """

    source: Tuple[int, ...]
    target: Tuple[int, ...]
    pairs: list
    distance: int
    assignment: Optional[Tuple] = None

    def __repr__(self) -> str:
        return f"GeodesicPath({self.source} → {self.target}, dist={self.distance})"
