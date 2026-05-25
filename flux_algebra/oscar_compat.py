"""
Oscar.jl-compatible API surface for Julia interop.

Provides an Oscar.jl-style interface to Flux Algebra structures.
This module maps Oscar.jl's API conventions to the Python implementations,
enabling easier translation of Julia code and potential Julia/Python interop.

Oscar.jl API patterns:
- Constructor functions: ring(), number_field(), group()
- Method naming: gens(), order(), elements()
- Type hierarchy: Ring → MPolyRing → ...
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

from flux_algebra.rings import HarmonicRing, IntervalRing, ChordIdeal
from flux_algebra.fields import TuningField, AlgebraicTone
from flux_algebra.groups import TranspositionInversionGroup, PLRGroup, Triad, PermutationVoiceLeading
from flux_algebra.geometry import DialPolytope, TraditionRegion, VoiceLeadingGeodesic
from flux_algebra.combinatorics import minimal_voice_leading as _min_vl
from flux_algebra.tropical import TropicalHarmony, TropicalVoiceLeading
from flux_algebra.modules import VoiceModule


# ─── Ring Constructors ───────────────────────────────────────────────────

def polynomial_ring(base_ring: str = "ZZ", modulus: int = 12, **kwargs) -> HarmonicRing:
    """Construct a harmonic ring (analog: Oscar.polynomial_ring).

    In Oscar.jl: R, x = polynomial_ring(QQ, "x")
    In Flux: R = polynomial_ring("ZZ", 12)

    Parameters
    ----------
    base_ring : str
        "ZZ" for integer ring (default), "QQ" for rationals.
    modulus : int
        Pitch-class modulus (default 12).

    Returns
    -------
    HarmonicRing
    """
    return HarmonicRing(modulus=modulus)


def integer_ring(modulus: int = 12) -> HarmonicRing:
    """Construct Z/nZ as a harmonic ring.

    Analog: Oscar.ZZ → Flux.ZZ(12).

    Parameters
    ----------
    modulus : int

    Returns
    -------
    HarmonicRing
    """
    return HarmonicRing(modulus=modulus)


def ideal(ring: HarmonicRing, gens: Sequence[int]) -> ChordIdeal:
    """Construct an ideal in a harmonic ring.

    Analog: Oscar.ideal(R, [x, y]) → Flux.ideal(R, [0, 4, 7]).

    Parameters
    ----------
    ring : HarmonicRing
        Parent ring.
    gens : sequence of int
        Generators (pitch classes).

    Returns
    -------
    ChordIdeal
    """
    return ring.chord_ideal(gens)


def gens(ring: HarmonicRing) -> list:
    """Generators of a ring (analog: Oscar.gens(R)).

    For Z/nZ, returns the standard generator: [1].

    Parameters
    ----------
    ring : HarmonicRing

    Returns
    -------
    list of int
    """
    return [1]


# ─── Number Field Constructors ───────────────────────────────────────────

def number_field(name: str = "ET-12", **kwargs) -> TuningField:
    """Construct a tuning field (analog: Oscar.number_field).

    In Oscar.jl: K, a = number_field(x^2 - 2, "a")
    In Flux: K = number_field("ET-12")

    Parameters
    ----------
    name : str
        Tuning system name: "ET-12", "ET-24", "meantone", "just", "pythagorean".

    Returns
    -------
    TuningField
    """
    name_lower = name.lower()
    if name_lower.startswith("et-"):
        n = int(name_lower.split("-")[1])
        return TuningField.ET(n)
    elif "meantone" in name_lower:
        return TuningField.meantone()
    elif name_lower == "just":
        primes = kwargs.get("primes", (2, 3, 5))
        return TuningField.just(primes=primes)
    elif name_lower == "pythagorean":
        return TuningField.pythagorean()
    else:
        return TuningField(name=name, **kwargs)


# ─── Group Constructors ──────────────────────────────────────────────────

def dihedral_group(n: int = 12) -> TranspositionInversionGroup:
    """Construct the T/I group as a dihedral group (analog: Oscar.dihedral_group).

    Parameters
    ----------
    n : int
        Number of pitch classes (default 12).

    Returns
    -------
    TranspositionInversionGroup
    """
    return TranspositionInversionGroup(modulus=n)


def permutation_group(n: int) -> PLRGroup:
    """Construct the PLR group (analog: Oscar.permutation_group).

    Parameters
    ----------
    n : int
        Modulus (default 12).

    Returns
    -------
    PLRGroup
    """
    return PLRGroup(modulus=n)


def order(G) -> int:
    """Order of a group (analog: Oscar.order(G)).

    Parameters
    ----------
    G : TranspositionInversionGroup or PLRGroup

    Returns
    -------
    int
    """
    return G.order()


def elements(G) -> list:
    """Elements of a group (analog: Oscar.elements(G)).

    Parameters
    ----------
    G : TranspositionInversionGroup

    Returns
    -------
    list
    """
    if isinstance(G, TranspositionInversionGroup):
        return G.elements()
    raise TypeError(f"elements() not implemented for {type(G)}")


# ─── Geometry Constructors ───────────────────────────────────────────────

def convex_hull(points: Sequence[Tuple[float, ...]], names: Optional[Sequence[str]] = None) -> DialPolytope:
    """Construct a dial polytope from points (analog: Oscar.convex_hull).

    Parameters
    ----------
    points : sequence of tuple
        Points in dial space.
    names : sequence of str, optional
        Names for each point.

    Returns
    -------
    DialPolytope
    """
    traditions = []
    for i, pt in enumerate(points):
        name = names[i] if names and i < len(names) else f"point_{i}"
        traditions.append(TraditionRegion(name=name, center=pt))
    return DialPolytope(traditions=traditions)


def voronoi_cells(points: Sequence[Tuple[float, ...]]) -> object:
    """Compute Voronoi diagram (analog: Oscar.voronoi_diagram).

    Parameters
    ----------
    points : sequence of tuple

    Returns
    -------
    scipy.spatial.Voronoi
    """
    from scipy.spatial import Voronoi
    import numpy as np
    return Voronoi(np.array(points))


# ─── Voice Leading ────────────────────────────────────────────────────────

def voice_leading(source: Sequence[int], target: Sequence[int], modulus: int = 12) -> list:
    """Compute minimal voice leading (Oscar-style shorthand).

    Parameters
    ----------
    source, target : sequence of int
    modulus : int

    Returns
    -------
    list of (int, int)
    """
    return _min_vl(source, target, modulus)


# ─── Tropical ─────────────────────────────────────────────────────────────

# Re-export TropicalHarmony for convenience


def tropical_semiring() -> TropicalHarmony:
    """Construct the tropical harmony object (analog: Oscar.tropical_semiring).

    Returns
    -------
    TropicalHarmony
    """
    return TropicalHarmony()


# ─── Module Constructors ─────────────────────────────────────────────────

def free_module(rank: int, modulus: int = 12) -> VoiceModule:
    """Construct a free voice module (analog: Oscar.free_module).

    Parameters
    ----------
    rank : int
        Number of voices.
    modulus : int
        Pitch-class modulus.

    Returns
    -------
    VoiceModule
    """
    return VoiceModule(rank=rank, modulus=modulus)


# ─── Oscar-style Chord/Triad ─────────────────────────────────────────────

def major_triad(root: int) -> Triad:
    """Construct a major triad.

    Parameters
    ----------
    root : int
        Root pitch class.

    Returns
    -------
    Triad
    """
    return Triad(root=root, quality="major")


def minor_triad(root: int) -> Triad:
    """Construct a minor triad.

    Parameters
    ----------
    root : int
        Root pitch class.

    Returns
    -------
    Triad
    """
    return Triad(root=root, quality="minor")
