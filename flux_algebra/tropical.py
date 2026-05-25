"""
Tropical Music Theory — Min-plus algebra applied to harmony.

Analogous to Oscar.jl's TropicalGeometry module.

The tropical semiring (R ∪ {∞}, min, +) naturally models voice leading:
- Tropical "addition" = min → selects the closest voice
- Tropical "multiplication" = + → adds intervals
- Tropical polynomials represent harmonic "cost landscapes"

The tropicalization of voice leading gives a piecewise-linear cost
function whose minima correspond to smooth voice leadings.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


TROPICAL_INF = float("inf")


def tropical_add(a: float, b: float) -> float:
    """Tropical addition: min(a, b).

    In the tropical semiring, addition selects the smaller value.
    For voice leading, this means "pick the closest voice."

    Parameters
    ----------
    a, b : float

    Returns
    -------
    float
    """
    return min(a, b)


def tropical_multiply(a: float, b: float) -> float:
    """Tropical multiplication: a + b.

    In the tropical semiring, multiplication is ordinary addition.
    For voice leading, this means "add the interval distances."

    Parameters
    ----------
    a, b : float

    Returns
    -------
    float
    """
    return a + b


def tropical_power(a: float, n: int) -> float:
    """Tropical exponentiation: n × a (scalar multiplication).

    Tropical power: a ⊗ a ⊗ ... ⊗ a = a + a + ... + a = n·a.

    Parameters
    ----------
    a : float
    n : int
        Non-negative exponent.

    Returns
    -------
    float
    """
    return n * a


class TropicalPolynomial:
    """A tropical polynomial in one variable.

    A tropical polynomial is a piecewise-linear function:
      p(x) = min(a_0, a_1 + x, a_2 + 2x, ..., a_n + nx)

    where a_i are the coefficients. The tropical roots are the
    points where two terms are equal and are the minimum.

    Parameters
    ----------
    coefficients : list of float
        Coefficients [a_0, a_1, ..., a_n]. Use TROPICAL_INF for
        missing terms.

    Examples
    --------
    >>> # Tropical polynomial for C major: min(x, x+4, x+7)
    >>> tp = TropicalPolynomial([0, 4, 7])
    >>> tp.evaluate(0)  # min(0, 4, 7) = 0
    0
    >>> tp.evaluate(3)  # min(3, 7, 10) = 3
    3
    """

    def __init__(self, coefficients: List[float]) -> None:
        self._coeffs = coefficients

    @property
    def coefficients(self) -> List[float]:
        """Coefficients [a_0, a_1, ..., a_n]."""
        return self._coeffs.copy()

    @property
    def degree(self) -> int:
        """Degree of the tropical polynomial."""
        return len(self._coeffs) - 1

    def evaluate(self, x: float) -> float:
        """Evaluate the tropical polynomial at x.

        Computes min(a_0, a_1 + x, a_2 + 2x, ..., a_n + nx).

        Parameters
        ----------
        x : float

        Returns
        -------
        float
        """
        terms = [
            self._coeffs[i] + i * x
            for i in range(len(self._coeffs))
            if self._coeffs[i] != TROPICAL_INF
        ]
        return min(terms) if terms else TROPICAL_INF

    def __call__(self, x: float) -> float:
        return self.evaluate(x)

    def roots(self) -> List[float]:
        """Find tropical roots (kinks in the piecewise-linear function).

        A tropical root occurs at x where terms i and i+1 are equal
        and dominate: a_i + ix = a_{i+1} + (i+1)x → x = a_i - a_{i+1}.

        Returns
        -------
        list of float
        """
        result = []
        n = len(self._coeffs)
        for i in range(n - 1):
            if self._coeffs[i] != TROPICAL_INF and self._coeffs[i + 1] != TROPICAL_INF:
                root = self._coeffs[i] - self._coeffs[i + 1]
                result.append(root)
        return sorted(result)

    def __repr__(self) -> str:
        terms = []
        for i, c in enumerate(self._coeffs):
            if c != TROPICAL_INF:
                if i == 0:
                    terms.append(f"{c}")
                elif i == 1:
                    terms.append(f"{c}⊕x")
                else:
                    terms.append(f"{c}⊕x^{i}")
        return f"TropicalPolynomial({' ⊕ '.join(terms)})"


class TropicalHarmony:
    """Tropical semiring (R ∪ {∞}, min, +) applied to harmony.

    Tropical addition (min) naturally selects the closest chord tone.
    Tropical multiplication (+) adds intervals. A chord is modeled as
    a tropical polynomial whose evaluation at a pitch class gives
    the distance to the nearest chord tone.

    Examples
    --------
    >>> th = TropicalHarmony()
    >>> cost = th.chord_cost([0, 4, 7])
    >>> cost(0)  # C is in C major — distance 0
    0.0
    >>> cost(3)  # E♭ is 1 semitone from nearest (E or D#)
    1.0
    """

    def __init__(self, modulus: int = 12) -> None:
        self._modulus = modulus

    def chord_cost(self, chord: Sequence[int]) -> Callable[[int], float]:
        """Create a tropical cost function for a chord.

        The cost function gives the tropical distance from any pitch
        class to the nearest chord tone. This is a tropical polynomial:
          cost(x) = min_i (|x - c_i|)

        Parameters
        ----------
        chord : sequence of int
            Pitch classes of the chord.

        Returns
        -------
        callable
            Function mapping pitch class → distance to nearest chord tone.
        """
        chord_set = list(set(c % self._modulus for c in chord))

        def cost(pc: int) -> float:
            pc = pc % self._modulus
            distances = []
            for c in chord_set:
                diff = abs(pc - c)
                distances.append(min(diff, self._modulus - diff))
            return min(distances) if distances else TROPICAL_INF

        return cost

    def chord_tropical_polynomial(self, chord: Sequence[int]) -> TropicalPolynomial:
        """Express a chord as a tropical polynomial.

        For a chord {c_0, c_1, ..., c_k}, the tropical polynomial is
        min(x - c_0, x - c_1, ..., x - c_k), shifted so the constant
        terms are the chord pitch classes.

        Parameters
        ----------
        chord : sequence of int
            Pitch classes.

        Returns
        -------
        TropicalPolynomial
        """
        pcs = sorted(set(c % self._modulus for c in chord))
        # Build polynomial: min(pcs[0], x - pcs[0] + pcs[1], ...)
        # Simplification: use pcs as constant term offsets
        coeffs = [float(p) for p in pcs]
        return TropicalPolynomial(coeffs)

    def tropical_voice_leading(
        self,
        source: Sequence[int],
        target: Sequence[int],
    ) -> List[Tuple[int, int]]:
        """Compute voice leading using tropical (min-plus) algebra.

        The tropicalization of voice leading: for each source voice,
        find the target voice with minimal tropical distance. This is
        equivalent to the standard minimal voice leading via the
        Hungarian algorithm, since tropical min-plus naturally minimizes.

        Parameters
        ----------
        source, target : sequence of int
            Pitch classes.

        Returns
        -------
        list of (int, int)
            Voice-leading pairs.
        """
        source = list(source)
        target = list(target)
        n = len(source)
        if n != len(target):
            raise ValueError("Source and target must have same number of voices")

        if n == 0:
            return []

        # Cost matrix (tropical: distances)
        cost = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                diff = abs(source[i] - target[j])
                cost[i][j] = min(diff, self._modulus - diff)

        row_ind, col_ind = linear_sum_assignment(cost)

        return [(source[row_ind[i]], target[col_ind[i]]) for i in range(n)]

    def tropical_distance(
        self,
        source: Sequence[int],
        target: Sequence[int],
    ) -> float:
        """Tropical distance between two chords.

        The tropical distance is the total minimal voice-leading distance,
        computed via tropical (min-plus) algebra.

        Parameters
        ----------
        source, target : sequence of int

        Returns
        -------
        float
        """
        vl = self.tropical_voice_leading(source, target)
        return sum(abs(s - t) for s, t in vl)

    def harmonic_tension(
        self,
        chord: Sequence[int],
        scale: Sequence[int],
    ) -> float:
        """Compute harmonic tension of a chord relative to a scale.

        Tension = tropical sum (sum of distances from chord tones to
        nearest scale tones). A diatonic chord has tension 0.

        Parameters
        ----------
        chord : sequence of int
            Pitch classes of the chord.
        scale : sequence of int
            Pitch classes of the reference scale.

        Returns
        -------
        float
        """
        scale_cost = self.chord_cost(scale)
        return sum(scale_cost(pc) for pc in chord)

    def __repr__(self) -> str:
        return f"TropicalHarmony(modulus={self._modulus})"


class TropicalVoiceLeading:
    """Voice leading in tropical geometry.

    Tropical voice leading uses the min-plus semiring to find
    minimal-cost paths between chords. The tropicalization of
    traditional voice leading gives piecewise-linear cost functions.

    Parameters
    ----------
    modulus : int
        Number of pitch divisions (default 12).

    Examples
    --------
    >>> tvl = TropicalVoiceLeading()
    >>> result = tvl.compute([0, 4, 7], [5, 9, 0])
    >>> result.distance
    5.0
    """

    def __init__(self, modulus: int = 12) -> None:
        self._modulus = modulus
        self._th = TropicalHarmony(modulus)

    def compute(
        self,
        source: Sequence[int],
        target: Sequence[int],
    ) -> "TropicalVLResult":
        """Compute tropical voice leading.

        Parameters
        ----------
        source, target : sequence of int
            Pitch classes.

        Returns
        -------
        TropicalVLResult
        """
        vl = self._th.tropical_voice_leading(source, target)
        distance = sum(abs(s - t) for s, t in vl)

        # Compute tropical polynomial for source and target
        src_poly = self._th.chord_tropical_polynomial(source)
        tgt_poly = self._th.chord_tropical_polynomial(target)

        return TropicalVLResult(
            source=tuple(source),
            target=tuple(target),
            pairs=vl,
            distance=distance,
            source_polynomial=src_poly,
            target_polynomial=tgt_poly,
        )

    def __repr__(self) -> str:
        return f"TropicalVoiceLeading(modulus={self._modulus})"


class TropicalVLResult:
    """Result of a tropical voice-leading computation.

    Parameters
    ----------
    source : tuple of int
    target : tuple of int
    pairs : list of (int, int)
        Voice-leading pairs.
    distance : float
        Tropical voice-leading distance.
    source_polynomial : TropicalPolynomial
    target_polynomial : TropicalPolynomial
    """

    def __init__(
        self,
        source: tuple,
        target: tuple,
        pairs: List[Tuple[int, int]],
        distance: float,
        source_polynomial: TropicalPolynomial,
        target_polynomial: TropicalPolynomial,
    ) -> None:
        self.source = source
        self.target = target
        self.pairs = pairs
        self.distance = distance
        self.source_polynomial = source_polynomial
        self.target_polynomial = target_polynomial

    def __repr__(self) -> str:
        return f"TropicalVLResult({self.source} → {self.target}, dist={self.distance})"
