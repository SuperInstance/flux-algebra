"""
Tuning Fields — Number-field analog for tuning systems.

Analogous to Oscar.jl's NumberTheory module (number fields, algebraic numbers).

Instead of number fields Q(α), we define tuning fields: algebraic extensions
of Q that encode the frequency ratios of a tuning system.

- ET-12: Q(ζ₁₂) — 12th roots of unity (equal temperament)
- Meantone: Q(√5) — the meantone fifth involves √5
- Just: Q — rational ratios (no extension needed)
- Pythagorean: Q(3/2) = Q — 3-limit just intonation
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Tuple
import math


class TuningField:
    """Field extension for a specific tuning system.

    A tuning field is an algebraic extension of Q whose elements are
    frequency ratios. The field is determined by the minimal polynomial
    of the generator.

    Parameters
    ----------
    name : str
        Name of the tuning system.
    generator : float or Fraction
        Algebraic generator (a frequency ratio).
    minimal_polynomial_coeffs : tuple of float, optional
        Coefficients of the minimal polynomial of the generator over Q.
        Given as (a_n, a_{n-1}, ..., a_1, a_0) for
        a_n·x^n + ... + a_1·x + a_0 = 0.
    primes : tuple of int
        Prime factors used in this tuning (e.g., (2,3,5) for 5-limit).

    Examples
    --------
    >>> # 12-tone equal temperament
    >>> et12 = TuningField.ET(12)
    >>> et12.degree
    12
    >>> # 5-limit just intonation
    >>> just5 = TuningField.just(primes=(2, 3, 5))
    >>> just5.name
    'just-5-limit'
    """

    def __init__(
        self,
        name: str,
        generator: float = 2.0,
        minimal_polynomial_coeffs: Optional[Tuple[float, ...]] = None,
        primes: Tuple[int, ...] = (2, 3, 5),
    ) -> None:
        self._name = name
        self._generator = generator
        self._min_poly = minimal_polynomial_coeffs
        self._primes = primes

    @classmethod
    def ET(cls, n: int = 12) -> "TuningField":
        """Create an equal-temperament tuning field.

        The generator is 2^(1/n), a root of x^n - 2 = 0.
        This creates the field Q(2^(1/n)).

        Parameters
        ----------
        n : int
            Number of divisions per octave (default 12).

        Returns
        -------
        TuningField
        """
        return cls(
            name=f"ET-{n}",
            generator=2.0 ** (1.0 / n),
            minimal_polynomial_coeffs=(1.0,) + (0.0,) * (n - 1) + (-2.0,),
            primes=(2,),
        )

    @classmethod
    def meantone(cls, quarter: bool = True) -> "TuningField":
        """Create a meantone tuning field.

        Quarter-comma meantone tempers the fifth so that four fifths
        give a pure major third (5/4). The generator satisfies
        4x = 5/1 in log space, so x = 5^(1/4) / 2.

        Parameters
        ----------
        quarter : bool
            If True, use quarter-comma meantone (default).
            If False, use third-comma meantone.

        Returns
        -------
        TuningField
        """
        if quarter:
            # Fifth = 5^(1/4), minimal polynomial: x^4 - 5 = 0
            return cls(
                name="quarter-comma-meantone",
                generator=5.0 ** 0.25,
                minimal_polynomial_coeffs=(1.0, 0.0, 0.0, 0.0, -5.0),
                primes=(2, 5),
            )
        else:
            # Third-comma: fifth = 2 * 6^(1/3) / 3
            return cls(
                name="third-comma-meantone",
                generator=(6.0 ** (1.0 / 3.0)),
                minimal_polynomial_coeffs=(1.0, 0.0, 0.0, -6.0),
                primes=(2, 3),
            )

    @classmethod
    def just(cls, primes: Tuple[int, ...] = (2, 3, 5)) -> "TuningField":
        """Create a just-intonation tuning field.

        Just intonation uses exact rational ratios. The field is Q itself.

        Parameters
        ----------
        primes : tuple of int
            Prime limit (default (2,3,5) = 5-limit).

        Returns
        -------
        TuningField
        """
        name = f"just-{'-'.join(str(p) for p in primes)}-limit"
        return cls(name=name, generator=1.0, primes=primes)

    @classmethod
    def pythagorean(cls) -> "TuningField":
        """Create a Pythagorean tuning field (3-limit just intonation).

        All ratios are powers of 2 and 3 only.

        Returns
        -------
        TuningField
        """
        return cls.just(primes=(2, 3))

    @property
    def name(self) -> str:
        """Name of the tuning system."""
        return self._name

    @property
    def generator(self) -> float:
        """The algebraic generator of this field."""
        return self._generator

    @property
    def degree(self) -> Optional[int]:
        """Degree of the field extension [F : Q].

        None if not specified.
        """
        if self._min_poly is None:
            return None
        return len(self._min_poly) - 1

    @property
    def primes(self) -> Tuple[int, ...]:
        """Prime factors used in this tuning."""
        return self._primes

    @property
    def minimal_polynomial(self) -> Optional[Tuple[float, ...]]:
        """Coefficients of the minimal polynomial of the generator."""
        return self._min_poly

    def ratio_to_cents(self, ratio: float) -> float:
        """Convert a frequency ratio to cents.

        Parameters
        ----------
        ratio : float
            Frequency ratio.

        Returns
        -------
        float
            Size in cents.
        """
        return 1200.0 * math.log2(ratio)

    def cents_to_ratio(self, cents: float) -> float:
        """Convert cents to a frequency ratio in this tuning.

        Parameters
        ----------
        cents : float
            Size in cents.

        Returns
        -------
        float
            Frequency ratio.
        """
        return 2.0 ** (cents / 1200.0)

    def step_cents(self) -> float:
        """Size of one step in this tuning, in cents.

        For ET-n, this is 1200/n.
        """
        if self._name.startswith("ET-"):
            n = int(self._name.split("-")[1])
            return 1200.0 / n
        return self.ratio_to_cents(self._generator)

    def approximate_ratio(self, ratio: float) -> Fraction:
        """Approximate a frequency ratio as a rational in this field.

        Uses continued-fraction approximation within the prime limit.

        Parameters
        ----------
        ratio : float
            Frequency ratio to approximate.

        Returns
        -------
        Fraction
            Best rational approximation.
        """
        return Fraction(ratio).limit_denominator(10000)

    def __repr__(self) -> str:
        return f"TuningField('{self._name}')"


@dataclass(frozen=True)
class AlgebraicTone:
    """A tone defined as an algebraic number over Q.

    Represents a pitch as a root of a minimal polynomial, enabling
    exact arithmetic for tuning systems beyond just intonation.

    Parameters
    ----------
    ratio : float
        Frequency ratio relative to the reference pitch.
    minimal_polynomial : tuple of float, optional
        Coefficients of the minimal polynomial.
    name : str, optional
        Human-readable name (e.g., "meantone M3").
    description : str, optional
        Longer description.

    Examples
    --------
    >>> # Meantone major third: pure 5/4
    >>> mt_m3 = AlgebraicTone(
    ...     ratio=1.25,
    ...     minimal_polynomial=(1, 0, 0, 0, -5),  # x^4 = 5 approximation
    ...     name="meantone-M3",
    ... )
    >>> mt_m3.cents
    386.3137138648348
    """

    ratio: float
    minimal_polynomial: Optional[Tuple[float, ...]] = None
    name: Optional[str] = None
    description: Optional[str] = None

    @property
    def cents(self) -> float:
        """Size of this tone in cents relative to unison."""
        return 1200.0 * math.log2(self.ratio)

    @property
    def semitones(self) -> float:
        """Approximate size in semitones (cents / 100)."""
        return self.cents / 100.0

    @property
    def degree(self) -> Optional[int]:
        """Algebraic degree of this tone over Q."""
        if self.minimal_polynomial is None:
            return None
        return len(self.minimal_polynomial) - 1

    def compose(self, other: "AlgebraicTone") -> "AlgebraicTone":
        """Compose two tones (multiply their frequency ratios).

        Parameters
        ----------
        other : AlgebraicTone
            Another tone.

        Returns
        -------
        AlgebraicTone
            Composed tone.
        """
        return AlgebraicTone(
            ratio=self.ratio * other.ratio,
            name=f"({self.name or '?'} × {other.name or '?'})",
        )

    def invert(self) -> "AlgebraicTone":
        """Invert this tone (1/ratio).

        Returns
        -------
        AlgebraicTone
            Inverted tone.
        """
        return AlgebraicTone(
            ratio=1.0 / self.ratio,
            name=f"inverse({self.name or '?'})",
        )

    def stack(self, n: int) -> "AlgebraicTone":
        """Stack this tone n times (ratio^n).

        Parameters
        ----------
        n : int
            Number of times to stack.

        Returns
        -------
        AlgebraicTone
        """
        return AlgebraicTone(
            ratio=self.ratio ** n,
            name=f"{n}×({self.name or '?'})",
        )

    @classmethod
    def from_cents(cls, cents: float, name: Optional[str] = None) -> "AlgebraicTone":
        """Create an AlgebraicTone from a cent value.

        Parameters
        ----------
        cents : float
            Size in cents.
        name : str, optional
            Name for this tone.

        Returns
        -------
        AlgebraicTone
        """
        ratio = 2.0 ** (cents / 1200.0)
        return cls(ratio=ratio, name=name)

    @classmethod
    def from_fraction(cls, num: int, den: int, name: Optional[str] = None) -> "AlgebraicTone":
        """Create an AlgebraicTone from a just-intonation ratio.

        Parameters
        ----------
        num, den : int
            Numerator and denominator.
        name : str, optional
            Name.

        Returns
        -------
        AlgebraicTone
        """
        ratio = num / den
        return cls(ratio=ratio, name=name or f"{num}/{den}")

    def __repr__(self) -> str:
        label = f" '{self.name}'" if self.name else ""
        return f"AlgebraicTone(ratio={self.ratio:.6f}{label}, cents={self.cents:.1f})"
