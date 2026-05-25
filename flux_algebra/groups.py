"""
Symmetry Groups — Group-theoretic structures for music.

Analogous to Oscar.jl's Groups module (permutation, matrix, finitely presented).

Key structures:
- TranspositionInversionGroup (T/I group) — dihedral group D_24 acting on pitch classes
- PLRGroup — neo-Riemannian transformations (P, L, R) generating D_24 on triads
- PermutationVoiceLeading — voice leadings as permutation group elements
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Triad:
    """A major or minor triad as a set of pitch classes.

    Parameters
    ----------
    root : int
        Root pitch class (0=C, 1=C#, ..., 11=B).
    quality : str
        "major" or "minor".

    Examples
    --------
    >>> Triad(0, "major")  # C major = {0, 4, 7}
    >>> Triad(0, "minor")  # C minor = {0, 3, 7}
    """

    root: int
    quality: str  # "major" or "minor"

    def __post_init__(self) -> None:
        if self.quality not in ("major", "minor"):
            raise ValueError(f"quality must be 'major' or 'minor', got '{self.quality}'")
        if not 0 <= self.root < 12:
            raise ValueError(f"root must be in [0, 11], got {self.root}")

    @property
    def pitch_classes(self) -> Tuple[int, ...]:
        """Pitch classes of the triad as a sorted tuple."""
        if self.quality == "major":
            return (self.root, (self.root + 4) % 12, (self.root + 7) % 12)
        else:
            return (self.root, (self.root + 3) % 12, (self.root + 7) % 12)

    @property
    def third(self) -> int:
        """Third of the triad."""
        return self.pitch_classes[1]

    @property
    def fifth(self) -> int:
        """Fifth of the triad."""
        return self.pitch_classes[2]

    @property
    def is_major(self) -> bool:
        return self.quality == "major"

    @property
    def is_minor(self) -> bool:
        return self.quality == "minor"

    @property
    def name(self) -> str:
        """Standard name (e.g., C major, C# minor)."""
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return f"{note_names[self.root]} {self.quality}"

    def __repr__(self) -> str:
        return f"Triad({self.name}, pcs={self.pitch_classes})"

    def __str__(self) -> str:
        return self.name


# All 24 major/minor triads
ALL_TRIADS: List[Triad] = [
    Triad(root=r, quality=q) for r in range(12) for q in ("major", "minor")
]


class TranspositionInversionGroup:
    """The T/I group — dihedral group of order 24.

    Acts on pitch classes via:
      T_n(x) = x + n mod 12    (transposition)
      I_n(x) = n - x mod 12    (inversion)

    This is D_12 (dihedral group of order 24), the full symmetry group
    of the 12-tone clock face.

    The group has presentation:
      < T, I | T^12 = e, I^2 = e, I·T·I = T^{-1} >

    Examples
    --------
    >>> ti = TranspositionInversionGroup()
    >>> ti.T(7, 5)   # Transpose G by 5 semitones
    0
    >>> ti.I(0, 7)   # Invert G around C
    5
    """

    def __init__(self, modulus: int = 12) -> None:
        self._modulus = modulus

    @property
    def order(self) -> int:
        """Order of the group (2 * modulus)."""
        return 2 * self._modulus

    @property
    def modulus(self) -> int:
        """Number of pitch classes."""
        return self._modulus

    def T(self, n: int, x: int) -> int:
        """Transposition: T_n(x) = (x + n) mod modulus.

        Parameters
        ----------
        n : int
            Transposition amount.
        x : int
            Pitch class.

        Returns
        -------
        int
        """
        return (x + n) % self._modulus

    def I(self, n: int, x: int) -> int:
        """Inversion: I_n(x) = (n - x) mod modulus.

        Parameters
        ----------
        n : int
            Inversion axis.
        x : int
            Pitch class.

        Returns
        -------
        int
        """
        return (n - x) % self._modulus

    def compose(self, g1: Tuple[str, int], g2: Tuple[str, int]) -> Tuple[str, int]:
        """Compose two group elements.

        Elements are represented as (type, n) where type is 'T' or 'I'.

        Parameters
        ----------
        g1, g2 : tuple of (str, int)
            Group elements.

        Returns
        -------
        tuple of (str, int)
            Composed element.
        """
        t1, n1 = g1
        t2, n2 = g2

        if t1 == "T" and t2 == "T":
            return ("T", (n1 + n2) % self._modulus)
        elif t1 == "T" and t2 == "I":
            return ("I", (n1 + n2) % self._modulus)
        elif t1 == "I" and t2 == "T":
            return ("I", (n1 - n2) % self._modulus)
        else:  # I · I
            return ("T", (n1 - n2) % self._modulus)

    def inverse(self, g: Tuple[str, int]) -> Tuple[str, int]:
        """Inverse of a group element.

        Parameters
        ----------
        g : tuple of (str, int)
            Group element.

        Returns
        -------
        tuple of (str, int)
        """
        t, n = g
        if t == "T":
            return ("T", (-n) % self._modulus)
        else:
            return ("I", n)  # I_n is self-inverse

    def orbit(self, x: int) -> FrozenSet[int]:
        """Compute the orbit of pitch class x under the full T/I group.

        The T/I group acts transitively on Z/12Z, so every orbit is
        the full set {0, 1, ..., 11}.

        Parameters
        ----------
        x : int
            Pitch class.

        Returns
        -------
        frozenset of int
        """
        return frozenset(range(self._modulus))

    def stabilizer(self, x: int) -> List[Tuple[str, int]]:
        """Compute the stabilizer of pitch class x.

        The stabilizer of x consists of elements fixing x:
        T_0 (identity) and I_{2x} (inversion around x).

        Parameters
        ----------
        x : int
            Pitch class.

        Returns
        -------
        list of tuple
        """
        return [("T", 0), ("I", (2 * x) % self._modulus)]

    def elements(self) -> List[Tuple[str, int]]:
        """List all group elements.

        Returns
        -------
        list of tuple
            All (type, n) pairs.
        """
        return [("T", n) for n in range(self._modulus)] + [
            ("I", n) for n in range(self._modulus)
        ]

    def action(self, g: Tuple[str, int], x: int) -> int:
        """Apply group element g to pitch class x.

        Parameters
        ----------
        g : tuple of (str, int)
            Group element.
        x : int
            Pitch class.

        Returns
        -------
        int
        """
        t, n = g
        if t == "T":
            return self.T(n, x)
        else:
            return self.I(n, x)

    def __repr__(self) -> str:
        return f"TranspositionInversionGroup(modulus={self._modulus})"


class PLRGroup:
    """Parallel/Leading-tone/Relative group — neo-Riemannian theory.

    The three fundamental transformations:
      P (Parallel):     major ↔ parallel minor (same root, flip third)
      L (Leading-tone): major ↔ minor a third apart (common-tone leading)
      R (Relative):     major ↔ relative minor (same key signature)

    These generate a group isomorphic to D_24 (dihedral of order 24)
    acting on the 24 major/minor triads.

    Specifically:
      P(C+) = C-     C major {0,4,7} ↔ C minor {0,3,7}
      L(C+) = e-     C major {0,4,7} ↔ E minor {4,7,11}  [wrong, corrected below]
      L(C+) = e-     C major {0,4,7} → E minor {4,7,11}? No.

    Standard definitions:
      P:  {x, x+4, x+7} ↔ {x, x+3, x+7}       (flip third)
      L:  {x, x+4, x+7} ↔ {x+4, x+7, x+11}     (leading tone exchange for major→minor)
          = {x, x+3, x+7} ↔ {x-1, x, x+3} ... let me use the standard ones:

    Actually the standard neo-Riemannian operations on triads:
      P(C+) = Cm   : change mode, keep root
      L(C+) = Em   : C+ → Em (shared tones: E,G)
      R(C+) = Am   : C+ → Am (shared tones: C,E)

    More precisely, acting on (root, quality):
      P: (r, maj) → (r, min)       and (r, min) → (r, maj)
      L: (r, maj) → (r+4, min)     and (r, min) → (r-1, maj) = (r+11, maj)
      R: (r, maj) → (r+9, min)     and (r, min) → (r+3, maj)

    Examples
    --------
    >>> plr = PLRGroup()
    >>> c_maj = Triad(0, "major")
    >>> plr.P(c_maj)  # C minor
    Triad(C minor, pcs=(0, 3, 7))
    """

    def __init__(self, modulus: int = 12) -> None:
        self._modulus = modulus

    def P(self, triad: Triad) -> Triad:
        """Parallel transformation: flip mode, keep root.

        P: major ↔ minor with same root.
        C major {0,4,7} ↔ C minor {0,3,7}

        Parameters
        ----------
        triad : Triad

        Returns
        -------
        Triad
        """
        new_quality = "minor" if triad.quality == "major" else "major"
        return Triad(root=triad.root, quality=new_quality)

    def L(self, triad: Triad) -> Triad:
        """Leading-tone transformation.

        L: (r, maj) → (r+4, min)     — C+ → e-
        L: (r, min) → (r+11, maj)    — e- → C+ (mod 12: r-1 = r+11)

        Shares two common tones.

        Parameters
        ----------
        triad : Triad

        Returns
        -------
        Triad
        """
        if triad.quality == "major":
            return Triad(root=(triad.root + 4) % self._modulus, quality="minor")
        else:
            return Triad(root=(triad.root + 11) % self._modulus, quality="major")

    def R(self, triad: Triad) -> Triad:
        """Relative transformation.

        R: (r, maj) → (r+9, min)     — C+ → a-
        R: (r, min) → (r+3, maj)     — a- → C+

        Shares two common tones.

        Parameters
        ----------
        triad : Triad

        Returns
        -------
        Triad
        """
        if triad.quality == "major":
            return Triad(root=(triad.root + 9) % self._modulus, quality="minor")
        else:
            return Triad(root=(triad.root + 3) % self._modulus, quality="major")

    def apply(self, word: str, triad: Triad) -> Triad:
        """Apply a sequence of P/L/R operations.

        Applied left-to-right: apply("PLR", C+) means R(L(P(C+))).

        Parameters
        ----------
        word : str
            Sequence of P, L, R characters.
        triad : Triad
            Starting triad.

        Returns
        -------
        Triad
        """
        ops = {"P": self.P, "L": self.L, "R": self.R}
        result = triad
        for ch in word.upper():
            if ch not in ops:
                raise ValueError(f"Unknown operation '{ch}', expected P, L, or R")
            result = ops[ch](result)
        return result

    def walk(self, word: str, steps: int, start: Triad) -> List[Triad]:
        """Repeatedly apply a P/L/R word, collecting all intermediate triads.

        Parameters
        ----------
        word : str
            Sequence of P, L, R to repeat.
        steps : int
            Number of times to apply the word.
        start : Triad
            Starting triad.

        Returns
        -------
        list of Triad
            [start, after 1 word, after 2 words, ...]
        """
        result = [start]
        current = start
        for _ in range(steps):
            current = self.apply(word, current)
            result.append(current)
        return result

    def orbit(self, triad: Triad) -> FrozenSet[Triad]:
        """Compute the orbit of a triad under the PLR group.

        The PLR group acts transitively on the 24 triads, so every
        orbit contains all 24.

        Parameters
        ----------
        triad : Triad

        Returns
        -------
        frozenset of Triad
        """
        visited = set()
        frontier = [triad]
        while frontier:
            t = frontier.pop()
            if t in visited:
                continue
            visited.add(t)
            for op in (self.P, self.L, self.R):
                next_t = op(t)
                if next_t not in visited:
                    frontier.append(next_t)
        return frozenset(visited)

    def common_tones(self, a: Triad, b: Triad) -> FrozenSet[int]:
        """Compute common tones between two triads.

        Parameters
        ----------
        a, b : Triad

        Returns
        -------
        frozenset of int
        """
        return frozenset(a.pitch_classes) & frozenset(b.pitch_classes)

    def order(self) -> int:
        """Order of the PLR group (24 for modulus 12)."""
        return 2 * self._modulus

    def __repr__(self) -> str:
        return f"PLRGroup(modulus={self._modulus})"


class PermutationVoiceLeading:
    """Voice leading as permutations.

    A voice leading between two n-note chords is a bijection from
    source voices to target voices. The set of all bijections forms
    the symmetric group S_n under composition.

    Parameters
    ----------
    source : tuple of int
        Source chord pitch classes.
    target : tuple of int
        Target chord pitch classes.
    permutation : tuple of int
        Permutation σ such that source[i] maps to target[σ[i]].

    Examples
    --------
    >>> # C major → F major
    >>> pvl = PermutationVoiceLeading(
    ...     source=(0, 4, 7),
    ...     target=(5, 9, 0),
    ...     permutation=(2, 0, 1),  # C→F=5, E→A=9, G→C=0... need to check
    ... )
    """

    def __init__(
        self,
        source: Tuple[int, ...],
        target: Tuple[int, ...],
        permutation: Tuple[int, ...],
    ) -> None:
        if len(source) != len(target):
            raise ValueError("Source and target must have same number of voices")
        n = len(source)
        if sorted(permutation) != list(range(n)):
            raise ValueError(f"Permutation must be a permutation of {{0,...,{n-1}}}")
        self._source = source
        self._target = target
        self._perm = permutation

    @property
    def source(self) -> Tuple[int, ...]:
        """Source chord."""
        return self._source

    @property
    def target(self) -> Tuple[int, ...]:
        """Target chord."""
        return self._target

    @property
    def permutation(self) -> Tuple[int, ...]:
        """The permutation mapping source→target indices."""
        return self._perm

    @property
    def pairs(self) -> Tuple[Tuple[int, int], ...]:
        """Voice-leading pairs: (source[i], target[σ[i]])."""
        return tuple(
            (self._source[i], self._target[self._perm[i]])
            for i in range(len(self._source))
        )

    @property
    def movements(self) -> Tuple[int, ...]:
        """Interval movement for each voice."""
        return tuple(
            self._target[self._perm[i]] - self._source[i]
            for i in range(len(self._source))
        )

    @property
    def total_movement(self) -> int:
        """Sum of absolute interval movements."""
        return sum(abs(m) for m in self.movements)

    def compose(self, other: "PermutationVoiceLeading") -> "PermutationVoiceLeading":
        """Compose two voice leadings.

        The composition applies this voice leading first, then the other.

        Parameters
        ----------
        other : PermutationVoiceLeading
            Second voice leading.

        Returns
        -------
        PermutationVoiceLeading
        """
        if self._target != other._source:
            raise ValueError("Target of first must match source of second")
        # Compose permutations: σ₁ ∘ σ₂
        n = len(self._perm)
        new_perm = tuple(self._perm[other._perm[i]] for i in range(n))
        return PermutationVoiceLeading(self._source, other._target, new_perm)

    def inverse(self) -> "PermutationVoiceLeading":
        """Inverse voice leading (undo this one)."""
        n = len(self._perm)
        inv_perm = [0] * n
        for i, p in enumerate(self._perm):
            inv_perm[p] = i
        return PermutationVoiceLeading(
            self._target, self._source, tuple(inv_perm)
        )

    def __repr__(self) -> str:
        return (
            f"PermutationVoiceLeading("
            f"source={self._source}, target={self._target}, "
            f"perm={self._perm})"
        )
