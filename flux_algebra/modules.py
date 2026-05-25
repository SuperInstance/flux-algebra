"""
Voice Modules — Free modules over harmonic rings.

Analogous to Oscar.jl's Modules module.

A voice module is a free Z/nZ-module of rank k, where k is the number
of voices. Each voice is a basis element, and voice leadings are module
homomorphisms (matrices over the ring).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


class VoiceModule:
    """Free module over a harmonic ring (Z/nZ)^k.

    Represents k independent voices over a ring of pitch classes modulo n.
    Voice leadings are module homomorphisms: k×k matrices over Z/nZ.

    Parameters
    ----------
    rank : int
        Number of voices (rank of the free module).
    modulus : int
        Pitch-class modulus (default 12).

    Examples
    --------
    >>> vm = VoiceModule(rank=4, modulus=12)
    >>> vm.zero  # All voices at unison
    array([0, 0, 0, 0])
    >>> # SATB chord: S=C, A=E, T=G, B=C
    >>> vm.act([0, 4, 7, 0], transposition=2)
    array([ 2,  6,  9,  2])
    """

    def __init__(self, rank: int, modulus: int = 12) -> None:
        if rank < 1:
            raise ValueError(f"rank must be ≥ 1, got {rank}")
        self._rank = rank
        self._modulus = modulus

    @property
    def rank(self) -> int:
        """Number of voices."""
        return self._rank

    @property
    def modulus(self) -> int:
        """Pitch-class modulus."""
        return self._modulus

    @property
    def zero(self) -> np.ndarray:
        """Zero element of the module (all voices silent)."""
        return np.zeros(self._rank, dtype=int)

    def add(self, a: Sequence[int], b: Sequence[int]) -> np.ndarray:
        """Add two module elements (voice-wise transposition).

        Parameters
        ----------
        a, b : sequence of int
            Module elements (one pitch class per voice).

        Returns
        -------
        np.ndarray
            Element-wise sum mod n.
        """
        a, b = np.array(a), np.array(b)
        if len(a) != self._rank or len(b) != self._rank:
            raise ValueError(f"Expected length {self._rank}")
        return (a + b) % self._modulus

    def subtract(self, a: Sequence[int], b: Sequence[int]) -> np.ndarray:
        """Subtract module elements.

        Parameters
        ----------
        a, b : sequence of int

        Returns
        -------
        np.ndarray
        """
        a, b = np.array(a), np.array(b)
        if len(a) != self._rank or len(b) != self._rank:
            raise ValueError(f"Expected length {self._rank}")
        return (a - b) % self._modulus

    def scalar_multiply(self, scalar: int, element: Sequence[int]) -> np.ndarray:
        """Scalar multiplication: scalar · element mod n.

        Parameters
        ----------
        scalar : int
            Ring element.
        element : sequence of int
            Module element.

        Returns
        -------
        np.ndarray
        """
        return (scalar * np.array(element)) % self._modulus

    def act(
        self,
        element: Sequence[int],
        transposition: int = 0,
        matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply a module endomorphism to an element.

        Parameters
        ----------
        element : sequence of int
            Module element (chord).
        transposition : int
            Global transposition amount.
        matrix : np.ndarray, optional
            k×k matrix over Z/nZ representing a voice reassignment.

        Returns
        -------
        np.ndarray
        """
        result = np.array(element, dtype=int)
        if len(result) != self._rank:
            raise ValueError(f"Expected length {self._rank}")
        if matrix is not None:
            if matrix.shape != (self._rank, self._rank):
                raise ValueError(f"Expected {self._rank}×{self._rank} matrix")
            result = (matrix @ result) % self._modulus
        result = (result + transposition) % self._modulus
        return result

    def voice_leading_matrix(
        self,
        source: Sequence[int],
        target: Sequence[int],
    ) -> np.ndarray:
        """Compute the matrix representation of a voice leading.

        Finds the module endomorphism mapping source → target.
        For a bijection σ, the matrix M has M[σ(i), i] = target[σ(i)] - source[i].

        Parameters
        ----------
        source, target : sequence of int

        Returns
        -------
        np.ndarray
            k×k matrix over Z/nZ.
        """
        from flux_algebra.combinatorics import minimal_voice_leading

        vl = minimal_voice_leading(source, target, self._modulus)

        matrix = np.zeros((self._rank, self._rank), dtype=int)
        for col, (s, t) in enumerate(vl):
            # Find which row this maps to
            row = list(source).index(s)
            matrix[row, col] = (t - s) % self._modulus

        return matrix

    def basis(self) -> List[np.ndarray]:
        """Return the standard basis vectors.

        Returns
        -------
        list of np.ndarray
            [e_1, e_2, ..., e_k] where e_i has 1 in position i and 0 elsewhere.
        """
        return [
            np.array([1 if j == i else 0 for j in range(self._rank)], dtype=int)
            for i in range(self._rank)
        ]

    def voice_leadings_between(
        self,
        source: Sequence[int],
        target: Sequence[int],
    ) -> List[np.ndarray]:
        """List all module endomorphisms mapping source to target.

        Parameters
        ----------
        source, target : sequence of int

        Returns
        -------
        list of np.ndarray
            All k×k matrices that send source → target.
        """
        from flux_algebra.combinatorics import all_voice_leadings

        all_vl = all_voice_leadings(list(source), list(target))
        matrices = []
        for vl in all_vl:
            matrix = np.zeros((self._rank, self._rank), dtype=int)
            for col, (s, t) in enumerate(vl):
                for row in range(self._rank):
                    if source[row] == s:
                        matrix[row, col] = (t - s) % self._modulus
                        break
            matrices.append(matrix)
        return matrices

    def __repr__(self) -> str:
        return f"VoiceModule(rank={self._rank}, modulus={self._modulus})"


