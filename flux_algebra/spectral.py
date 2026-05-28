"""
Spectral Analysis — Tension-Graph Laplacian and eigenbasis decomposition.

Implements the Eigenbasis Hypothesis for music theory: conservation laws
in music (e.g., tension, voice-leading smoothness) are expressed most
naturally in the eigenbasis of the Tension-Graph Laplacian, not in the
naive measurement basis (e.g., pitch-class coordinates).

Key structures:
- HarmonicLaplacian: builds the Tension-Graph Laplacian from PLR group
  operations, where vertices are triads and edge weights encode both
  transition probability and interval tension (Tenney height).
- EigenbasisHarmonicRing: extends HarmonicRing with eigenvector
  decomposition and conservation-scoring methods.
- TonalityFingerprint: eigenvalue-signature-based corpus classification.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.linalg import eigh

from flux_algebra.groups import PLRGroup, Triad, ALL_TRIADS
from flux_algebra.rings import HarmonicRing


# ═══════════════════════════════════════════════════════════════════════════
#  Part 1: Tenney Height (Tension)
# ═══════════════════════════════════════════════════════════════════════════

def tenney_height(ratio_num: int, ratio_den: int) -> float:
    """Compute the Tenney height of a frequency ratio.

    Tenney height = log₂(numerator × denominator), a measure of
    harmonic complexity. Lower = more consonant.

    For an interval a:b in lowest terms:
      H = log₂(a·b)

    Parameters
    ----------
    ratio_num : int
        Numerator of the frequency ratio (in lowest terms).
    ratio_den : int
        Denominator of the frequency ratio (in lowest terms).

    Returns
    -------
    float
        Tenney height in bits.

    Examples
    --------
    >>> tenney_height(2, 1)   # octave
    1.0
    >>> tenney_height(3, 2)   # perfect fifth
    2.585
    """
    return math.log2(ratio_num * ratio_den)


def semitone_to_ratio(semitones: int) -> Tuple[int, int]:
    """Approximate a semitone interval as a simple just-intonation ratio.

    Used to assign a rough Tenney height to equal-tempered intervals.

    Parameters
    ----------
    semitones : int
        Number of semitones (0-12).

    Returns
    -------
    tuple of (int, int)
        Numerator, denominator of the best simple ratio.
    """
    # Well-known just approximations for equal-tempered intervals
    _APPROX: Dict[int, Tuple[int, int]] = {
        0:  (1, 1),    # unison
        1:  (25, 24),  # minor second
        2:  (9, 8),    # major second
        3:  (6, 5),    # minor third
        4:  (5, 4),    # major third
        5:  (4, 3),    # perfect fourth
        6:  (45, 32),  # tritone
        7:  (3, 2),    # perfect fifth
        8:  (8, 5),    # minor sixth
        9:  (5, 3),    # major sixth
        10: (16, 9),   # minor seventh
        11: (15, 8),   # major seventh
        12: (2, 1),    # octave
    }
    # For exact octave or multiples, return (2^n, 1)
    octave_check = semitones // 12
    remainder = semitones % 12
    if octave_check > 0 and remainder == 0:
        return (2**octave_check, 1)
    if remainder in _APPROX:
        return _APPROX[remainder]
    # Fallback: use the complementary interval
    comp = 12 - remainder
    return _APPROX.get(comp, (comp + 1, 1))


def interval_tension(interval: int, modulus: int = 12) -> float:
    """Compute tension of a pitch-class interval using Tenney height.

    Maps an interval in semitones to its Tenney height via the best
    simple just-intonation approximation. Uses the smaller of the
    ascending and descending forms (interval class).

    Parameters
    ----------
    interval : int
        Interval in semitones (0 to modulus-1).
    modulus : int
        Pitch-class modulus (default 12).

    Returns
    -------
    float
        Tension value.
    """
    ic = min(interval % modulus, modulus - (interval % modulus))
    num, den = semitone_to_ratio(ic)
    return tenney_height(num, den)


def chord_tension(chord_pcs: Sequence[int], modulus: int = 12) -> float:
    """Compute total tension of a chord as sum of pairwise Tenney heights.

    Parameters
    ----------
    chord_pcs : sequence of int
        Pitch classes of the chord.
    modulus : int
        Pitch-class modulus.

    Returns
    -------
    float
        Sum of Tenney heights for all dyads in the chord.
    """
    pcs = list(chord_pcs)
    if len(pcs) < 2:
        return 0.0
    total = 0.0
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):
            interval = abs(pcs[i] - pcs[j]) % modulus
            total += interval_tension(interval, modulus)
    return total


# ═══════════════════════════════════════════════════════════════════════════
#  Part 2: HarmonicLaplacian — Tension-Graph Laplacian
# ═══════════════════════════════════════════════════════════════════════════

class HarmonicLaplacian:
    """Tension-Graph Laplacian built from PLR group operations.

    Constructs a weighted graph where:
      - Vertices = the 24 major/minor triads
      - Edge weight W[i,j] = P(i → j) × exp(-tension(i, j) / σ)
      - Transition probability P(i → j) reflects the PLR group distance
      - L = D - W (standard graph Laplacian)

    The eigenvectors of L define the eigenbasis in which tension
    conservation is expected to hold best.

    Parameters
    ----------
    sigma : float
        Temperature parameter controlling the exp(-tension/σ) weighting.
        Lower σ amplifies tension differences (default 1.0).
    modulus : int
        Pitch-class modulus (default 12).

    Examples
    --------
    >>> hl = HarmonicLaplacian()
    >>> L = hl.laplacian()
    >>> L.shape
    (24, 24)
    >>> eigvals = hl.eigenvalues[:5]  # smallest eigenvalues
    >>> len(eigvals)
    5
    """

    def __init__(self, sigma: float = 1.0, modulus: int = 12) -> None:
        self._sigma = sigma
        self._modulus = modulus
        self._plr = PLRGroup(modulus=modulus)
        self._triads = tuple(ALL_TRIADS[:])  # all 24 triads
        self._index: Dict[Triad, int] = {t: i for i, t in enumerate(self._triads)}
        self._L: Optional[np.ndarray] = None
        self._evals: Optional[np.ndarray] = None
        self._evecs: Optional[np.ndarray] = None

    # ── Graph Construction ────────────────────────────────────────────────

    @property
    def triads(self) -> Tuple[Triad, ...]:
        """All vertices of the graph (24 major/minor triads)."""
        return self._triads

    def triad_index(self, triad: Triad) -> int:
        """Get the graph vertex index for a triad.

        Parameters
        ----------
        triad : Triad

        Returns
        -------
        int
        """
        return self._index[triad]

    def _plr_distance(self, a: Triad, b: Triad) -> int:
        """Compute PLR group word distance between two triads.

        Returns the minimal number of P/L/R operations needed to
        transform one triad into another. Uses BFS on the PLR
        Cayley graph (diameter ≤ 4 for 24 triads under P,L,R).

        Parameters
        ----------
        a, b : Triad

        Returns
        -------
        int
            Minimum number of PLR steps.
        """
        if a == b:
            return 0
        visited = {a: 0}
        queue = [a]
        while queue:
            current = queue.pop(0)
            dist = visited[current]
            for op in (self._plr.P, self._plr.L, self._plr.R):
                neighbor = op(current)
                if neighbor == b:
                    return dist + 1
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append(neighbor)
        return self._max_dist()  # fallback

    def _max_dist(self) -> int:
        """Compute the diameter of the PLR Cayley graph."""
        max_d = 0
        for a in self._triads:
            for b in self._triads:
                d = self._plr_distance(a, b) if hasattr(self, '_plr_distance') else 0
                if d > max_d:
                    max_d = d
        return max_d

    def _transition_probability(self, a: Triad, b: Triad) -> float:
        """Transition probability P(i → j) based on PLR distance.

        Probability decreases exponentially with PLR distance:
          P(i → j) ∝ exp(-d(i,j) / τ)
        where τ = 1.0 (characteristic PLR step length).

        Parameters
        ----------
        a, b : Triad

        Returns
        -------
        float
            Unnormalized transition weight.
        """
        d = self._plr_distance(a, b)
        tau = 1.0
        return math.exp(-d / tau)

    def _chord_pcs(self, triad: Triad) -> Tuple[int, ...]:
        """Get pitch classes of a triad."""
        return triad.pitch_classes

    def _tension_between(self, a: Triad, b: Triad) -> float:
        """Tension between two triads: sum of Tenney heights of intervals
        between each voice's movement. This captures the harmonic friction
        of transitioning from chord a to chord b.

        Uses the minimal voice-leading assignment to pair pitches.

        Parameters
        ----------
        a, b : Triad

        Returns
        -------
        float
            Total interval tension of the voice-leading pairs.
        """
        pcs_a = self._chord_pcs(a)
        pcs_b = self._chord_pcs(b)

        # Minimal voice-leading assignment (Hungarian algorithm)
        from flux_algebra.combinatorics import minimal_voice_leading
        vl = minimal_voice_leading(pcs_a, pcs_b, modulus=self._modulus)

        total = 0.0
        for src, tgt in vl:
            interval = abs(src - tgt) % self._modulus
            total += interval_tension(interval, self._modulus)
        return total

    def adjacency(self, normalize: bool = True) -> np.ndarray:
        """Build the weighted adjacency matrix W.

        W[i,j] = P(i → j) × exp(-tension(i,j) / σ)

        Parameters
        ----------
        normalize : bool
            If True, row-normalize so each row sums to 1.

        Returns
        -------
        np.ndarray
            (24, 24) weighted adjacency matrix.
        """
        n = len(self._triads)
        W = np.zeros((n, n))
        for i, a in enumerate(self._triads):
            for j, b in enumerate(self._triads):
                if i == j:
                    continue
                prob = self._transition_probability(a, b)
                tension = self._tension_between(a, b)
                W[i, j] = prob * math.exp(-tension / self._sigma)

        if normalize:
            row_sums = W.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            W = W / row_sums

        return W

    def laplacian(self) -> np.ndarray:
        """Compute the graph Laplacian L = D - W.

        Returns
        -------
        np.ndarray
            (24, 24) symmetric graph Laplacian.
        """
        if self._L is not None:
            return self._L

        W = self.adjacency(normalize=True)
        D = np.diag(W.sum(axis=1))
        L = D - W
        # Symmetrize for stability (the row-normalized W may not yield
        # a perfectly symmetric L; take (L + L^T)/2)
        L = (L + L.T) / 2.0
        self._L = L
        return L

    # ── Eigendecomposition ────────────────────────────────────────────────

    def _decompose(self) -> None:
        """Compute eigenvalues and eigenvectors of the Laplacian."""
        if self._evals is not None and self._evecs is not None:
            return
        L = self.laplacian()
        evals, evecs = eigh(L)
        # Sort by eigenvalue ascending
        idx = np.argsort(evals)
        self._evals = evals[idx]
        self._evecs = evecs[:, idx]

    @property
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the Laplacian, sorted ascending.

        Returns
        -------
        np.ndarray
            Shape (24,) array of eigenvalues λ_0 ≤ λ_1 ≤ ... ≤ λ_23.
        """
        self._decompose()
        assert self._evals is not None
        return self._evals

    @property
    def eigenvectors(self) -> np.ndarray:
        """Eigenvectors of the Laplacian, sorted by eigenvalue ascending.

        Returns
        -------
        np.ndarray
            Shape (24, 24) matrix where column k is the eigenvector
            for eigenvalue λ_k.
        """
        self._decompose()
        assert self._evecs is not None
        return self._evecs

    def eigengap_ratio(self, k: int = 1) -> float:
        """Ratio of successive eigengaps: (λ_{k+1} - λ_k) / (λ_k - λ_{k-1}).

        Large values indicate a natural cluster count of k.

        Parameters
        ----------
        k : int
            Index of the gap (default 1 = between λ_1 and λ_2).

        Returns
        -------
        float
        """
        evals = self.eigenvalues
        if k < 1 or k >= len(evals) - 1:
            return 0.0
        return (evals[k + 1] - evals[k]) / (evals[k] - evals[k - 1])

    def spectral_gap(self) -> float:
        """The spectral gap λ_1 - λ_0: connectivity measure.

        A larger gap means the graph is more strongly connected.

        Returns
        -------
        float
        """
        return float(self.eigenvalues[1] - self.eigenvalues[0])

    def algebraic_connectivity(self) -> float:
        """Fiedler value λ_1: the algebraic connectivity.

        A measure of how well-connected the graph is. Larger = more
        robust connectivity.

        Returns
        -------
        float
        """
        return float(self.eigenvalues[1])

    # ── Projection ────────────────────────────────────────────────────────

    def project_triad_vector(self, values: Dict[Triad, float]) -> np.ndarray:
        """Project a triad-valued function onto the Laplacian eigenvectors.

        Parameters
        ----------
        values : dict mapping Triad → float
            A function defined on the 24 triads.

        Returns
        -------
        np.ndarray
            Coefficient vector in the eigenbasis (one coefficient per
            eigenvector, sorted by eigenvalue ascending).
        """
        vec = np.zeros(len(self._triads))
        for triad, val in values.items():
            vec[self.triad_index(triad)] = val
        return self.eigenvectors.T @ vec

    def reconstruct_from_eigenbasis(
        self, coefficients: np.ndarray, k: Optional[int] = None
    ) -> np.ndarray:
        """Reconstruct a triad function from its eigenbasis coefficients,
        optionally truncated to the first k components.

        Parameters
        ----------
        coefficients : np.ndarray
            Coefficients in the eigenbasis.
        k : int, optional
            Number of eigenvectors to use. If None, uses all.

        Returns
        -------
        np.ndarray
            Reconstructed function value at each triad.
        """
        if k is None:
            k = len(self._triads)
        return self.eigenvectors[:, :k] @ coefficients[:k]

    def __repr__(self) -> str:
        return f"HarmonicLaplacian(sigma={self._sigma}, modulus={self._modulus})"


# ═══════════════════════════════════════════════════════════════════════════
#  Part 3: EigenbasisHarmonicRing — Conservation-Aware Harmonic Analysis
# ═══════════════════════════════════════════════════════════════════════════

class EigenbasisHarmonicRing(HarmonicRing):
    """Extends HarmonicRing with eigenvector decomposition of the
    Tension-Graph Laplacian for conservation-based analysis.

    Key methods:
      - conservation_score(chord_sequence): measures how well a sequence
        respects the eigenbasis conservation law.
      - optimal_voice_leading(from, to): finds the voice leading with
        minimal conservation violation.
      - detect_modulation(sequence): sliding window conservation drops.

    Parameters
    ----------
    modulus : int
        Number of pitch divisions (default 12).
    sigma : float
        Tension-temperature parameter for the Laplacian (default 1.0).

    Examples
    --------
    >>> ehr = EigenbasisHarmonicRing()
    >>> seq = [Triad(0, "major"), Triad(0, "minor"), Triad(7, "major")]
    >>> score = ehr.conservation_score(seq)
    >>> isinstance(score, float)
    True
    """

    def __init__(self, modulus: int = 12, sigma: float = 1.0) -> None:
        super().__init__(modulus=modulus)
        self._laplacian = HarmonicLaplacian(sigma=sigma, modulus=modulus)
        self._sigma = sigma

    @property
    def laplacian(self) -> HarmonicLaplacian:
        """The underlying HarmonicLaplacian instance."""
        return self._laplacian

    # ── Conservation Score ────────────────────────────────────────────────

    def _triad_to_vector(self, triad: Triad) -> np.ndarray:
        """Map a triad to its eigenbasis coefficient vector.

        Projects the indicator function (1 at this triad, 0 elsewhere)
        onto the Laplacian eigenvectors.

        Parameters
        ----------
        triad : Triad

        Returns
        -------
        np.ndarray
            Coefficient vector in the eigenbasis.
        """
        indicator = {t: 0.0 for t in self._laplacian.triads}
        indicator[triad] = 1.0
        return self._laplacian.project_triad_vector(indicator)

    def conservation_score(
        self, chord_sequence: Sequence[Triad],
        k: Optional[int] = None,
    ) -> float:
        """Compute conservation score for a chord sequence.

        Projects each chord onto the first k Laplacian eigenvectors
        (low-frequency components) and computes the gradient variance.
        When k is small, this measures how much the sequence moves in
        the most "structurally conserved" directions of harmonic space.

        A LOW score means the sequence moves primarily in directions
        that are naturally conserved (aligned with the eigenbasis).
        A HIGH score means the sequence jumps in high-frequency,
        non-conserved directions.

        Parameters
        ----------
        chord_sequence : sequence of Triad
            Sequence of triads to evaluate.
        k : int, optional
            Number of top eigenvectors to consider. If None, uses all
            weighted by inverse eigenvalue (so low-frequency components
            contribute more). Default None — eigenvalue-weighted.

        Returns
        -------
        float
            Conservation violation score (lower = more conserved).
        """
        if len(chord_sequence) < 2:
            return 0.0

        evals = self._laplacian.eigenvalues
        evecs = self._laplacian.eigenvectors
        triads = self._laplacian.triads

        # Project each triad onto eigenvectors
        projections = []
        for chord in chord_sequence:
            vec = np.zeros(len(triads))
            vec[self._laplacian.triad_index(chord)] = 1.0
            coeffs = evecs.T @ vec
            projections.append(coeffs)

        # Gradient variance with weighting
        total_variance = 0.0
        for i in range(len(projections) - 1):
            diff = projections[i + 1] - projections[i]

            if k is not None:
                # Truncate to first k components
                total_variance += float(np.dot(diff[:k], diff[:k]))
            else:
                # Weight by inverse eigenvalue: low-freq components
                # (small λ) are more conserved, so changes there cost less
                # High-freq components (large λ) cost more
                weights = 1.0 / (evals + 1.0)  # avoid /0 with +1
                weighted_diff = diff * weights
                total_variance += float(np.dot(weighted_diff, weighted_diff))

        return total_variance / (len(chord_sequence) - 1)

    # ── Optimal Voice Leading ─────────────────────────────────────────────

    def optimal_voice_leading(
        self,
        source: Triad,
        target: Triad,
        alpha: float = 0.5,
    ) -> Tuple[Tuple[int, ...], float]:
        """Find the voice leading between two triads that minimizes
        a combined cost: (1-α) × voice-leading smoothness + α ×
        conservation violation.

        The conservation violation is measured by how much the
        eigenbasis projection changes (the eigenvector-weighted
        squared difference between source and target).

        Parameters
        ----------
        source : Triad
            Source triad.
        target : Triad
            Target triad.
        alpha : float
            Weight for conservation violation vs smoothness
            (0 = pure smoothness, 1 = pure conservation).

        Returns
        -------
        tuple of (tuple, float)
            (optimal_permutation_of_target_pcs, total_cost)
        """
        from flux_algebra.combinatorics import all_voice_leadings, smoothness

        src_pcs = source.pitch_classes
        tgt_pcs = target.pitch_classes

        # Project both triads onto eigenbasis
        src_coeffs = self._laplacian.project_triad_vector(
            {t: 1.0 if t == source else 0.0 for t in self._laplacian.triads}
        )
        tgt_coeffs = self._laplacian.project_triad_vector(
            {t: 1.0 if t == target else 0.0 for t in self._laplacian.triads}
        )

        # Conservation cost: eigenvalue-weighted projection change
        evals = self._laplacian.eigenvalues + 1.0  # avoid /0
        weights = 1.0 / evals
        diff = tgt_coeffs - src_coeffs
        cons_cost = float(np.dot(diff * weights, diff * weights))

        best_cost = float("inf")
        best_vl: Optional[Tuple[int, ...]] = None

        for vl in all_voice_leadings(src_pcs, tgt_pcs):
            sm = smoothness(vl, self._modulus)
            rearranged = tuple(t for _, t in vl)
            cost = (1 - alpha) * sm + alpha * cons_cost
            if cost < best_cost:
                best_cost = cost
                best_vl = rearranged

        if best_vl is None:
            best_vl = target.pitch_classes

        return (best_vl, best_cost)

    # ── Modulation Detection ──────────────────────────────────────────────

    def detect_modulation(
        self,
        sequence: Sequence[Triad],
        window: int = 8,
    ) -> List[Tuple[int, float]]:
        """Detect modulations in a chord sequence by sliding-window
        conservation scoring. Drops in the conservation score indicate
        key changes.

        Parameters
        ----------
        sequence : sequence of Triad
            Full chord sequence.
        window : int
            Sliding window size (default 8 chords).

        Returns
        -------
        list of (int, float)
            List of (position, local_conservation_score) for each
            window start. Modulation boundaries correspond to positions
            where the score changes sharply.
        """
        if len(sequence) < window:
            return [(0, self.conservation_score(sequence))]

        results: List[Tuple[int, float]] = []
        for i in range(len(sequence) - window + 1):
            window_seq = sequence[i : i + window]
            score = self.conservation_score(window_seq)
            results.append((i, score))

        return results

    # ── Tension Correlation Matrix ────────────────────────────────────────

    def tension_correlation_matrix(
        self,
        sequences: Sequence[Sequence[Triad]],
    ) -> np.ndarray:
        """Compute the 3×3 correlation matrix of three tension metrics
        across a set of chord sequences.

        The three metrics are:
          1. Spectral tension (pairwise Tenney height sum)
          2. Voice-leading smoothness (total semitone movement)
          3. Conservation score (eigenbasis gradient variance)

        The eigenvectors of this matrix define the rotated basis where
        tension conservation is most naturally expressed (per the
        Eigenbasis Hypothesis, Experiment 4).

        Parameters
        ----------
        sequences : sequence of sequence of Triad
            Collection of chord sequences.

        Returns
        -------
        np.ndarray
            3×3 correlation matrix: [spectral, smoothness, conservation].
        """
        metrics: List[List[float]] = [[], [], []]

        for seq in sequences:
            if len(seq) < 2:
                continue

            # Spectral tension
            for chord in seq:
                metrics[0].append(chord_tension(chord.pitch_classes, self._modulus))

            # Voice-leading smoothness between consecutive chords
            from flux_algebra.combinatorics import minimal_voice_leading, smoothness
            for i in range(len(seq) - 1):
                vl = minimal_voice_leading(
                    seq[i].pitch_classes, seq[i + 1].pitch_classes, self._modulus
                )
                metrics[1].append(float(smoothness(vl, self._modulus)))

            # Conservation score
            metrics[2].append(self.conservation_score(seq))

        # Pad to equal lengths
        min_len = min(len(m) for m in metrics)
        data = np.array([m[:min_len] for m in metrics])

        return np.corrcoef(data)

    def __repr__(self) -> str:
        return f"EigenbasisHarmonicRing(modulus={self._modulus}, sigma={self._sigma})"


# ═══════════════════════════════════════════════════════════════════════════
#  Part 4: TonalityFingerprint — Corpus Classification via Eigenvalues
# ═══════════════════════════════════════════════════════════════════════════

class TonalityFingerprint:
    """Eigenvalue signature of a chord corpus.

    Given a corpus of chords (sequence of Triads), builds the
    Tension-Graph Laplacian and returns the eigenvalue signature.
    This signature can be used to compare musical traditions,
    identify historical periods, and classify corpora.

    Parameters
    ----------
    sigma : float
        Tension temperature (default 1.0).
    modulus : int
        Pitch-class modulus (default 12).

    Examples
    --------
    >>> tf = TonalityFingerprint()
    >>> corpus = [Triad(0, "major"), Triad(0, "minor"), Triad(7, "major")]
    >>> sig = tf.fingerprint(corpus)
    >>> len(sig) == 24
    True
    """

    def __init__(self, sigma: float = 1.0, modulus: int = 12) -> None:
        self._sigma = sigma
        self._modulus = modulus
        self._laplacian = HarmonicLaplacian(sigma=sigma, modulus=modulus)

    def fingerprint(self, corpus: Sequence[Triad]) -> np.ndarray:
        """Compute the eigenvalue signature of a chord corpus.

        The Laplacian is built from the PLR graph (always 24 vertices);
        the corpus is used to weight the adjacency matrix by transition
        frequencies observed in the corpus. Returns the 24 eigenvalues
        sorted ascending.

        Parameters
        ----------
        corpus : sequence of Triad
            Chords in the corpus (in sequence order).

        Returns
        -------
        np.ndarray
            Shape (24,) eigenvalue array sorted ascending.
        """
        if len(corpus) < 2:
            return self._laplacian.eigenvalues.copy()

        # Count observed transitions in the corpus
        transition_counts: Dict[Tuple[Triad, Triad], int] = Counter()
        for i in range(len(corpus) - 1):
            a, b = corpus[i], corpus[i + 1]
            transition_counts[(a, b)] += 1

        # Build corpus-weighted adjacency
        n = len(self._laplacian.triads)
        W = np.zeros((n, n))
        for (a, b), count in transition_counts.items():
            i = self._laplacian.triad_index(a)
            j = self._laplacian.triad_index(b)
            tension = self._laplacian._tension_between(a, b)
            W[i, j] += count * math.exp(-tension / self._sigma)

        # Symmetrize before normalization to preserve PSD property
        W = (W + W.T) / 2.0

        # Row-normalize (degree-normalized)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W = W / row_sums

        # Laplacian L = D - W  (with symmetric correction)
        D = np.diag(W.sum(axis=1))
        L = D - W
        L = (L + L.T) / 2.0

        # Eigendecomposition (use np.linalg.eigh for symmetric matrices)
        evals, _ = eigh(L)
        evals = np.clip(np.sort(evals), 0.0, None)  # clamp numerical noise

        return evals

    @staticmethod
    def compare_corpora(
        corpus_a: Sequence[Triad],
        corpus_b: Sequence[Triad],
        sigma: float = 1.0,
        modulus: int = 12,
    ) -> float:
        """Cosine similarity between eigenvalue signatures of two corpora.

        Higher similarity means the two corpora have similar harmonic
        structure.

        Parameters
        ----------
        corpus_a, corpus_b : sequence of Triad
            Two chord corpora to compare.
        sigma : float
            Tension temperature.
        modulus : int
            Pitch-class modulus.

        Returns
        -------
        float
            Cosine similarity between eigenvalue vectors (0 to 1).
        """
        tf_a = TonalityFingerprint(sigma=sigma, modulus=modulus)
        tf_b = TonalityFingerprint(sigma=sigma, modulus=modulus)
        evals_a = tf_a.fingerprint(corpus_a)
        evals_b = tf_b.fingerprint(corpus_b)

        # Cosine similarity
        norm_a = np.linalg.norm(evals_a)
        norm_b = np.linalg.norm(evals_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(evals_a, evals_b) / (norm_a * norm_b))

    @staticmethod
    def identify_period(
        corpus: Sequence[Triad],
        reference_corpora: Dict[str, Sequence[Triad]],
        sigma: float = 1.0,
        modulus: int = 12,
    ) -> Tuple[str, float]:
        """Classify a corpus by identifying its closest historical period.

        Compares the corpus fingerprint against reference corpora
        from known periods and returns the best match.

        Parameters
        ----------
        corpus : sequence of Triad
            Corpus to classify.
        reference_corpora : dict of str → sequence of Triad
            Reference corpora keyed by period/name.
        sigma : float
            Tension temperature.
        modulus : int
            Pitch-class modulus.

        Returns
        -------
        tuple of (str, float)
            (best_period_name, similarity_score).
        """
        best_period = ""
        best_score = -1.0

        for period, ref_corpus in reference_corpora.items():
            score = TonalityFingerprint.compare_corpora(
                corpus, ref_corpus, sigma=sigma, modulus=modulus
            )
            if score > best_score:
                best_score = score
                best_period = period

        return (best_period, best_score)


# ═══════════════════════════════════════════════════════════════════════════
#  Part 5: Convenience functions
# ═══════════════════════════════════════════════════════════════════════════

def build_common_practice_walk() -> List[Triad]:
    """Build a common-practice chord walk: I-IV-V-I-I-V-I-V-I.

    Represents the harmonic skeleton of a typical common-practice
    period progression in C major.

    Returns
    -------
    list of Triad
    """
    return [
        Triad(0, "major"),   # I:  C major
        Triad(5, "major"),   # IV: F major
        Triad(7, "major"),   # V:  G major
        Triad(0, "major"),   # I:  C major
        Triad(7, "major"),   # V:  G major
        Triad(0, "major"),   # I:  C major
        Triad(5, "major"),   # IV: F major
        Triad(7, "major"),   # V:  G major
        Triad(0, "major"),   # I:  C major
    ]


def build_chromatic_walk() -> List[Triad]:
    """Build a chromatic walk: C → Cm → Ab → Abm → E → Em → C.

    Each step is one PLR operation, exploring the chromatic-third relation.

    Returns
    -------
    list of Triad
    """
    return [
        Triad(0, "major"),   # C+
        Triad(0, "minor"),   # C-  (P)
        Triad(8, "major"),   # Ab+ (R from C-)
        Triad(8, "minor"),   # Ab- (P)
        Triad(4, "major"),   # E+  (R from Ab-)
        Triad(4, "minor"),   # E-  (P)
        Triad(0, "major"),   # C+  (R from E-)
    ]


def build_neapolitan_walk() -> List[Triad]:
    """Build a Neapolitan-like walk: C → Db → C → Fm → G → C.

    Includes a Neapolitan sixth (Db major) and modal mixture.

    Returns
    -------
    list of Triad
    """
    return [
        Triad(0, "major"),   # C+
        Triad(1, "major"),   # Db+  (Neapolitan)
        Triad(0, "major"),   # C+
        Triad(5, "minor"),   # F-   (iv from minor)
        Triad(7, "major"),   # G+   (V)
        Triad(0, "major"),   # C+
    ]
