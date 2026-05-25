"""
Voice Leading Combinatorics — Minimal voice leading between chords.

Analogous to Oscar.jl's Combinatorics module.

Uses the Hungarian algorithm for optimal voice-leading assignment and
provides metrics (smoothness, efficiency) for comparing voice leadings.
"""

from __future__ import annotations

from itertools import permutations
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def minimal_voice_leading(
    source: Sequence[int],
    target: Sequence[int],
    modulus: int = 12,
) -> List[Tuple[int, int]]:
    """Find minimal (smoothest) voice leading between two chords.

    Uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment) to
    find the bijection from source to target voices minimizing total
    semitone movement, respecting pitch-class equivalence.

    Parameters
    ----------
    source : sequence of int
        Source chord pitch classes (or MIDI notes).
    target : sequence of int
        Target chord pitch classes (or MIDI notes).
    modulus : int
        Pitch-class modulus (default 12).

    Returns
    -------
    list of (int, int)
        Voice-leading pairs (source_voice, target_voice).

    Examples
    --------
    >>> minimal_voice_leading([0, 4, 7], [5, 9, 0])
    [(0, 0), (4, 5), (7, 9)]
    """
    source = list(source)
    target = list(target)
    n = len(source)
    if n != len(target):
        raise ValueError(
            f"Source and target must have same length, got {len(source)} vs {len(target)}"
        )
    if n == 0:
        return []

    # Cost matrix: cost[i][j] = minimal distance source[i] → target[j]
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = abs(source[i] - target[j])
            cost[i][j] = min(diff, modulus - diff)

    row_ind, col_ind = linear_sum_assignment(cost)

    return [(source[row_ind[i]], target[col_ind[i]]) for i in range(n)]


def all_voice_leadings(
    source: Sequence[int],
    target: Sequence[int],
) -> List[List[Tuple[int, int]]]:
    """Enumerate all bijective voice leadings between two chords.

    Generates all permutations of the target assignment. For n voices,
    this yields n! voice leadings.

    Parameters
    ----------
    source : sequence of int
        Source chord.
    target : sequence of int
        Target chord.

    Returns
    -------
    list of list of (int, int)
        All voice leadings, each as a list of (source, target) pairs.

    Examples
    --------
    >>> vls = all_voice_leadings([0, 4, 7], [5, 9, 0])
    >>> len(vls)  # 3! = 6
    6
    """
    source = list(source)
    target = list(target)
    n = len(source)
    if n != len(target):
        raise ValueError("Source and target must have same length")

    result = []
    for perm in permutations(range(n)):
        vl = [(source[i], target[perm[i]]) for i in range(n)]
        result.append(vl)
    return result


def smoothness(voice_leading: List[Tuple[int, int]], modulus: int = 12) -> int:
    """Sum of absolute interval movements (voice-leading size).

    The smoothness of a voice leading is the total semitone distance
    traversed by all voices. Lower is smoother.

    Parameters
    ----------
    voice_leading : list of (int, int)
        Voice-leading pairs.
    modulus : int
        Pitch-class modulus (default 12).

    Returns
    -------
    int
        Total voice-leading distance.

    Examples
    --------
    >>> smoothness([(0, 0), (4, 5), (7, 9)])
    5
    """
    total = 0
    for s, t in voice_leading:
        diff = abs(s - t)
        total += min(diff, modulus - diff)
    return total


def efficiency(voice_leading: List[Tuple[int, int]], modulus: int = 12) -> float:
    """Voice-leading efficiency: smoothness / voices moved.

    Efficiency measures how economical the voice leading is. A value of 0
    means no voices moved (same chord). Lower is better.

    Parameters
    ----------
    voice_leading : list of (int, int)
        Voice-leading pairs.
    modulus : int
        Pitch-class modulus.

    Returns
    -------
    float
        Smoothness divided by number of voices that actually moved.

    Examples
    --------
    >>> efficiency([(0, 0), (4, 5), (7, 9)])
    2.5
    """
    voices_moved = sum(1 for s, t in voice_leading if s != t)
    if voices_moved == 0:
        return 0.0
    return smoothness(voice_leading, modulus) / voices_moved


def voice_leading_distance(
    chord_a: Sequence[int],
    chord_b: Sequence[int],
    modulus: int = 12,
) -> int:
    """Minimal voice-leading distance between two chords.

    Computes the smoothest voice leading and returns its distance.

    Parameters
    ----------
    chord_a, chord_b : sequence of int
        Pitch classes of the two chords.
    modulus : int
        Pitch-class modulus.

    Returns
    -------
    int
        Minimal total semitone movement.
    """
    vl = minimal_voice_leading(chord_a, chord_b, modulus)
    return smoothness(vl, modulus)


def sort_by_smoothness(
    chord: Sequence[int],
    targets: Sequence[Sequence[int]],
    modulus: int = 12,
) -> List[Tuple[int, Sequence[int]]]:
    """Sort target chords by voice-leading smoothness from a source chord.

    Parameters
    ----------
    chord : sequence of int
        Source chord.
    targets : sequence of sequence of int
        Target chords to rank.
    modulus : int
        Pitch-class modulus.

    Returns
    -------
    list of (int, sequence)
        Targets sorted by smoothness, as (distance, chord) pairs.
    """
    scored = []
    for target in targets:
        vl = minimal_voice_leading(chord, target, modulus)
        d = smoothness(vl, modulus)
        scored.append((d, target))
    return sorted(scored, key=lambda x: x[0])
