"""
Flux Algebra — Algebraic structures for music theory.

Inspired by Oscar.jl, adapted for SuperInstance's music theory/constraint ecosystem.
"""

from flux_algebra.rings import HarmonicRing, IntervalRing, ChordIdeal
from flux_algebra.fields import TuningField, AlgebraicTone
from flux_algebra.groups import TranspositionInversionGroup, PLRGroup, Triad, PermutationVoiceLeading
from flux_algebra.geometry import DialPolytope, VoiceLeadingGeodesic, TraditionRegion
from flux_algebra.combinatorics import minimal_voice_leading, all_voice_leadings, smoothness, efficiency
from flux_algebra.tropical import TropicalHarmony, TropicalVoiceLeading
from flux_algebra.modules import VoiceModule
from flux_algebra.serialization import save, load

__version__ = "0.1.0"

__all__ = [
    "HarmonicRing",
    "IntervalRing",
    "ChordIdeal",
    "TuningField",
    "AlgebraicTone",
    "TranspositionInversionGroup",
    "PLRGroup",
    "Triad",
    "PermutationVoiceLeading",
    "DialPolytope",
    "VoiceLeadingGeodesic",
    "TraditionRegion",
    "minimal_voice_leading",
    "all_voice_leadings",
    "smoothness",
    "efficiency",
    "TropicalHarmony",
    "TropicalVoiceLeading",
    "VoiceModule",
    "save",
    "load",
]
