# flux-algebra

Algebraic structures for music theory — rings, fields, groups, tropical semiring, spectral analysis, and voice leading, inspired by Oscar.jl.

## What This Gives You

- **Harmonic rings** — Z/nZ ring arithmetic for pitch-class theory
- **PLR group** — Neo-Riemannian Parallel/Leading-tone/Relative transformations
- **Tropical semiring** — min-plus algebra for voice-leading optimization
- **Tuning fields** — algebraic extensions for equal temperament, just intonation, and microtonal systems
- **Voice leading** — minimal-distance chord transitions with smoothness and efficiency metrics
- **Spectral analysis** — harmonic Laplacian, eigenbasis rings, tonality fingerprints
- **OSCAR-compatible** — designed for integration with algebraic computation systems

## Quick Start

```python
from flux_algebra import (
    HarmonicRing, PLRGroup, Triad, TuningField,
    TropicalHarmony, minimal_voice_leading,
    HarmonicLaplacian, chord_tension
)

# Pitch-class arithmetic in Z/12Z
ring = HarmonicRing(12)
print(ring.add(7, 5))  # 0 (perfect fifth + perfect fourth = octave)

# Neo-Riemannian transformations
plr = PLRGroup()
c_major = Triad(0, 4, 7)
c_minor = plr.parallel(c_major)    # C major → C minor
a_minor = plr.relative(c_major)    # C major → A minor
e_minor = plr.leading_tone(c_major) # C major → E minor

# Minimal voice leading
dist, mapping = minimal_voice_leading([0, 4, 7], [0, 3, 7])
print(f"Cmaj → Cmin: distance={dist}, mapping={mapping}")

# Tropical harmony
trop = TropicalHarmony()
progression = trop.voice_lead([(0,4,7), (5,9,0), (7,11,2)])  # I → IV → V

# Spectral analysis
lap = HarmonicLaplacian.from_chords(["C", "Dm", "Em", "F", "G", "Am"])
tension = chord_tension("Bdim", lap)
print(f"Bdim tension: {tension:.3f}")
```

## API Reference

| Module | Key Types | Description |
|---|---|---|
| `rings` | `HarmonicRing`, `IntervalRing`, `ChordIdeal` | Z/nZ ring structures |
| `fields` | `TuningField`, `AlgebraicTone` | Algebraic tuning systems |
| `groups` | `TranspositionInversionGroup`, `PLRGroup`, `Triad` | Transformation groups |
| `geometry` | `DialPolytope`, `VoiceLeadingGeodesic`, `TraditionRegion` | Geometric voice leading |
| `combinatorics` | `minimal_voice_leading`, `all_voice_leadings` | VL search algorithms |
| `tropical` | `TropicalHarmony`, `TropicalVoiceLeading` | Min-plus music theory |
| `spectral` | `HarmonicLaplacian`, `TonalityFingerprint` | Spectral graph theory |
| `modules` | `VoiceModule` | Module-theoretic voice leading |

## How It Fits

The **algebraic foundation** of the FLUX music ecosystem:

- [flux-algebra-rs](https://github.com/SuperInstance/flux-algebra-rs) — Rust port
- [flux-algebra-c](https://github.com/SuperInstance/flux-algebra-c) — C port
- [flux-tensor-midi](https://github.com/SuperInstance/flux-tensor-midi) — uses algebra for composition
- [constraint-toolkit](https://github.com/SuperInstance/constraint-toolkit) — dial analysis using algebra
- [conservation-spectral-python](https://github.com/SuperInstance/conservation-spectral-python) — spectral Laplacian analysis

## Testing

```bash
pip install -e ".[dev]"
pytest -v  # 10 test files
```

## Installation

```bash
pip install flux-algebra
```

Requires Python ≥ 3.10.

## License

MIT
