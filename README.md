# flux-algebra

**Algebraic structures for music theory** — harmonic rings, tuning fields, symmetry groups, tropical semiring, spectral analysis, and voice leading. Inspired by [Oscar.jl](https://www.oscar-system.org/), built for Python.

## What This Does

Flux-algebra treats music theory as applied abstract algebra. Pitch classes form the ring Z/12Z, chord progressions are paths through group orbits, voice leading is a linear optimisation problem over free modules, and tonal tension is the spectral decomposition of a graph Laplacian. If that sounds like overkill for writing a chord progression — it is, and that's the point.

The library gives you:

- **Harmonic rings** — Z/nZ ring arithmetic for pitch-class theory, with ideals that correspond to real musical objects (tritone pairs, augmented triads, diminished seventh chords, whole-tone scales)
- **Tuning fields** — algebraic extensions of Q encoding frequency ratios for equal temperament, just intonation, meantone, and microtonal systems
- **PLR group** — the neo-Riemannian Parallel / Leading-tone / Relative transformation group acting transitively on the 24 major/minor triads
- **T / I group** — the full transposition-inversion group of order 24 with orbit/stabilizer computations
- **Tropical semiring** — min-plus algebra applied to voice-leading optimisation and harmonic tension measurement
- **Spectral analysis** — harmonic graph Laplacian, eigenbasis decomposition, Tenney-height-weighted tension, tonality fingerprints
- **Voice modules** — free modules over Z/nZ with homomorphisms (matrices) representing voice leadings
- **Voice leading** — Hungarian-algorithm optimal assignment, smoothness/efficiency metrics, all-voice-leading enumeration
- **Polyhedral geometry** — convex hulls and Voronoi diagrams over a 3D "dial space" of musical traditions
- **Oscar.jl compatibility layer** — Julia-style API surface for cross-language interop
- **JSON serialization** — save and load every algebraic object

## Key Idea

The divisors of 12 determine the ideal structure of Z/12Z, and every ideal is a recognisable musical object:

| Ideal (d | 12) | Set | Musical name |
|---|---|---|
| d = 1 | {0} | Unison |
| d = 2 | {0, 6} | Tritone pair |
| d = 3 | {0, 4, 8} | Augmented triad |
| d = 4 | {0, 3, 6, 9} | Diminished seventh chord |
| d = 6 | {0, 2, 4, 6, 8, 10} | Whole-tone scale |
| d = 12 | Z/12Z | Chromatic scale |

The PLR group (order 24) acts transitively on all 24 major/minor triads — any triad can reach any other through a word in {P, L, R}. The tropical semiring (min-plus) provides a natural distance metric for voice leading that avoids the cyclic discontinuity of standard semitone distance.

## Install

```bash
pip install flux-algebra
```

Requires Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10.

## Quick Start

### Ring arithmetic

```python
from flux_algebra import HarmonicRing

hr = HarmonicRing(12)
print(hr.add(7, 5))       # 0   — G + E = C (mod 12)
print(hr.multiply(3, 4))  # 0   — M3 × M3 = 0 (wraps)

# Ideals = musical objects
for d in hr.ideals():
    print(f"d={d.generator}: {d.elements}")
    # d=1: {0}           — unison
    # d=2: {0, 6}        — tritone pair
    # d=3: {0, 4, 8}     — augmented triad
    # d=4: {0, 3, 6, 9}  — diminished seventh
    # d=6: {0, 2, 4, 6, 8, 10} — whole-tone scale
```

### PLR group

```python
from flux_algebra import PLRGroup, Triad

plr = PLRGroup()
c_maj = Triad(0, "major")

# P = Parallel: flip mode, keep root
c_min = plr.P(c_maj)         # Triad(0, "minor") — C minor

# L = Leading-tone: swap third
e_min = plr.L(c_maj)         # Triad(4, "minor") — E minor

# R = Relative: relative major/minor
a_min = plr.R(c_maj)         # Triad(9, "minor") — A minor

# The PLR group acts transitively on all 24 triads
orbit = plr.orbit(c_maj)
print(len(orbit))            # 24

# Common tones between transformations
ct = plr.common_tones(c_maj, c_min)
print(ct)                    # {0, 7} — root and fifth shared
```

### Tropical voice leading

```python
from flux_algebra import TropicalHarmony

th = TropicalHarmony()

# Chord cost function: semitone distance from nearest chord tone
cost = th.chord_cost([0, 4, 7])  # C major
print(cost(0))   # 0.0 — C is in the chord
print(cost(3))   # 1.0 — E♭ is 1 semitone from E

# Harmonic tension against a scale
tension = th.harmonic_tension([1, 5, 8], [0, 2, 4, 5, 7, 9, 11])
print(tension)   # > 0 — non-scale tones create tension

# Tropical voice leading
vl = th.tropical_voice_leading([0, 4, 7], [5, 9, 0])
print(vl)  # [(0,0), (4,5), (7,9)] — minimal movement
```

### Spectral analysis

```python
from flux_algebra import HarmonicLaplacian, tenney_height, chord_tension

# Tenney height: measures harmonic complexity
print(tenney_height(2, 1))   # 1.0  — octave (simple)
print(tenney_height(3, 2))   # ~1.58 — perfect fifth
print(tenney_height(45, 32)) # ~3.83 — tritone (complex)

# Build a tension-weighted Laplacian over the PLR graph
lap = HarmonicLaplacian()
eigenvalues = lap.eigenvalues()  # Sorted tension modes
fingerprint = lap.tonality_fingerprint("bach_chorales")
```

### Voice leading (Hungarian algorithm)

```python
from flux_algebra import minimal_voice_leading, smoothness, efficiency

# Optimal voice assignment from C major to F major
vl = minimal_voice_leading([0, 4, 7], [5, 9, 0])
print(vl)          # [(0, 0), (4, 5), (7, 9)]
print(smoothness(vl))   # total semitone movement
print(efficiency(vl))   # ratio of direct to actual movement
```

### Tuning fields

```python
from flux_algebra import TuningField

et12 = TuningField.ET(12)            # 12-tone equal temperament
just5 = TuningField.just(primes=(2, 3, 5))  # 5-limit just intonation

print(et12.degree)        # 12 — [Q(ζ₁₂) : Q]
print(just5.primes)        # (2, 3, 5)
print(et12.ratio(7))       # frequency ratio for the 7th semitone

# Compare tunings for the same interval
print(et12.cents(7))       # 700.0 — ET fifth
print(just5.cents_for_ratio(3, 2))  # 701.96 — just fifth
```

## API Reference

### `flux_algebra.rings` — Harmonic Rings

| Class / Function | Description |
|---|---|
| `HarmonicRing(modulus=12)` | Ring Z/nZ of pitch classes |
| `HarmonicRing.add(a, b)` | Ring addition (transposition) |
| `HarmonicRing.multiply(a, b)` | Ring multiplication (interval composition) |
| `HarmonicRing.ideals()` | All ideals of Z/nZ (musical objects) |
| `IntervalRing(prime_limit=(2,3,5))` | Ring of p-limit just intonation intervals |
| `ChordIdeal(generator, ring)` | Ideal corresponding to a chord/scale |

### `flux_algebra.fields` — Tuning Fields

| Class / Function | Description |
|---|---|
| `TuningField(name, generator, ...)` | Number-field analogue for a tuning system |
| `TuningField.ET(n)` | n-tone equal temperament |
| `TuningField.just(primes=(2,3,5))` | p-limit just intonation |
| `TuningField.degree` | Degree of the field extension [K:Q] |
| `TuningField.ratio(k)` | Frequency ratio for the k-th step |
| `TuningField.cents(k)` | Cents value for the k-th step |
| `AlgebraicTone(ratio, field)` | An element of a tuning field |

### `flux_algebra.groups` — Symmetry Groups

| Class / Function | Description |
|---|---|
| `Triad(root, quality)` | Major/minor triad as a group element |
| `PLRGroup()` | Neo-Riemannian P/L/R transformation group |
| `PLRGroup.P(t)`, `.L(t)`, `.R(t)` | Apply single transformation |
| `PLRGroup.apply(word, triad)` | Apply a word in {P, L, R} |
| `PLRGroup.orbit(triad)` | Full orbit (all 24 triads reachable) |
| `PLRGroup.common_tones(a, b)` | Shared pitch classes between two triads |
| `PLRGroup.walk(word, n, start)` | Apply word n times, returning the walk |
| `TranspositionInversionGroup(modulus=12)` | Full T/I group of order 24 |
| `PermutationVoiceLeading(source, target, perm)` | Voice leading as a group element |

### `flux_algebra.tropical` — Tropical Semiring

| Class / Function | Description |
|---|---|
| `tropical_add(a, b)` | Min (tropical addition) |
| `tropical_multiply(a, b)` | Sum (tropical multiplication) |
| `tropical_power(a, n)` | n·a (tropical exponentiation) |
| `TropicalPolynomial(coeffs)` | Polynomial over the tropical semiring |
| `TropicalHarmony(modulus=12)` | Tropical distance for chords |
| `TropicalHarmony.chord_cost(chord)` | Cost function (distance to nearest tone) |
| `TropicalHarmony.harmonic_tension(chord, scale)` | Tropical tension metric |
| `TropicalHarmony.tropical_voice_leading(src, tgt)` | Optimal voice leading |
| `TropicalVoiceLeading(modulus=12)` | Standalone voice-leading engine |

### `flux_algebra.spectral` — Spectral Analysis

| Class / Function | Description |
|---|---|
| `tenney_height(num, den)` | Harmonic complexity measure |
| `interval_tension(pc1, pc2)` | Tension between two pitch classes |
| `chord_tension(pcs)` | Aggregate tension of a chord |
| `HarmonicLaplacian()` | Tension-weighted Laplacian on the PLR graph |
| `HarmonicLaplacian.eigenvalues()` | Sorted tension modes |
| `EigenbasisHarmonicRing(ring, laplacian)` | Ring with eigenvector decomposition |
| `TonalityFingerprint` | Eigenvalue signature for corpus classification |

### `flux_algebra.modules` — Voice Modules

| Class / Function | Description |
|---|---|
| `VoiceModule(rank, modulus=12)` | Free module (Z/nZ)^k for k voices |
| `VoiceModule.act(element, transposition=0)` | Apply ring action to module element |
| `VoiceModule.homomorphism(matrix)` | Create a module homomorphism (voice leading) |

### `flux_algebra.combinatorics` — Voice Leading

| Function | Description |
|---|---|
| `minimal_voice_leading(source, target, modulus=12)` | Optimal assignment via Hungarian algorithm |
| `all_voice_leadings(source, target, modulus=12)` | Enumerate all possible voice leadings |
| `smoothness(pairs)` | Total semitone movement |
| `efficiency(pairs)` | Ratio of direct to actual movement |

### `flux_algebra.geometry` — Polyhedral Geometry

| Class / Function | Description |
|---|---|
| `TraditionRegion(name, center, radius)` | Convex region in dial space |
| `DialPolytope(points)` | Convex hull of tradition regions |
| `VoiceLeadingGeodesic(from_point, to_point, polytope)` | Shortest voice-leading path |

### `flux_algebra.oscar_compat` — Oscar.jl API

| Function | Oscar.jl equivalent |
|---|---|
| `polynomial_ring("ZZ", 12)` | `R, x = polynomial_ring(QQ, "x")` |
| `number_field("ET-12")` | `K, a = number_field(x^12 - 2)` |
| `group("PLR")` | `G = symmetric_group(24)` |

### `flux_algebra.serialization` — Persistence

| Function | Description |
|---|---|
| `save(obj, path)` | Serialise any algebraic object to JSON |
| `load(path)` | Deserialise from JSON |

## How It Works

The architecture mirrors the module structure of Oscar.jl, adapted to pitch-class theory:

```
                    ┌─────────────┐
                    │ rings.py    │  Z/nZ pitch-class ring
                    │ IntervalRing│  p-limit interval ring
                    │ ChordIdeal  │  ideals = chords/scales
                    └──────┬──────┘
                           │
              ┌────────────┼────────────────┐
              │            │                │
     ┌────────▼───┐  ┌────▼─────┐  ┌──────▼──────┐
     │ fields.py  │  │ groups.py│  │ modules.py  │
     │ TuningField│  │ PLRGroup │  │ VoiceModule │
     │ AlgebraicT │  │ T/I Group│  │ (Z/nZ)^k    │
     └────────────┘  └────┬─────┘  └─────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
  ┌───────▼──────┐ ┌─────▼──────┐ ┌──────▼──────┐
  │ tropical.py  │ │ spectral.py│ │ geometry.py │
  │ min-plus VL  │ │ Laplacian  │ │ dial space  │
  │ tension      │ │ eigenbasis │ │ traditions  │
  └──────────────┘ └────────────┘ └─────────────┘
          │               │               │
          └───────────────┼───────────────┘
                          │
                ┌─────────▼─────────┐
                │ combinatorics.py  │
                │ Hungarian alg.    │
                │ smooth/efficient  │
                └───────────────────┘
```

Each layer builds on the one below: rings define the arithmetic, groups define symmetries of ring elements, modules give those symmetries a multi-voice context, and the upper layers (tropical, spectral, geometric) provide analytical tools.

## The Math

### Ring theory of pitch classes

The ring Z/12Z is a principal ideal ring. Its ideals correspond to divisors of 12:

> For each d | 12, the set d·Z/12Z = {0, d, 2d, ...} is an ideal.

Since 12 = 2² × 3, the divisors are {1, 2, 3, 4, 6, 12}, giving exactly 6 ideals — each one a familiar musical object.

### The PLR group

The neo-Riemannian PLR group is generated by three involutions on the set of 24 triads:

- **P** (Parallel): major ↔ minor with same root. C+ → c−
- **L** (Leading-tone): swaps a chord tone by semitone. C+ → e−
- **R** (Relative): relative major/minor. C+ → a−

These three involutions generate a group isomorphic to D₁₂ (dihedral of order 24) that acts *simply transitively* on the 24 major/minor triads. Any triad can be written as a word w(P, L, R) applied to C+.

### Tropical voice leading

The tropical semiring (R ∪ {∞}, ⊕, ⊗) replaces addition with min and multiplication with +. This is useful for music because:

1. Voice-leading distance is naturally a min-plus problem: find the assignment that minimises total semitone movement
2. The tropical polynomial min_i(x + aᵢ) gives a piecewise-linear cost function that matches the circular structure of pitch-class space
3. Tropical convex hulls of chord tones define "regions of belonging" that avoid boundary discontinuities

### The Eigenbasis Hypothesis

Conservation laws in music (tension, smoothness, voice-leading efficiency) are expressed most naturally in the eigenbasis of the Tension-Graph Laplacian — a weighted graph Laplacian over the 24 triads where edge weights combine PLR transition probability with Tenney-height tension. The eigenvalues sort tension modes from smoothest (tonic vicinity) to roughest (remote modulations).

## Testing

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

382 tests across 10 test files covering every module.

## License

MIT
