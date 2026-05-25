"""Tests for flux_algebra.serialization module."""

import pytest
import tempfile
from pathlib import Path

from flux_algebra.rings import HarmonicRing, IntervalRing, ChordIdeal
from flux_algebra.fields import TuningField, AlgebraicTone
from flux_algebra.groups import TranspositionInversionGroup, PLRGroup, Triad, PermutationVoiceLeading
from flux_algebra.geometry import DialPolytope, TraditionRegion, VoiceLeadingGeodesic
from flux_algebra.tropical import TropicalHarmony, TropicalPolynomial
from flux_algebra.modules import VoiceModule
from flux_algebra.serialization import serialize, deserialize, save, load, save_collection, load_collection


class TestSerializeDeserialize:
    def test_harmonic_ring_roundtrip(self):
        original = HarmonicRing(12)
        data = serialize(original)
        assert data["type"] == "HarmonicRing"
        restored = deserialize(data)
        assert isinstance(restored, HarmonicRing)
        assert restored.modulus == 12

    def test_interval_ring_roundtrip(self):
        original = IntervalRing(prime_limit=(2, 3, 5))
        data = serialize(original)
        restored = deserialize(data)
        assert isinstance(restored, IntervalRing)
        assert restored.prime_limit == (2, 3, 5)

    def test_chord_ideal_roundtrip(self):
        ring = HarmonicRing(12)
        original = ring.chord_ideal([0, 4, 7])
        data = serialize(original)
        restored = deserialize(data)
        assert isinstance(restored, ChordIdeal)
        assert restored.elements == original.elements

    def test_tuning_field_roundtrip(self):
        original = TuningField.ET(12)
        data = serialize(original)
        restored = deserialize(data)
        assert isinstance(restored, TuningField)
        assert restored.name == "ET-12"

    def test_algebraic_tone_roundtrip(self):
        original = AlgebraicTone.from_fraction(3, 2, name="P5")
        data = serialize(original)
        restored = deserialize(data)
        assert isinstance(restored, AlgebraicTone)
        assert restored.name == "P5"
        assert abs(restored.ratio - 1.5) < 0.01

    def test_triad_roundtrip(self):
        original = Triad(0, "major")
        data = serialize(original)
        restored = deserialize(data)
        assert isinstance(restored, Triad)
        assert restored.root == 0
        assert restored.quality == "major"

    def test_ti_group_roundtrip(self):
        original = TranspositionInversionGroup(12)
        data = serialize(original)
        restored = deserialize(data)
        assert isinstance(restored, TranspositionInversionGroup)
        assert restored.order == 24

    def test_plr_group_roundtrip(self):
        original = PLRGroup(12)
        data = serialize(original)
        restored = deserialize(data)
        assert isinstance(restored, PLRGroup)

    def test_tradition_region_roundtrip(self):
        original = TraditionRegion("jazz", center=(3.5, 2.0, 4.0), radius=0.8)
        data = serialize(original)
        restored = deserialize(data)
        assert isinstance(restored, TraditionRegion)
        assert restored.name == "jazz"
        assert restored.center == (3.5, 2.0, 4.0)

    def test_dial_polytope_roundtrip(self):
        traditions = [
            TraditionRegion("jazz", center=(3.5, 2.0, 4.0)),
            TraditionRegion("blues", center=(4.0, 1.5, 3.0)),
        ]
        original = DialPolytope(traditions)
        data = serialize(original)
        restored = deserialize(data)
        assert isinstance(restored, DialPolytope)
        assert len(restored.traditions) == 2

    def test_tropical_harmony_roundtrip(self):
        original = TropicalHarmony(12)
        data = serialize(original)
        restored = deserialize(data)
        assert isinstance(restored, TropicalHarmony)

    def test_tropical_polynomial_roundtrip(self):
        original = TropicalPolynomial([0.0, 4.0, 7.0])
        data = serialize(original)
        restored = deserialize(data)
        assert isinstance(restored, TropicalPolynomial)
        assert restored.coefficients == [0.0, 4.0, 7.0]

    def test_voice_module_roundtrip(self):
        original = VoiceModule(rank=4, modulus=12)
        data = serialize(original)
        restored = deserialize(data)
        assert isinstance(restored, VoiceModule)
        assert restored.rank == 4

    def test_unknown_type(self):
        with pytest.raises(TypeError):
            serialize("not_a_flux_object")

    def test_unknown_deserialize(self):
        with pytest.raises(ValueError):
            deserialize({"type": "UnknownType"})


class TestSaveLoad:
    def test_save_load_triad(self, tmp_path):
        original = Triad(5, "major")
        path = tmp_path / "triad.json"
        save(original, path)
        restored = load(path)
        assert isinstance(restored, Triad)
        assert restored.root == 5

    def test_save_load_tuning_field(self, tmp_path):
        original = TuningField.meantone()
        path = tmp_path / "tuning.json"
        save(original, path)
        restored = load(path)
        assert restored.name == original.name

    def test_save_collection(self, tmp_path):
        objects = {
            "ring": HarmonicRing(12),
            "triad": Triad(0, "major"),
            "field": TuningField.ET(12),
        }
        path = tmp_path / "collection.json"
        save_collection(objects, path)
        restored = load_collection(path)
        assert isinstance(restored["ring"], HarmonicRing)
        assert isinstance(restored["triad"], Triad)
        assert isinstance(restored["field"], TuningField)
