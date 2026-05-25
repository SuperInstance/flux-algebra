"""
Serialization — Save/load algebraic structures as JSON.

Analogous to Oscar.jl's Serialization module.

All Flux Algebra objects can be serialized to/from JSON for persistence,
sharing, and interoperability with the constraint-toolkit ecosystem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from flux_algebra.rings import HarmonicRing, IntervalRing, ChordIdeal
from flux_algebra.fields import TuningField, AlgebraicTone
from flux_algebra.groups import TranspositionInversionGroup, PLRGroup, Triad, PermutationVoiceLeading
from flux_algebra.geometry import DialPolytope, TraditionRegion, VoiceLeadingGeodesic
from flux_algebra.tropical import TropicalHarmony, TropicalVoiceLeading, TropicalPolynomial
from flux_algebra.modules import VoiceModule


# Registry of type names → classes
_TYPE_REGISTRY: Dict[str, type] = {
    "HarmonicRing": HarmonicRing,
    "IntervalRing": IntervalRing,
    "ChordIdeal": ChordIdeal,
    "TuningField": TuningField,
    "AlgebraicTone": AlgebraicTone,
    "TranspositionInversionGroup": TranspositionInversionGroup,
    "PLRGroup": PLRGroup,
    "Triad": Triad,
    "PermutationVoiceLeading": PermutationVoiceLeading,
    "DialPolytope": DialPolytope,
    "TraditionRegion": TraditionRegion,
    "VoiceLeadingGeodesic": VoiceLeadingGeodesic,
    "TropicalHarmony": TropicalHarmony,
    "TropicalVoiceLeading": TropicalVoiceLeading,
    "TropicalPolynomial": TropicalPolynomial,
    "VoiceModule": VoiceModule,
}


def _serialize_harmonic_ring(obj: HarmonicRing) -> Dict[str, Any]:
    return {"type": "HarmonicRing", "modulus": obj.modulus}


def _deserialize_harmonic_ring(data: Dict[str, Any]) -> HarmonicRing:
    return HarmonicRing(modulus=data["modulus"])


def _serialize_interval_ring(obj: IntervalRing) -> Dict[str, Any]:
    return {"type": "IntervalRing", "prime_limit": list(obj.prime_limit)}


def _deserialize_interval_ring(data: Dict[str, Any]) -> IntervalRing:
    return IntervalRing(prime_limit=tuple(data["prime_limit"]))


def _serialize_chord_ideal(obj: ChordIdeal) -> Dict[str, Any]:
    return {
        "type": "ChordIdeal",
        "modulus": obj.ring.modulus,
        "generators": sorted(obj.generators),
        "elements": sorted(obj.elements),
    }


def _deserialize_chord_ideal(data: Dict[str, Any]) -> ChordIdeal:
    ring = HarmonicRing(modulus=data["modulus"])
    return ChordIdeal(
        ring=ring,
        generators=frozenset(data["generators"]),
        elements=frozenset(data["elements"]),
    )


def _serialize_tuning_field(obj: TuningField) -> Dict[str, Any]:
    return {
        "type": "TuningField",
        "name": obj.name,
        "generator": obj.generator,
        "minimal_polynomial": obj.minimal_polynomial,
        "primes": list(obj.primes),
    }


def _deserialize_tuning_field(data: Dict[str, Any]) -> TuningField:
    return TuningField(
        name=data["name"],
        generator=data["generator"],
        minimal_polynomial_coeffs=data.get("minimal_polynomial"),
        primes=tuple(data["primes"]),
    )


def _serialize_algebraic_tone(obj: AlgebraicTone) -> Dict[str, Any]:
    return {
        "type": "AlgebraicTone",
        "ratio": obj.ratio,
        "minimal_polynomial": obj.minimal_polynomial,
        "name": obj.name,
        "description": obj.description,
    }


def _deserialize_algebraic_tone(data: Dict[str, Any]) -> AlgebraicTone:
    return AlgebraicTone(
        ratio=data["ratio"],
        minimal_polynomial=tuple(data["minimal_polynomial"]) if data.get("minimal_polynomial") else None,
        name=data.get("name"),
        description=data.get("description"),
    )


def _serialize_triad(obj: Triad) -> Dict[str, Any]:
    return {"type": "Triad", "root": obj.root, "quality": obj.quality}


def _deserialize_triad(data: Dict[str, Any]) -> Triad:
    return Triad(root=data["root"], quality=data["quality"])


def _serialize_ti_group(obj: TranspositionInversionGroup) -> Dict[str, Any]:
    return {"type": "TranspositionInversionGroup", "modulus": obj.modulus}


def _deserialize_ti_group(data: Dict[str, Any]) -> TranspositionInversionGroup:
    return TranspositionInversionGroup(modulus=data["modulus"])


def _serialize_plr_group(obj: PLRGroup) -> Dict[str, Any]:
    return {"type": "PLRGroup", "modulus": obj.modulus}


def _deserialize_plr_group(data: Dict[str, Any]) -> PLRGroup:
    return PLRGroup(modulus=data["modulus"])


def _serialize_tradition_region(obj: TraditionRegion) -> Dict[str, Any]:
    return {
        "type": "TraditionRegion",
        "name": obj.name,
        "center": list(obj.center),
        "radius": obj.radius,
        "description": obj.description,
    }


def _deserialize_tradition_region(data: Dict[str, Any]) -> TraditionRegion:
    return TraditionRegion(
        name=data["name"],
        center=tuple(data["center"]),
        radius=data["radius"],
        description=data.get("description"),
    )


def _serialize_dial_polytope(obj: DialPolytope) -> Dict[str, Any]:
    return {
        "type": "DialPolytope",
        "traditions": [_serialize_tradition_region(t) for t in obj.traditions],
    }


def _deserialize_dial_polytope(data: Dict[str, Any]) -> DialPolytope:
    traditions = [_deserialize_tradition_region(t) for t in data["traditions"]]
    return DialPolytope(traditions=traditions)


def _serialize_tropical_harmony(obj: TropicalHarmony) -> Dict[str, Any]:
    return {"type": "TropicalHarmony", "modulus": obj._modulus}


def _deserialize_tropical_harmony(data: Dict[str, Any]) -> TropicalHarmony:
    return TropicalHarmony(modulus=data["modulus"])


def _serialize_tropical_polynomial(obj: TropicalPolynomial) -> Dict[str, Any]:
    return {"type": "TropicalPolynomial", "coefficients": obj.coefficients}


def _deserialize_tropical_polynomial(data: Dict[str, Any]) -> TropicalPolynomial:
    return TropicalPolynomial(data["coefficients"])


def _serialize_voice_module(obj: VoiceModule) -> Dict[str, Any]:
    return {"type": "VoiceModule", "rank": obj.rank, "modulus": obj.modulus}


def _deserialize_voice_module(data: Dict[str, Any]) -> VoiceModule:
    return VoiceModule(rank=data["rank"], modulus=data["modulus"])


# Serializer/deserializer dispatch
_SERIALIZERS = {
    HarmonicRing: _serialize_harmonic_ring,
    IntervalRing: _serialize_interval_ring,
    ChordIdeal: _serialize_chord_ideal,
    TuningField: _serialize_tuning_field,
    AlgebraicTone: _serialize_algebraic_tone,
    Triad: _serialize_triad,
    TranspositionInversionGroup: _serialize_ti_group,
    PLRGroup: _serialize_plr_group,
    TraditionRegion: _serialize_tradition_region,
    DialPolytope: _serialize_dial_polytope,
    TropicalHarmony: _serialize_tropical_harmony,
    TropicalPolynomial: _serialize_tropical_polynomial,
    VoiceModule: _serialize_voice_module,
}

_DESERIALIZERS = {
    "HarmonicRing": _deserialize_harmonic_ring,
    "IntervalRing": _deserialize_interval_ring,
    "ChordIdeal": _deserialize_chord_ideal,
    "TuningField": _deserialize_tuning_field,
    "AlgebraicTone": _deserialize_algebraic_tone,
    "Triad": _deserialize_triad,
    "TranspositionInversionGroup": _deserialize_ti_group,
    "PLRGroup": _deserialize_plr_group,
    "TraditionRegion": _deserialize_tradition_region,
    "DialPolytope": _deserialize_dial_polytope,
    "TropicalHarmony": _deserialize_tropical_harmony,
    "TropicalPolynomial": _deserialize_tropical_polynomial,
    "VoiceModule": _deserialize_voice_module,
}


def serialize(obj: Any) -> Dict[str, Any]:
    """Serialize a Flux Algebra object to a JSON-compatible dict.

    Parameters
    ----------
    obj : any
        A Flux Algebra object.

    Returns
    -------
    dict
        JSON-compatible dictionary with type information.

    Raises
    ------
    TypeError
        If the object type is not supported.

    Examples
    --------
    >>> from flux_algebra import HarmonicRing
    >>> data = serialize(HarmonicRing(12))
    >>> data["type"]
    'HarmonicRing'
    """
    obj_type = type(obj)
    serializer = _SERIALIZERS.get(obj_type)
    if serializer is None:
        raise TypeError(f"Cannot serialize {obj_type.__name__}")
    return serializer(obj)


def deserialize(data: Dict[str, Any]) -> Any:
    """Deserialize a Flux Algebra object from a dict.

    Parameters
    ----------
    data : dict
        Dictionary with "type" key and type-specific fields.

    Returns
    -------
    any
        Reconstructed Flux Algebra object.

    Raises
    ------
    ValueError
        If the type is unknown.

    Examples
    --------
    >>> obj = deserialize({"type": "HarmonicRing", "modulus": 12})
    >>> obj.modulus
    12
    """
    type_name = data.get("type")
    if type_name not in _DESERIALIZERS:
        raise ValueError(f"Unknown type '{type_name}', expected one of {list(_DESERIALIZERS.keys())}")
    return _DESERIALIZERS[type_name](data)


def save(obj: Any, path: Union[str, Path]) -> None:
    """Save a Flux Algebra object to a JSON file.

    Parameters
    ----------
    obj : any
        A Flux Algebra object.
    path : str or Path
        Output file path.

    Examples
    --------
    >>> from flux_algebra import Triad
    >>> save(Triad(0, "major"), "c_major.json")
    """
    data = serialize(obj)
    Path(path).write_text(json.dumps(data, indent=2))


def load(path: Union[str, Path]) -> Any:
    """Load a Flux Algebra object from a JSON file.

    Parameters
    ----------
    path : str or Path
        Input file path.

    Returns
    -------
    any
        Deserialized Flux Algebra object.

    Examples
    --------
    >>> obj = load("c_major.json")
    """
    data = json.loads(Path(path).read_text())
    return deserialize(data)


def save_collection(objects: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save a collection of named Flux Algebra objects.

    Parameters
    ----------
    objects : dict
        Mapping of names → objects.
    path : str or Path
        Output file path.
    """
    data = {name: serialize(obj) for name, obj in objects.items()}
    Path(path).write_text(json.dumps(data, indent=2))


def load_collection(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a collection of named Flux Algebra objects.

    Parameters
    ----------
    path : str or Path
        Input file path.

    Returns
    -------
    dict
        Mapping of names → deserialized objects.
    """
    data = json.loads(Path(path).read_text())
    return {name: deserialize(obj_data) for name, obj_data in data.items()}
