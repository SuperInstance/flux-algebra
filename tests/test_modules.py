"""Tests for flux_algebra.modules module."""

import pytest
import numpy as np
from flux_algebra.modules import VoiceModule


class TestVoiceModule:
    def test_creation(self):
        vm = VoiceModule(rank=4, modulus=12)
        assert vm.rank == 4
        assert vm.modulus == 12

    def test_invalid_rank(self):
        with pytest.raises(ValueError):
            VoiceModule(rank=0)

    def test_zero(self):
        vm = VoiceModule(rank=3, modulus=12)
        np.testing.assert_array_equal(vm.zero, [0, 0, 0])

    def test_add(self):
        vm = VoiceModule(rank=3, modulus=12)
        result = vm.add([0, 4, 7], [2, 2, 2])
        np.testing.assert_array_equal(result, [2, 6, 9])

    def test_add_wraps(self):
        vm = VoiceModule(rank=2, modulus=12)
        result = vm.add([10, 11], [5, 3])
        np.testing.assert_array_equal(result, [3, 2])

    def test_subtract(self):
        vm = VoiceModule(rank=3, modulus=12)
        result = vm.subtract([2, 6, 9], [2, 2, 2])
        np.testing.assert_array_equal(result, [0, 4, 7])

    def test_scalar_multiply(self):
        vm = VoiceModule(rank=3, modulus=12)
        result = vm.scalar_multiply(7, [1, 1, 1])
        np.testing.assert_array_equal(result, [7, 7, 7])

    def test_scalar_multiply_wraps(self):
        vm = VoiceModule(rank=2, modulus=12)
        result = vm.scalar_multiply(2, [7, 7])  # tritone doubled = octave = 2
        np.testing.assert_array_equal(result, [2, 2])

    def test_act_transposition(self):
        vm = VoiceModule(rank=4, modulus=12)
        result = vm.act([0, 4, 7, 0], transposition=2)
        np.testing.assert_array_equal(result, [2, 6, 9, 2])

    def test_act_identity(self):
        vm = VoiceModule(rank=3, modulus=12)
        result = vm.act([0, 4, 7])
        np.testing.assert_array_equal(result, [0, 4, 7])

    def test_act_with_matrix(self):
        vm = VoiceModule(rank=2, modulus=12)
        # Swap voices
        matrix = np.array([[0, 1], [1, 0]])
        result = vm.act([0, 7], matrix=matrix)
        np.testing.assert_array_equal(result, [7, 0])

    def test_act_wrong_length(self):
        vm = VoiceModule(rank=3, modulus=12)
        with pytest.raises(ValueError):
            vm.act([0, 4])

    def test_act_wrong_matrix_shape(self):
        vm = VoiceModule(rank=3, modulus=12)
        with pytest.raises(ValueError):
            vm.act([0, 4, 7], matrix=np.eye(2))

    def test_add_wrong_length(self):
        vm = VoiceModule(rank=3, modulus=12)
        with pytest.raises(ValueError):
            vm.add([0, 4], [0, 4, 7])

    def test_basis(self):
        vm = VoiceModule(rank=3, modulus=12)
        basis = vm.basis()
        assert len(basis) == 3
        np.testing.assert_array_equal(basis[0], [1, 0, 0])
        np.testing.assert_array_equal(basis[1], [0, 1, 0])
        np.testing.assert_array_equal(basis[2], [0, 0, 1])

    def test_voice_leading_matrix(self):
        vm = VoiceModule(rank=3, modulus=12)
        mat = vm.voice_leading_matrix([0, 4, 7], [0, 4, 7])
        # Identity-like (zero movement)
        assert mat.shape == (3, 3)

    def test_voice_leadings_between(self):
        vm = VoiceModule(rank=3, modulus=12)
        mats = vm.voice_leadings_between([0, 4, 7], [5, 9, 0])
        assert len(mats) == 6  # 3! permutations

    def test_non12_modulus(self):
        vm = VoiceModule(rank=3, modulus=6)
        result = vm.add([4, 5, 3], [3, 2, 4])
        np.testing.assert_array_equal(result, [1, 1, 1])  # (7,7,7) mod 6

    def test_subtract_wraps(self):
        vm = VoiceModule(rank=2, modulus=12)
        result = vm.subtract([0, 0], [1, 1])
        np.testing.assert_array_equal(result, [11, 11])

    def test_repr(self):
        vm = VoiceModule(rank=4, modulus=12)
        assert "VoiceModule" in repr(vm)
