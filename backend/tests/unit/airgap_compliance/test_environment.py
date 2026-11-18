"""Unit tests for Air-Gap Environment Validator.

Verifies strict enforcement of offline library flags (HF_DATASETS_OFFLINE,
TRANSFORMERS_OFFLINE) for isolated air-gapped environments.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from backend.app.modules.airgap_compliance.environment import AirgapEnvironmentValidator


class TestAirgapEnvironmentValidator:
    """Test suite for AirgapEnvironmentValidator flag enforcement."""

    def test_validate_offline_flags_both_set_strictly_returns_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns True when both HF_DATASETS_OFFLINE and TRANSFORMERS_OFFLINE are set to '1'."""
        monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

        assert AirgapEnvironmentValidator.validate_offline_flags() is True
        # Verify instance invocation as well
        assert AirgapEnvironmentValidator().validate_offline_flags() is True

    def test_validate_offline_flags_missing_hf_datasets_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns False when HF_DATASETS_OFFLINE is missing."""
        monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

        assert AirgapEnvironmentValidator.validate_offline_flags() is False

    def test_validate_offline_flags_missing_transformers_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns False when TRANSFORMERS_OFFLINE is missing."""
        monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        assert AirgapEnvironmentValidator.validate_offline_flags() is False

    def test_validate_offline_flags_both_missing_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns False when both mandatory offline flags are missing."""
        monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        assert AirgapEnvironmentValidator.validate_offline_flags() is False

    @pytest.mark.parametrize(
        ("hf_val", "tf_val"),
        [
            ("0", "1"),
            ("1", "0"),
            ("0", "0"),
            ("true", "1"),
            ("1", "true"),
            ("True", "1"),
            ("yes", "1"),
            ("", "1"),
            ("1", ""),
            ("2", "1"),
            ("false", "false"),
        ],
    )
    def test_validate_offline_flags_non_one_values_return_false(
        self, monkeypatch: pytest.MonkeyPatch, hf_val: str, tf_val: str
    ) -> None:
        """Returns False if either flag is set to any string other than strictly '1'."""
        monkeypatch.setenv("HF_DATASETS_OFFLINE", hf_val)
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", tf_val)

        assert AirgapEnvironmentValidator.validate_offline_flags() is False

    def test_environment_patching_and_unpatching_isolation(self) -> None:
        """Ensures unittest.mock.patch.dict cleanly modifies and restores environment."""
        # Ensure clean initial state for tested keys
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_DATASETS_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            assert AirgapEnvironmentValidator.validate_offline_flags() is False

            # Within patched block, set both flags to '1'
            with patch.dict(
                os.environ,
                {"HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            ):
                assert AirgapEnvironmentValidator.validate_offline_flags() is True

            # Outside patched block, unpatched state must be restored
            assert AirgapEnvironmentValidator.validate_offline_flags() is False

    def test_get_offline_status_breakdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies get_offline_status returns per-flag boolean status."""
        monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        status = AirgapEnvironmentValidator.get_offline_status()
        assert status == {
            "HF_DATASETS_OFFLINE": True,
            "TRANSFORMERS_OFFLINE": False,
        }

        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        status_compliant = AirgapEnvironmentValidator.get_offline_status()
        assert status_compliant == {
            "HF_DATASETS_OFFLINE": True,
            "TRANSFORMERS_OFFLINE": True,
        }
