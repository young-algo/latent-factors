"""Integration tests for the briefing command."""

import pytest
import subprocess
import sys


class TestBriefingIntegration:
    """End-to-end tests for briefing generation."""

    def test_briefing_command_runs(self):
        """Briefing command should run without errors."""
        result = subprocess.run(
            [sys.executable, "-m", "src", "briefing", "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "morning briefing" in result.stdout.lower() or "recommendations" in result.stdout.lower()

    @pytest.mark.slow
    def test_briefing_with_cached_data(self):
        """Briefing should work with existing cached factor data."""
        # This test requires cached data to exist
        # Skip if no cache available
        from pathlib import Path

        cache_files = list(Path('.').glob('factor_cache_*.pkl'))
        if not cache_files:
            pytest.skip("No cached factor data available")

        # Extract universe from first cache file
        cache_name = cache_files[0].stem
        parts = cache_name.replace('factor_cache_', '').rsplit('_', 1)
        universe = parts[0]
        method = parts[1] if len(parts) > 1 else 'pca'

        result = subprocess.run(
            [sys.executable, "-m", "src", "briefing",
             "--universe", universe,
             "--method", method],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Should complete (may have warnings but shouldn't crash)
        assert "MORNING BRIEFING" in result.stdout or result.returncode == 0
