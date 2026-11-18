"""Unit tests verifying isolation reassessment and granular control verification.

Validates that HostIsolationAssessor positively establishes enforceable OS controls
rather than relying solely on container marker heuristics or memory limits:
1. Standard Windows host fails closed.
2. Linux host with only container marker fails closed.
3. Linux host missing cgroup pids quota fails closed.
4. Linux host missing NoNewPrivs enforcement fails closed.
5. Linux host running as root user fails closed.
6. Linux host with read-write root mount fails closed.
7. Linux host with all controls positively established returns HARDENED_ISOLATION_VERIFIED.
"""

from pathlib import Path
from unittest.mock import patch

from backend.app.modules.model_integrity.inspection.worker import (
    HostIsolationAssessor,
    IsolationStatus,
)


def test_windows_host_fails_closed_with_granular_report() -> None:
    """Verifies that Windows host fails closed and reports missing controls."""
    assessor = HostIsolationAssessor()
    with patch("platform.system", return_value="Windows"):
        status, details, reason = assessor.assess()
        assert status == IsolationStatus.UNHARDENED_HOST_SUBPROCESS
        assert details["platform"] == "Windows"
        assert details["appcontainer"] is False
        assert details["no_new_privs"] is False
        assert details["non_root_user"] is False
        assert "lacks proven OS AppContainer sandboxing" in reason


def test_linux_host_with_only_container_marker_fails_closed(tmp_path: Path) -> None:
    """Verifies that spoofing a container marker (e.g. /.dockerenv) alone cannot verify isolation."""
    assessor = HostIsolationAssessor()

    with (
        patch("platform.system", return_value="Linux"),
        patch.object(
            Path, "exists", lambda self: self.as_posix() in ("/.dockerenv", "/run/.containerenv")
        ),
    ):
        status, details, reason = assessor.assess()
        assert status == IsolationStatus.UNHARDENED_HOST_SUBPROCESS
        assert details["container"] is True
        assert "cgroup_memory_limit" in details["missing_controls"]
        assert "no_new_privileges" in details["missing_controls"]
        assert "read_only_root_filesystem" in details["missing_controls"]


def test_linux_host_missing_cgroup_pids_limit_fails_closed() -> None:
    """Verifies that lack of cgroup pids limit (fork-bomb defense) fails closed."""
    assessor = HostIsolationAssessor()

    def fake_exists(p: Path) -> bool:
        return p.as_posix() in (
            "/.dockerenv",
            "/sys/fs/cgroup/memory.max",
            "/proc/net/route",
            "/proc/self/status",
            "/proc/mounts",
        )

    def fake_read_text(p: Path) -> str:
        data = {
            "/sys/fs/cgroup/memory.max": "536870912\n",  # 512 MB
            "/proc/net/route": "Iface\tDestination\nlo\t00000000\n",
            "/proc/self/status": "Name:\tpython\nNoNewPrivs:\t1\n",
            "/proc/mounts": "rootfs / rootfs ro,relatime 0 0\n",
        }
        return data.get(p.as_posix(), "")

    with (
        patch("platform.system", return_value="Linux"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch("os.getuid", return_value=1001, create=True),
    ):
        status, details, reason = assessor.assess()
        assert status == IsolationStatus.UNHARDENED_HOST_SUBPROCESS
        assert "cgroup_pids_limit" in details["missing_controls"]


def test_linux_host_missing_no_new_privs_fails_closed() -> None:
    """Verifies that Linux container without NoNewPrivs fails closed."""
    assessor = HostIsolationAssessor()

    def fake_exists(p: Path) -> bool:
        return p.as_posix() in (
            "/.dockerenv",
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/pids.max",
            "/proc/net/route",
            "/proc/self/status",
            "/proc/mounts",
        )

    def fake_read_text(p: Path) -> str:
        data = {
            "/sys/fs/cgroup/memory.max": "536870912\n",
            "/sys/fs/cgroup/pids.max": "100\n",
            "/proc/net/route": "Iface\tDestination\nlo\t00000000\n",
            "/proc/self/status": "Name:\tpython\nNoNewPrivs:\t0\n",  # Privilege escalation possible!
            "/proc/mounts": "rootfs / rootfs ro,relatime 0 0\n",
        }
        return data.get(p.as_posix(), "")

    with (
        patch("platform.system", return_value="Linux"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch("os.getuid", return_value=1001, create=True),
    ):
        status, details, reason = assessor.assess()
        assert status == IsolationStatus.UNHARDENED_HOST_SUBPROCESS
        assert "no_new_privileges" in details["missing_controls"]


def test_linux_host_running_as_root_fails_closed() -> None:
    """Verifies that running inside container as root (uid=0) fails closed without unprivileged user."""
    assessor = HostIsolationAssessor()

    def fake_exists(p: Path) -> bool:
        return p.as_posix() in (
            "/.dockerenv",
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/pids.max",
            "/proc/net/route",
            "/proc/self/status",
            "/proc/mounts",
        )

    def fake_read_text(p: Path) -> str:
        data = {
            "/sys/fs/cgroup/memory.max": "536870912\n",
            "/sys/fs/cgroup/pids.max": "100\n",
            "/proc/net/route": "Iface\tDestination\nlo\t00000000\n",
            "/proc/self/status": "Name:\tpython\nNoNewPrivs:\t1\n",
            "/proc/mounts": "rootfs / rootfs ro,relatime 0 0\n",
        }
        return data.get(p.as_posix(), "")

    with (
        patch("platform.system", return_value="Linux"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch("os.getuid", return_value=0, create=True),  # Root user!
    ):
        status, details, reason = assessor.assess()
        assert status == IsolationStatus.UNHARDENED_HOST_SUBPROCESS
        assert "non_root_user" in details["missing_controls"]


def test_linux_host_with_read_write_root_fails_closed() -> None:
    """Verifies that container with writable root filesystem fails closed."""
    assessor = HostIsolationAssessor()

    def fake_exists(p: Path) -> bool:
        return p.as_posix() in (
            "/.dockerenv",
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/pids.max",
            "/proc/net/route",
            "/proc/self/status",
            "/proc/mounts",
        )

    def fake_read_text(p: Path) -> str:
        data = {
            "/sys/fs/cgroup/memory.max": "536870912\n",
            "/sys/fs/cgroup/pids.max": "100\n",
            "/proc/net/route": "Iface\tDestination\nlo\t00000000\n",
            "/proc/self/status": "Name:\tpython\nNoNewPrivs:\t1\n",
            "/proc/mounts": "rootfs / rootfs rw,relatime 0 0\n",  # Writable root!
        }
        return data.get(p.as_posix(), "")

    with (
        patch("platform.system", return_value="Linux"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch("os.getuid", return_value=1001, create=True),
    ):
        status, details, reason = assessor.assess()
        assert status == IsolationStatus.UNHARDENED_HOST_SUBPROCESS
        assert "read_only_root_filesystem" in details["missing_controls"]


def test_linux_host_with_all_controls_verified_succeeds() -> None:
    """Verifies that positive establishment of all OS sandbox controls returns HARDENED_ISOLATION_VERIFIED."""
    assessor = HostIsolationAssessor()

    def fake_exists(p: Path) -> bool:
        return p.as_posix() in (
            "/.dockerenv",
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/pids.max",
            "/proc/net/route",
            "/proc/self/status",
            "/proc/mounts",
        )

    def fake_read_text(p: Path) -> str:
        data = {
            "/sys/fs/cgroup/memory.max": "536870912\n",
            "/sys/fs/cgroup/pids.max": "100\n",
            "/proc/net/route": "Iface\tDestination\nlo\t00000000\n",
            "/proc/self/status": "Name:\tpython\nNoNewPrivs:\t1\n",
            "/proc/mounts": "rootfs / rootfs ro,relatime 0 0\n",
        }
        return data.get(p.as_posix(), "")

    with (
        patch("platform.system", return_value="Linux"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch("os.getuid", return_value=1001, create=True),  # Non-root user
    ):
        status, details, reason = assessor.assess()
        assert status == IsolationStatus.HARDENED_ISOLATION_VERIFIED
        assert details["container"] is True
        assert details["cgroup_memory_max"] is True
        assert details["cgroup_pids_max"] is True
        assert details["network_severed"] is True
        assert details["no_new_privs"] is True
        assert details["non_root_user"] is True
        assert details["filesystem_restricted"] is True
        assert len(details["missing_controls"]) == 0
        assert "Hardened container sandbox verified" in reason
