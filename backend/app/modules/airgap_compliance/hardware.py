"""Hardware fingerprinting and machine binding module for air-gapped environments.

Extracts deterministic hardware identifiers (MAC address, machine GUID, CPU serial)
and enforces hardware binding invariants per ADR-012 epistemic gating.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
import uuid


class HardwareBindingError(Exception):
    """Raised when hardware identification fails or returns an invalid/randomized dummy MAC."""


class HardwareFingerprinter:
    """Extracts deterministic hardware identifiers for cryptographic device binding."""

    @classmethod
    def get_mac_address(cls) -> int:
        """Retrieve hardware MAC address via uuid.getnode(), enforcing genuine hardware MAC.

        In RFC 4122, if uuid.getnode() fails to obtain a genuine hardware MAC address,
        it generates a pseudo-random 48-bit number with the multicast bit (bit 40,
        the eighth bit of the first octet) set to 1.
        """
        node = uuid.getnode()
        if not isinstance(node, int) or node <= 0 or node >= (1 << 48):
            raise HardwareBindingError(f"Invalid MAC address integer representation: {node}.")
        # Multicast bit check: if bit 40 is 1, it is a randomized dummy MAC
        if (node >> 40) & 1:
            raise HardwareBindingError(
                f"Failed to detect a genuine hardware MAC address: node 0x{node:012x} has multicast bit set."
            )
        return node

    @classmethod
    def get_stable_host_id(cls) -> str:
        """Retrieve stable host identifier (machine-id, Windows MachineGuid, or CPU fallback)."""
        # 1. Linux machine-id inspection
        for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as f:
                        val = f.read().strip()
                        if val:
                            return val
            except Exception:
                pass

        # 2. Windows Registry MachineGuid inspection
        if sys.platform == "win32":
            try:
                import winreg

                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\Cryptography",
                ) as key:
                    guid, _ = winreg.QueryValueEx(key, "MachineGuid")
                    if guid and str(guid).strip():
                        return str(guid).strip()
            except Exception:
                pass

        # 3. Stable processor/CPU fallback
        cpu_id = (
            os.environ.get("CPU_SERIAL")
            or os.environ.get("PROCESSOR_IDENTIFIER")
            or platform.processor()
            or platform.machine()
            or "CPU-SERIAL-FALLBACK"
        )
        return cpu_id.strip()

    @classmethod
    def get_hardware_fingerprint(cls) -> str:
        """Deterministically combine MAC address and stable host identifier, hashed via SHA-256."""
        mac = cls.get_mac_address()
        mac_str = f"{mac:012x}"
        host_id = cls.get_stable_host_id()
        raw_combined = f"MAC:{mac_str}|HOST:{host_id}"
        return hashlib.sha256(raw_combined.encode("utf-8")).hexdigest()
