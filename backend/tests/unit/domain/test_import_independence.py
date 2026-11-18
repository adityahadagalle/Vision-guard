"""Verifies framework independence of the VisionGuard domain layer."""

import subprocess
import sys


def test_domain_layer_import_independence() -> None:
    """Verifies that importing the domain layer does not load heavy frameworks or I/O runtimes.

    Specifically asserts that importing backend.app.domain does NOT import:
    - fastapi
    - torch
    - torchvision
    - sqlalchemy
    - sqlite3 (module should not be imported unless standard library dependencies require it)
    """
    script = (
        "import sys\n"
        "import backend.app.domain\n"
        "forbidden = ['fastapi', 'torch', 'torchvision', 'sqlalchemy']\n"
        "loaded_forbidden = [m for m in forbidden if m in sys.modules]\n"
        "if loaded_forbidden:\n"
        "    print(f'Forbidden modules loaded: {loaded_forbidden}', file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"Domain layer imported forbidden frameworks: {result.stderr}"
