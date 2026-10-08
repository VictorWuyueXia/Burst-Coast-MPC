#!/usr/bin/env python3
"""Inject this repository's native virtual environment into Cursor sessions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VENV_PREFIX = REPOSITORY_ROOT / ".venv"


def main() -> None:
    """Emit the session environment needed to prioritize the local virtual environment."""
    sys.stdin.read()
    python_executable = VENV_PREFIX / "bin" / "python"

    if not python_executable.is_file():
        print(
            json.dumps(
                {
                    "additional_context": (
                        "The native `.venv` is missing. Run the project environment setup "
                        "before starting a Cursor session."
                    )
                }
            )
        )
        return

    print(
        json.dumps(
            {
                "env": {
                    "PATH": os.pathsep.join([str(VENV_PREFIX / "bin"), os.environ["PATH"]]),
                    "VIRTUAL_ENV": str(VENV_PREFIX),
                },
                "additional_context": (
                    "This project uses its native `.venv`; shell commands inherit its PATH "
                    "automatically."
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
