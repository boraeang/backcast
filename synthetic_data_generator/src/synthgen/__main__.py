"""Entry point for ``python -m synthgen``."""
from __future__ import annotations

import sys

from synthgen.cli import main

if __name__ == "__main__":
    sys.exit(main())
