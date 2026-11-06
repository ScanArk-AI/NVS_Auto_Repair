#!/usr/bin/env python
"""Entry point: ``python run.py all --config configs/scenes/example.yaml``."""

import sys

from repair_pipeline.cli import main

if __name__ == "__main__":
    sys.exit(main())
