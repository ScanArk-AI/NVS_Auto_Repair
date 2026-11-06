#!/usr/bin/env python
"""Entry point: ``python run.py all --config configs/scenes/generated/<scene>/workflow.yaml``."""

import sys

from nvs_repair.cli import main

if __name__ == "__main__":
    sys.exit(main())
