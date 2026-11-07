"""Compatibility entry point for the centrally configured evaluator.

Experiment selection belongs to :mod:`util`: edit ``SELECTED_DATASET`` and
``WIP_MODE`` there, then run either this file or ``evaluator.py``.
"""

import runpy


if __name__ == "__main__":
    runpy.run_module("evaluator", run_name="__main__")
