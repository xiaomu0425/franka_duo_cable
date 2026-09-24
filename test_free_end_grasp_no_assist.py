#!/usr/bin/env python3
"""Validate a free-end cable grasp using MuJoCo contact physics only.

This entry point deliberately disables both the gripper grasp-assist force and
the C-clip guide force.  It closes on ``B_last``, waits for real bilateral pad
contact (each side plus the 14-N total threshold), slowly completes the close
with the arm held still, then lifts, holds, opens, and returns a JSON verdict.
A successful result is therefore not an attached or spring-assisted grasp.

Example:
    python test_free_end_grasp_no_assist.py --viewer
"""

from __future__ import annotations

import sys

from test_assisted_free_end_grasp import main


if __name__ == "__main__":
    if "--physical-no-assist" not in sys.argv:
        sys.argv.append("--physical-no-assist")
    raise SystemExit(main())
