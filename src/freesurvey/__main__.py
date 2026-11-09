"""
Main module for running freesurvey as a module.

This allows `python -m freesurvey` to work properly.
"""

from freesurvey.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
