"""Launch the GeoRede Ajuste desktop interface from a source checkout."""

import os
import sys


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(project_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from freesurvey.gui import launch_gui
    launch_gui()


if __name__ == "__main__":
    main()
