"""
Free Station Survey Method

A Python package for performing free station adjustments using total station data.
"""

__version__ = "0.1.0"
__author__ = "Victor Carvalho"
__description__ = "Geodetic network adjustment, residual analysis, Plotly outputs, and matrix visualization"

from .data_manager import DataManager, SurveyData
from .results_analyzer import (
    SurveyVisualizer,
    load_method_summary,
    method_code,
    method_metadata,
    select_method_summary,
    standardize_adjustment_outputs,
)


def launch_gui(*args, **kwargs):
    """Start the GUI without forcing PyQt imports during package import."""
    from .gui import launch_gui as _launch_gui
    return _launch_gui(*args, **kwargs)


__all__ = [
    "DataManager",
    "SurveyData",
    "SurveyVisualizer",
    "launch_gui",
    "load_method_summary",
    "method_code",
    "method_metadata",
    "select_method_summary",
    "standardize_adjustment_outputs",
]
