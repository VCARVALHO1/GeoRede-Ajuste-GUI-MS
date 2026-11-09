"""
Graphical User Interface Module

Provides PyQt5-based GUI for the TFS (Total Free Station) Method.

This module contains the main graphical user interface components for the
Total Free Station (TFS) survey adjustment software. It provides:

- Data input and management (station folders)
- Multiple TFS adjustment method selection
- Results visualization and statistics
- Lightweight Plotly HTML outputs
- Station data inspection and editing
- Coordinate and matrix export functions

Classes:
    WorkerThread: Handles background calculation processing
    AboutDialog: Application information dialog
    DataInputWidget: Data loading and management interface
    ResultsWidget: Results display and visualization
    StationInspectorDialog: Station data inspection and editing
    MainWindow: Main application window

Author: TFS Development Team
Version: 0.2.0
"""

import sys
import os
import subprocess
import logging
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem,
    QTabWidget, QFileDialog, QMessageBox, QProgressBar, QScrollArea,
    QGroupBox, QComboBox, QTextBrowser, QDialog, QListWidget, QListWidgetItem,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QFormLayout, QHeaderView,
    QSplitter, QSizePolicy, QInputDialog, QStackedWidget, QAbstractItemView,
    QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDateTime, QUrl
from PyQt5.QtGui import QFont, QColor, QPixmap, QIcon, QDesktopServices
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
except Exception:
    QWebEngineView = None

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.image as mpl_image
except Exception:
    FigureCanvas = None
    Figure = None
    mpl_image = None

# Import registration workflows
from freesurvey.Survey_Registration import (
    sequential_registration, Method1_SequentialSVD, Schut1973_SpaceTime,
    Stepien2020_QST,
    Tao2020_CMTLS_Rigorous, TotalFreeStation_TFS, run_network_adjustment,
)
from freesurvey.methods import (
    CMTLS,
    N_CUQ_LM,
    Q_ST,
    SCH_Q,
    SVD_P,
    TFS,
)

from freesurvey.data_manager import DataManager, SurveyData, point_role
from freesurvey.results_analyzer import (
    generate_gui_diagnostic_plots,
    load_method_summary,
    method_metadata,
)


GUI_METHODS = [
    {
        "label": "Sequential: SVD-P - SVD-based Orthogonal Procrustes",
        "registration_type": "sequential",
        "class_name": "SVD_P",
        "method_key": "SVD-P",
    },
    {
        "label": "Sequential: SCH-Q - Schut quaternion transformation",
        "registration_type": "sequential",
        "class_name": "SCH_Q",
        "method_key": "SCH-Q",
    },
    {
        "label": "Sequential: Q-ST - Quasi-Similarity Transformation",
        "registration_type": "sequential",
        "class_name": "Q_ST",
        "method_key": "Q-ST",
    },
    {
        "label": "Sequential: CMTLS - constrained multivariate TLS",
        "registration_type": "sequential",
        "class_name": "CMTLS",
        "method_key": "CMTLS",
    },
    {
        "label": "Sequential: TFS - Total Free Station",
        "registration_type": "sequential",
        "class_name": "TFS",
        "method_key": "TFS",
    },
    {
        "label": "Constrained: N-CUQ-LM - network constrained unit-quaternion LM",
        "registration_type": "network",
        "class_name": "N_CUQ_LM",
        "method_key": "N-CUQ-LM",
    },
]


def _advanced_processing_args(options):
    """Build analyzer arguments that are fixed for the article workflow."""
    options = dict(options or {})
    args = [
        "--min-common-cp", str(int(options.get("min_common_cp", 4) or 4)),
        "--max-iter", str(int(options.get("max_iter", 50) or 50)),
        "--tol", str(float(options.get("tol", 1e-8) or 1e-8)),
        "--apriori-sigma-mm", str(
            float(options.get("apriori_sigma_mm", 3.0) or 3.0)),
        "--outlier-sigma", str(
            float(options.get("outlier_sigma", 3.0) or 3.0)),
        "--outlier-mad", str(
            float(options.get("outlier_mad", 3.5) or 3.5)),
        "--fixed-scale",
    ]
    control_points = str(options.get("control_points", "")).strip()
    if control_points:
        args.append("--control-points")
        args.extend([item.strip() for item in control_points.replace(
            ",", " ").split() if item.strip()])
    outlier_methods = options.get("outlier_methods", ["all"])
    if outlier_methods:
        args.append("--outlier-methods")
        args.extend(outlier_methods)
    independent_check_points = options.get("independent_check_points", [])
    if independent_check_points:
        args.append("--independent-check-points")
        args.extend([str(point_id) for point_id in independent_check_points])
    if options.get("auto_remove_outliers"):
        args.extend([
            "--outlier-removal-iterations",
            str(int(options.get("outlier_removal_iterations", 5) or 5)),
            "--outlier-max-remove-per-iteration",
            str(int(options.get("outlier_max_remove_per_iteration", 1) or 1)),
        ])
    return args


def _decode_process_output(data):
    """Decode captured subprocess output without crashing on Windows encodings."""
    if isinstance(data, str):
        return data
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding, errors="replace")
        except Exception:
            continue
    return str(data)


def _build_analyzer_command(
    script_path,
    data_folder,
    output_dir,
    method_key,
    processing_options,
    outlier_action="report",
    manual_exclusions=None,
):
    """Create the analyzer command used by both processing and Results reruns."""
    cmd = [
        sys.executable,
        str(script_path),
        "--data", str(data_folder),
        "--out", str(output_dir),
        "--methods", str(method_key),
        "--selected-plot-method", str(method_key),
        "--outlier-action", str(outlier_action),
    ]
    cmd.extend(_advanced_processing_args(processing_options))
    manual_exclusions = [str(item).strip() for item in (manual_exclusions or []) if str(item).strip()]
    if manual_exclusions:
        cmd.append("--manual-exclusions")
        cmd.extend(manual_exclusions)
    return cmd


class WorkerThread(QThread):
    """
    Background worker thread for running TFS adjustment calculations.

    This class handles the computationally intensive adjustment calculations
    in a separate thread to prevent blocking the GUI. It supports multiple
    TFS adjustment methods and provides progress updates and error handling.

    Attributes:
        survey_data (SurveyData): Input survey data for adjustment
        adjustment_method (str): Selected adjustment method identifier
        adjustment_engine: Default adjustment engine instance
        tfs_center_of_gravity: TFS Center of Gravity adjustment instance
        tfs_quasi_similarity: TFS Quasi-Similarity adjustment instance
        tfs_linalg: TFS Linear Algebra adjustment instance

    Signals:
        finished (dict): Emitted when calculation completes successfully
        error (str): Emitted when calculation fails with error message
        progress (str): Emitted to update progress status
    """

    # Signal definitions for thread communication
    finished = pyqtSignal(dict)  # Emitted with results dictionary
    error = pyqtSignal(str)      # Emitted with error message string
    progress = pyqtSignal(str)   # Emitted with progress status message

    def __init__(self, survey_data: SurveyData, adjustment_method='default', adjustment_engine=None):
        """
        Initialize the worker thread.
        Args:
                Options: 'default', 'center_of_gravity', 'tfs_center_of_gravity',
                        'tfs_quasi_similarity', 'tfs_linalg'
            adjustment_engine: Optional custom adjustment engine instance
        """
        super().__init__()
        self.survey_data = survey_data
        self.adjustment_method = adjustment_method
        self.registration_type = ""  # "sequential" or "network"
        self.method_class_name = ""  # e.g., "Method1_SequentialSVD"
        self.method_display_name = ""
        self.advanced_outputs_enabled = False
        self.advanced_data_folder = ""
        self.advanced_method_key = ""
        self.advanced_output_dir = ""
        self.processing_options = {}
        self.outlier_action = "report"

    def run(self):
        """
        Execute the TFS adjustment calculation in the background thread.
        This method performs the actual adjustment calculation based on the
        selected method. It handles different TFS algorithms and provides
        progress updates throughout the process.
        The method supports the following adjustment approaches:
        - center_of_gravity: Traditional center of gravity method
        - tfs_center_of_gravity: TFS sequential center of gravity
        - tfs_quasi_similarity: TFS with Euler angles and scale
        - tfs_linalg: TFS direct linear algebra solution
        - default: Standard least squares adjustment
        Emits:
            progress: Status updates during calculation
            finished: Results dictionary on successful completion
            error: Error message on failure
        """
        try:
            self.progress.emit(
                f"Initializing {self.registration_type} registration with {self.method_display_name or self.method_class_name}...")

            # Get the actual method class from its name
            method_class = globals().get(self.method_class_name)
            if not method_class:
                raise ValueError(
                    f"Unknown method class: {self.method_class_name}")
            method_class = self._configured_method_class(method_class)

            # The sequential and network registration functions expect a list of station dictionaries
            # The DataInputWidget's survey_data.stations attribute is a dictionary of stations
            stations_list = list(self.survey_data.stations.values())
            selected_control_points = self._selected_control_points()

            if not stations_list:
                raise ValueError("No station data loaded for registration.")

            # Get min_paired_cps from processing options
            min_paired_cps = int(
                self.processing_options.get("min_common_cp", 4) or 4)

            results = {}
            if self.registration_type == "sequential":
                self.progress.emit(
                    f"Running Sequential Registration with {self.method_display_name or self.method_class_name}...")
                results = sequential_registration(
                    stations_list, method_class, selected_control_points, min_paired_cps)
                # Add type for ResultsWidget
                results['registration_type'] = 'sequential'
            elif self.registration_type == "network":
                self.progress.emit(
                    f"Running Network Adjustment with {self.method_display_name or self.method_class_name}...")
                results = run_network_adjustment(
                    stations_list, method_class, selected_control_points, min_paired_cps)
                # Add type for ResultsWidget
                results['registration_type'] = 'network'
            else:
                raise ValueError(
                    f"Unknown registration type: {self.registration_type}")

            # Check calculation success and emit appropriate signal
            if results and 'stations' in results:
                results['solver_class_name'] = self.method_class_name
                if self.method_display_name:
                    results['method'] = self.method_display_name
                self._run_advanced_outputs(results)
                self.progress.emit("Registration completed successfully!")
                self.finished.emit(results)
            else:
                error_msg = "Unknown error occurred during registration."
                if results and 'stations' in results:
                    failed_stations = [
                        s for s in results['stations'] if 'error' in s]
                    if failed_stations:
                        error_msg = f"Registration failed for {len(failed_stations)} station(s). First error: {failed_stations[0]['error']}"
                elif not results:
                    error_msg = "Registration process returned no results. Check input data and method."
                self.error.emit(f"Registration failed: {error_msg}")

        except Exception as e:
            # Handle any unexpected errors during calculation
            error_msg = f"Calculation error: {str(e)}"
            self.error.emit(error_msg)

    def _run_advanced_outputs(self, results):
        """Generate method-specific output files inside the same worker thread."""
        if not self.advanced_outputs_enabled:
            return
        if not self.advanced_data_folder or not os.path.isdir(self.advanced_data_folder):
            return
        if not self.advanced_method_key:
            return
        root_dir = Path(__file__).resolve().parents[2]
        script_path = root_dir / "analyze_methods_data.py"
        output_dir = Path(self.advanced_output_dir) if self.advanced_output_dir else root_dir / \
            "results" / "article_new_nomenclature_outputs"
        if not script_path.exists():
            return
        if self.outlier_action == "remove":
            self.progress.emit(
                "Generating outputs with automatic outlier removal...")
        else:
            self.progress.emit(
                "Generating selected-method article outputs, diagnostics and HTML...")
        cmd = _build_analyzer_command(
            script_path=script_path,
            data_folder=self.advanced_data_folder,
            output_dir=output_dir,
            method_key=self.advanced_method_key,
            processing_options=self.processing_options,
            outlier_action=self.outlier_action,
        )
        run = subprocess.run(
            cmd,
            cwd=str(root_dir),
            capture_output=True,
            check=False,
            timeout=120,
        )
        stdout = _decode_process_output(run.stdout)
        stderr = _decode_process_output(run.stderr)
        run.stdout = stdout
        run.stderr = stderr
        if run.returncode == 0:
            results['advanced_output_dir'] = str(output_dir)
            results['advanced_method_key'] = self.advanced_method_key
            results['outlier_action'] = self.outlier_action
            results['outlier_removal_applied'] = self.outlier_action == "remove"
            html_path = output_dir / "selected_method_3d.html"
            if html_path.exists():
                results['advanced_html_path'] = str(html_path)
            if self.outlier_action == "remove":
                rmse_html_path = output_dir / "selected_method_rmse_3d.html"
                if not rmse_html_path.exists():
                    rmse_html_path = output_dir / "plotly_3d_by_method" / \
                        f"{self.advanced_method_key}_rmse_3d.html"
            else:
                rmse_html_path = output_dir / "plotly_3d_by_method" / \
                    f"{self.advanced_method_key}_rmse_3d.html"
                if not rmse_html_path.exists():
                    rmse_html_path = output_dir / "selected_method_rmse_3d.html"
            if rmse_html_path.exists():
                results['advanced_rmse_html_path'] = str(rmse_html_path)
            results['advanced_stdout'] = run.stdout
        else:
            results['advanced_error'] = run.stderr or run.stdout

    def _apply_global_options(self):
        # Options are now passed as parameters to registration functions for thread-safety
        pass

    def _configured_method_class(self, method_class):
        options = dict(self.processing_options or {})
        max_iter = int(options.get("max_iter", 50) or 50)
        tol = float(options.get("tol", 1e-8) or 1e-8)
        weight_constraint = 1e5
        use_quaternion_constraint = True
        fixed_scale = True

        def fixed_scale_pair(source, target):
            started = QDateTime.currentDateTime()
            source = np.asarray(source, dtype=float)
            target = np.asarray(target, dtype=float)
            mean_s = np.mean(source, axis=0)
            mean_t = np.mean(target, axis=0)
            source_c = source - mean_s
            target_c = target - mean_t
            u, _, vt = np.linalg.svd(source_c.T @ target_c)
            rotation = vt.T @ u.T
            if np.linalg.det(rotation) < 0:
                vt[2, :] *= -1
                rotation = vt.T @ u.T
            scale = 1.0
            translation = mean_t - mean_s @ rotation.T
            transformed = scale * (source @ rotation.T) + translation
            residuals = transformed - target
            elapsed = started.msecsTo(QDateTime.currentDateTime()) / 1000.0
            return scale, rotation, translation, residuals, elapsed

        def qst_fixed_scale_pair(source, target):
            started = QDateTime.currentDateTime()
            source = np.asarray(source, dtype=float)
            target = np.asarray(target, dtype=float)
            n_points = len(source)
            d_source = []
            d_target = []
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    d_source.append(np.linalg.norm(source[j] - source[i]))
                    d_target.append(np.linalg.norm(target[j] - target[i]))
            d_source_all = np.asarray(d_source, dtype=float).reshape(-1, 1)
            d_target_all = np.asarray(d_target, dtype=float).reshape(-1, 1)
            distance_residuals_all = d_source_all - d_target_all
            if len(distance_residuals_all) > 1:
                sigma_d = float(
                    np.sqrt(
                        (distance_residuals_all.T @
                         distance_residuals_all)[0, 0]
                        / (len(distance_residuals_all) - 1)
                    )
                )
            else:
                sigma_d = 0.0
            pair_mask = np.ones(len(d_source_all), dtype=bool)
            d_source_used = d_source_all
            d_target_used = d_target_all
            if sigma_d > 0:
                pair_mask = np.abs(
                    distance_residuals_all.ravel()) < 2.0 * sigma_d
                if np.count_nonzero(pair_mask) >= 1 and np.count_nonzero(~pair_mask) > 0:
                    d_source_used = d_source_all[pair_mask]
                    d_target_used = d_target_all[pair_mask]

            scale = 1.0
            source_scaled = source
            source_centroid_scaled = np.mean(source_scaled, axis=0)
            source_centroid = np.mean(source, axis=0)
            target_centroid = np.mean(target, axis=0)
            source_centered = source_scaled - source_centroid_scaled
            target_centered = target - target_centroid
            b_matrix, _, _, _ = np.linalg.lstsq(
                source_centered, target_centered, rcond=None)
            quasi_rotation = b_matrix.T
            initial_centroid_residuals = target_centered - source_centered @ quasi_rotation.T
            correction_design = np.zeros((3 * n_points, 12), dtype=float)
            correction_observation = initial_centroid_residuals.ravel()
            for row, point in enumerate(source_centered):
                base = 3 * row
                x, y, z = point
                correction_design[base, 0] = 1.0
                correction_design[base, 3:6] = [x, y, z]
                correction_design[base + 1, 1] = 1.0
                correction_design[base + 1, 6:9] = [x, y, z]
                correction_design[base + 2, 2] = 1.0
                correction_design[base + 2, 9:12] = [x, y, z]
            correction, _, _, _ = np.linalg.lstsq(
                correction_design, correction_observation, rcond=None
            )
            centroid_correction = correction[:3]
            quasi_rotation = quasi_rotation + correction[3:].reshape(3, 3)
            translation = target_centroid - \
                source_centroid @ quasi_rotation.T + centroid_correction
            residuals = source @ quasi_rotation.T + translation - target
            dof = max(1, 3 * n_points - 9)
            sigma0_sq = float(np.trace(residuals.T @ residuals) / dof)
            orthogonality_matrix = np.eye(
                3) - quasi_rotation.T @ quasi_rotation
            stats = {
                "sigma0_sq": sigma0_sq,
                "rmse": float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))),
                "max_residual": float(np.max(np.abs(residuals))),
                "distance_design_matrix": d_source_used,
                "distance_observation_vector": d_target_used,
                "distance_residuals": d_source_used - d_target_used,
                "distance_residuals_all": distance_residuals_all,
                "distance_sigma_D": sigma_d,
                "distance_pair_mask": pair_mask,
                "scale_pairs_removed": int(np.count_nonzero(~pair_mask)),
                "source_scaled": source_scaled,
                "source_centroid_scaled": source_centroid_scaled,
                "target_centroid": target_centroid,
                "source_centered_scaled": source_centered,
                "target_centered": target_centered,
                "initial_centroid_residuals": initial_centroid_residuals,
                "qst_correction_design_matrix": correction_design,
                "qst_correction_vector": correction,
                "qst_centroid_correction": centroid_correction,
                "quasi_rotation_matrix": quasi_rotation,
                "orthogonality_matrix": orthogonality_matrix,
                "orthogonality_error": float(np.linalg.norm(orthogonality_matrix)),
                "zero_deviation_max_abs": float(np.max(np.abs(residuals))),
                "fixed_scale_applied": True,
                "fixed_scale_value": 1.0,
            }
            elapsed = started.msecsTo(QDateTime.currentDateTime()) / 1000.0
            return scale, quasi_rotation, translation, residuals, stats, elapsed

        class ConfiguredMethod(method_class):
            __name__ = method_class.__name__
            __qualname__ = method_class.__qualname__

            @classmethod
            def solve(cls, *args, **kwargs):
                if fixed_scale:
                    if getattr(method_class, "is_global_lm", False):
                        source_list, target_input = args[0], args[1]
                        target_is_list = isinstance(target_input, list)
                        transforms = []
                        residual_chunks = []
                        elapsed_total = 0.0
                        for index, source in enumerate(source_list):
                            target = target_input[index] if target_is_list else target_input
                            scale, rotation, translation, residuals, elapsed = fixed_scale_pair(
                                source, target)
                            transforms.append((scale, rotation, translation))
                            residual_chunks.append(residuals.ravel())
                            elapsed_total += elapsed
                        residual_vector = np.concatenate(
                            residual_chunks) if residual_chunks else np.array([])
                        dof = max(1, residual_vector.size -
                                  6 * max(1, len(transforms)))
                        sigma0_sq = float(np.sum(residual_vector ** 2) / dof)
                        covariance = np.eye(
                            8 * max(1, len(transforms))) * sigma0_sq
                        return transforms, sigma0_sq, covariance, residual_vector, elapsed_total

                    method_name = method_class.__name__
                    if getattr(method_class, "method_code", "") == "TFS" or method_name in {"TotalFreeStation_TFS", "TFS"}:
                        return method_class.solve(*args, **kwargs)
                    if method_name == "Stepien2020_QST":
                        scale, rotation, translation, residuals, stats, elapsed = qst_fixed_scale_pair(
                            args[0], args[1])
                        return scale, rotation, translation, stats, elapsed

                    scale, rotation, translation, residuals, elapsed = fixed_scale_pair(
                        args[0], args[1])
                    if getattr(method_class, "method_code", "") == "CMTLS" or method_name in {"Tao2020_CMTLS_Rigorous", "CMTLS"}:
                        stats = {
                            "sigma0_sq": float(np.mean(residuals ** 2)),
                            "rmse": float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))),
                            "fixed_scale_applied": True,
                        }
                        return scale, rotation, translation, np.eye(9) * stats["sigma0_sq"], stats, elapsed
                    return scale, rotation, translation, elapsed

                import inspect
                signature = inspect.signature(method_class.solve)
                if "max_iter" in signature.parameters:
                    kwargs.setdefault("max_iter", max_iter)
                if "tol" in signature.parameters:
                    kwargs.setdefault("tol", tol)
                if "weight_constraint" in signature.parameters:
                    kwargs.setdefault("weight_constraint", weight_constraint)
                if "use_quaternion_constraint" in signature.parameters:
                    if getattr(method_class, "force_quaternion_constraint", None) is False:
                        kwargs.setdefault("use_quaternion_constraint", False)
                    else:
                        kwargs.setdefault(
                            "use_quaternion_constraint", use_quaternion_constraint)
                return method_class.solve(*args, **kwargs)

        for attr in ("is_global_lm", "parameterization", "force_quaternion_constraint"):
            if hasattr(method_class, attr):
                setattr(ConfiguredMethod, attr, getattr(method_class, attr))
        ConfiguredMethod.__name__ = method_class.__name__
        return ConfiguredMethod

    def _advanced_option_args(self):
        return _advanced_processing_args(self.processing_options)

    def _selected_control_points(self):
        control_points = str(self.processing_options.get(
            "control_points", "")).strip()
        return [item.strip() for item in control_points.replace(",", " ").split() if item.strip()] if control_points else None


class AnalyzerWorkerThread(QThread):
    """Run analyzer-only jobs, such as confirmed outlier removal, without freezing the GUI."""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        data_folder,
        output_dir,
        method_key,
        processing_options,
        outlier_action="report",
        manual_exclusions=None,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.output_dir = str(output_dir)
        self.method_key = method_key
        self.processing_options = dict(processing_options or {})
        self.outlier_action = outlier_action
        self.manual_exclusions = list(manual_exclusions or [])

    def run(self):
        try:
            root_dir = Path(__file__).resolve().parents[2]
            script_path = root_dir / "analyze_methods_data.py"
            output_dir = Path(self.output_dir)
            if not script_path.exists():
                self.error.emit("Analyzer script was not found.")
                return
            if not self.data_folder or not os.path.isdir(self.data_folder):
                self.error.emit(
                    "Station data folder is not available for analyzer rerun.")
                return

            if self.manual_exclusions:
                self.progress.emit(
                    f"Reprocessing {self.method_key} after manual residual review...")
            else:
                self.progress.emit(
                    f"Reprocessing {self.method_key} with confirmed outlier removal...")
            cmd = _build_analyzer_command(
                script_path=script_path,
                data_folder=self.data_folder,
                output_dir=output_dir,
                method_key=self.method_key,
                processing_options=self.processing_options,
                outlier_action=self.outlier_action,
                manual_exclusions=self.manual_exclusions,
            )
            run = subprocess.run(
                cmd,
                cwd=str(root_dir),
                capture_output=True,
                check=False,
                timeout=120,
            )
            stdout = _decode_process_output(run.stdout)
            stderr = _decode_process_output(run.stderr)
            if run.returncode != 0:
                self.error.emit(stderr or stdout or "Analyzer rerun failed.")
                return

            result = {
                "advanced_output_dir": str(output_dir),
                "advanced_method_key": self.method_key,
                "advanced_stdout": stdout,
                "outlier_action": self.outlier_action,
            }
            html_path = output_dir / "selected_method_3d.html"
            if html_path.exists():
                result["advanced_html_path"] = str(html_path)
            rmse_html_path = output_dir / "plotly_3d_by_method" / \
                f"{self.method_key}_rmse_3d.html"
            if not rmse_html_path.exists():
                rmse_html_path = output_dir / "selected_method_rmse_3d.html"
            if rmse_html_path.exists():
                result["advanced_rmse_html_path"] = str(rmse_html_path)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class AboutDialog(QDialog):
    """
    About dialog displaying comprehensive information about the TFS application.

    This dialog provides detailed information about GeoNet Adjust
    software, including its purpose, mathematical foundation, key features,
    applications, and technical specifications.

    Features:
    - Program title and version information
    - Detailed description of capabilities
    - Mathematical foundation explanation
    - Key features and applications
    - Technical specifications
    - Educational content about least squares adjustment

    Attributes:
        parent: Parent widget (typically MainWindow)
    """

    def __init__(self, parent=None):
        """
        Initialize the About dialog.

        Args:
            parent: Parent widget for modal behavior
        """
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """
        Initialize the user interface components of the About dialog.

        Sets up the dialog layout with:
        - Program title and version information
        - Rich HTML description with features and technical details
        - Close button for user interaction
        - Proper window sizing and positioning
        """
        layout = QVBoxLayout()

        # Title
        title = QLabel("GeoNet Adjust")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Version
        version = QLabel("Version 0.2.0")
        version.setFont(QFont("Arial", 10))
        version.setAlignment(Qt.AlignCenter)
        layout.addWidget(version)

        # Description
        description = QTextBrowser()
        description.setMinimumHeight(400)
        description.setHtml("""
        <h3>About GeoNet Adjust</h3>

        <p><b>GeoNet Adjust</b> integrates free-station processing,
        similarity transformations, residual analysis, covariance inspection,
        quality diagnostics and Plotly HTML outputs for geodetic network adjustment.</p>

        <h4>Main capabilities:</h4>
        <ul>
        <li><b>Processing methods:</b> SVD-P, SCH-Q, Q-ST,
        CMTLS, TFS and the network method N-CUQ-LM.</li>
        <li><b>Residual analysis:</b> residual vectors, RMS/RMSE, outlier flags and
        per-station quality summaries when the selected method exposes the data.</li>
        <li><b>Uncertainty outputs:</b> compact precision, covariance and quality indicators
        whenever they are available or can be estimated safely.</li>
        <li><b>Visualization:</b> point clouds by method, error ellipses when
        covariance is available, coordinate tables and article-ready CSV/HTML outputs.</li>
        </ul>

        <h4>Reference basis:</h4>
        <ul>
        <li>Arun, Huang and Blostein (1987), SVD-based least-squares fitting of 3D point sets.</li>
        <li>Schut (1973) and Horn (1987), quaternion-based absolute orientation.</li>
        <li>Stepien et al. (2020), Quasi-Similarity Transformation (Q-ST).</li>
        <li>Tao, Hua and Feng (2020), Constrained Multivariate Total Least Squares (CMTLS).</li>
        <li>Lehmann (2013), <i>The 3-sigma rule for outlier detection from the viewpoint of geodetic adjustment</i>.</li>
        </ul>

        <p><i>Author:</i> Victor Carvalho.</p>
        <p><i>Objective:</i> support transparent comparison of adjustment methods,
        generated matrices, residuals and statistical diagnostics in one GUI.</p>
        """)
        layout.addWidget(description)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        self.setLayout(layout)

        self.setWindowTitle("About - GeoNet Adjust")
        self.setGeometry(300, 300, 720, 560)


class DataInputWidget(QWidget):
    """
    Widget for survey data input, management, and visualization.

    This widget provides the primary interface for loading, viewing, and managing
    survey data in the TFS application. It supports multiple data formats:

    - Station-based data (multiple stations from folder)

    Features:
    - Station folder loading for multi-station surveys
    - Data validation and statistics display
    - Interactive data table with editing capabilities
    - Export functionality for different data formats

    Attributes:
        survey_data (SurveyData): Current loaded survey data
        data_type (str): Type of loaded data ("station_data", "unknown")
        logger: Logging instance for debug information
        measurement_table (QTableWidget): Main data display table
    """

    data_loaded_signal = pyqtSignal(bool)

    def __init__(self):
        """
        Initialize the DataInputWidget.

        Sets up the widget with default empty survey data and initializes
        the user interface components for data management.
        """
        super().__init__()
        self.survey_data = SurveyData()
        self.data_type = "unknown"  # Tracks current data format
        self.station_folder_path = ""
        self.point_table_updating = False
        self.point_table_dirty = False
        self.current_point_station_name = ""
        self.logger = logging.getLogger(__name__)
        self.init_ui()

    def init_ui(self):
        """
        Initialize the user interface components for data input and management.

        Sets up the complete UI layout including:
        - Station folder loading
        - Data management buttons
        - Interactive data table for viewing/editing survey data
        - Statistics display panel
        - Data type indicators and status information

        The UI is organized in logical groups for better user experience
        and includes tooltips for user guidance.
        """
        layout = QVBoxLayout()

        header_layout = QHBoxLayout()

        import_group = QGroupBox("1. Configuration")
        import_layout = QFormLayout()
        self.station_delimiter_combo = QComboBox()
        self.station_delimiter_combo.addItems([
            "Auto", "Tab", "Semicolon (;)", "Comma (,)", "Space"
        ])
        import_layout.addRow("Separator:", self.station_delimiter_combo)

        self.coordinate_order_combo = QComboBox()
        self.coordinate_order_combo.addItems([
            "X,Y,Z", "Y,X,Z", "X,Z,Y", "Z,X,Y", "Y,Z,X", "Z,Y,X"
        ])
        import_layout.addRow("Coordinate order:", self.coordinate_order_combo)
        import_group.setLayout(import_layout)
        header_layout.addWidget(import_group, 1)

        button_group = QGroupBox("2. Data Loading")
        button_layout = QVBoxLayout()
        self.load_folder_btn = QPushButton("Load Station Folder")
        self.load_folder_btn.clicked.connect(self.load_station_folder)
        self.load_folder_btn.setToolTip(
            "Load survey data from station files in a folder")
        self.load_folder_btn.setMinimumHeight(34)
        self.load_folder_btn.setMinimumWidth(170)
        self.load_folder_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        button_layout.addWidget(self.load_folder_btn)
        self.clear_btn = QPushButton("Clear Data")
        self.clear_btn.clicked.connect(self.clear_data)
        self.clear_btn.setToolTip("Clear all loaded data")
        self.clear_btn.setMinimumHeight(30)
        self.clear_btn.setMinimumWidth(120)
        button_layout.addWidget(self.clear_btn)
        button_group.setLayout(button_layout)
        header_layout.addWidget(button_group, 2)
        layout.addLayout(header_layout)

        self.input_tabs = QTabWidget()
        self.input_tabs.setStyleSheet("""
            QTabWidget::pane { border-top: 1px solid #d0d0d0; }
            QTabBar::tab {
                min-width: 86px;
                min-height: 22px;
                padding: 3px 8px;
                font-size: 11px;
            }
        """)
        self.input_tabs.addTab(
            self._create_point_configuration_tab(), "Point Configuration")

        loaded_data_tab = QWidget()
        loaded_data_layout = QVBoxLayout()
        self.measurement_table = QTableWidget()
        self.setup_measurement_table()
        self.measurement_table.setMinimumHeight(260)
        self.measurement_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded)
        self.measurement_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        loaded_data_layout.addWidget(self.measurement_table)
        loaded_data_tab.setLayout(loaded_data_layout)
        self.input_tabs.addTab(loaded_data_tab, "Loaded Data Preview")

        layout.addWidget(self.input_tabs)

        # Statistics
        self.stats_group = QGroupBox("Survey Data Statistics")
        stats_layout = QHBoxLayout()

        self.survey_points_label = QLabel("Survey Points: 0")
        stats_layout.addWidget(self.survey_points_label)

        self.measurement_count_label = QLabel("Measurements: 0")
        stats_layout.addWidget(self.measurement_count_label)

        # Add data type indicator
        self.data_type_label = QLabel("Data Type: Unknown")
        self.data_type_label.setStyleSheet("font-weight: bold; color: #666;")
        stats_layout.addWidget(self.data_type_label)

        self.source_folder_label = QLabel("Source: not loaded")
        self.source_folder_label.setToolTip("Current station input folder")
        self.source_folder_label.setStyleSheet("color: #555;")
        stats_layout.addWidget(self.source_folder_label, 1)

        self.stats_group.setLayout(stats_layout)
        layout.addWidget(self.stats_group)

        self.setLayout(layout)
        self.refresh_point_station_selector()

    def setup_measurement_table(self):
        """Setup the measurements table."""
        self.measurement_table.setColumnCount(8)
        self.measurement_table.setHorizontalHeaderLabels([
            "From Point", "To Point", "Hz Angle", "V Angle",
            "Distance", "Inst Height", "Target Height"
        ])
        self.measurement_table.setAlternatingRowColors(True)

    def refresh_point_station_selector(self):
        """Refresh the station selector and load the visible point table."""
        if not hasattr(self, 'point_station_combo'):
            return
        self.point_table_updating = True
        self.point_station_combo.clear()
        stations = getattr(self.survey_data, 'stations', {})
        if stations:
            self.point_station_combo.addItems(sorted(stations.keys()))
        else:
            self.point_station_combo.addItem("Manual Station")
        self.point_table_updating = False
        self.load_selected_station_points(
            self.point_station_combo.currentText())

    def _on_point_station_changed(self, station_name):
        if self.point_table_updating:
            return
        if self.point_table_dirty and self.current_point_station_name:
            self._commit_point_table_to_station(
                self.current_point_station_name, show_message=False)
        self.load_selected_station_points(station_name)

    def load_selected_station_points(self, station_name=None):
        """Load one station into the editable point table."""
        if not hasattr(self, 'point_config_table'):
            return
        station_name = station_name or self.point_station_combo.currentText()
        stations = getattr(self.survey_data, 'stations', {})
        self.point_table_updating = True
        self.point_config_table.setRowCount(0)
        if station_name in stations:
            station = stations[station_name]
            coords_arr = station.get('coords', [])
            all_ids = station.get('all_ids', [])
            ctrl_ids = set(station.get('ctrl_ids', []))
            point_codes = station.get('point_codes', {})
            point_comments = station.get('point_comments', {})
            icp_ids = set(station.get('icp_ids', []))
            for idx, point_id in enumerate(all_ids):
                if idx >= len(coords_arr):
                    continue
                coords = coords_arr[idx]
                self.add_point_row({
                    'name': point_id,
                    'x': coords[0],
                    'y': coords[1],
                    'z': coords[2],
                    'point_code': point_codes.get(point_id, 'CONTROL' if point_id in ctrl_ids else 'SURVEY'),
                    'is_control_point': point_id in ctrl_ids,
                    'is_icp': point_id in icp_ids or point_role(point_id, point_codes.get(point_id)) == "icp",
                    'comment': point_comments.get(point_id, '')
                }, role='manual')
        else:
            self.add_point_row()
        self.point_table_updating = False
        self.current_point_station_name = station_name
        self.point_table_dirty = False
        self.update_point_preview()

    def add_point_row(self, point=None, role='empty'):
        """Add one editable point row."""
        point = point or {}
        row = self.point_config_table.rowCount()
        previous_state = self.point_table_updating
        self.point_table_updating = True
        self.point_config_table.insertRow(row)
        self._set_point_row(row, point, role)
        self.point_table_updating = previous_state
        if not previous_state:
            self.point_table_dirty = True
        self.update_point_preview()

    def _set_point_row(self, row, point, role='empty'):
        values = [
            str(point.get('name', '')),
            self._format_optional_number(point.get('x', '')),
            self._format_optional_number(point.get('y', '')),
            self._format_optional_number(point.get('z', '')),
            str(point.get('point_code', '')),
            '',
            '',
            str(point.get('comment', ''))
        ]
        for col, value in enumerate(values):
            if col in (5, 6):
                item = QTableWidgetItem()
                item.setFlags(
                    (item.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
                checked = point.get(
                    'is_control_point', False) if col == 5 else point.get('is_icp', False)
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                item.setData(Qt.UserRole, role)
            else:
                item = QTableWidgetItem(value)
            self.point_config_table.setItem(row, col, item)

    def _format_optional_number(self, value):
        if value is None or value == '':
            return ''
        try:
            return f"{float(value):.6f}"
        except (TypeError, ValueError):
            return str(value)

    def remove_selected_points(self):
        rows = sorted(
            {index.row() for index in self.point_config_table.selectedIndexes()},
            reverse=True
        )
        for row in rows:
            self.point_config_table.removeRow(row)
        self.point_table_dirty = True
        self.update_point_preview()

    def clear_point_table(self):
        self.point_config_table.setRowCount(0)
        self.point_table_dirty = True
        self.update_point_preview()

    def _on_point_table_item_changed(self, item):
        if self.point_table_updating:
            return
        self.point_table_dirty = True
        if item.column() in (0, 4):
            trimmed = item.text().strip()
            if trimmed != item.text():
                self.point_table_updating = True
                item.setText(trimmed)
                self.point_table_updating = False
            self._apply_control_rule_to_row(
                item.row(), overwrite=item.column() == 4)
        elif item.column() in (5, 6):
            if item.column() == 6 and item.checkState() == Qt.Checked:
                control_item = self.point_config_table.item(item.row(), 5)
                if control_item is not None and control_item.checkState() == Qt.Checked:
                    self.point_table_updating = True
                    control_item.setCheckState(Qt.Unchecked)
                    self.point_table_updating = False
            elif item.column() == 5 and item.checkState() == Qt.Checked:
                icp_item = self.point_config_table.item(item.row(), 6)
                if icp_item is not None and icp_item.checkState() == Qt.Checked:
                    self.point_table_updating = True
                    icp_item.setCheckState(Qt.Unchecked)
                    self.point_table_updating = False
            item.setData(Qt.UserRole, 'manual')
        self.update_point_preview()

    def _point_cell_text(self, row, col):
        item = self.point_config_table.item(row, col)
        return item.text().strip() if item else ''

    def _parse_coordinate(self, text):
        clean = text.strip().replace(',', '.')
        return float(clean)

    def _control_prefixes(self):
        raw = self.control_prefix_input.text().replace(';', ',').replace(' ', ',')
        return [part.strip().upper() for part in raw.split(',') if part.strip()]

    def _is_control_name(self, name):
        clean_name = name.strip().upper()
        return any(clean_name.startswith(prefix) for prefix in self._control_prefixes())

    def _apply_control_rule_to_row(self, row, overwrite=False):
        if not self.control_rule_enabled_check.isChecked():
            return
        control_item = self.point_config_table.item(row, 5)
        icp_item = self.point_config_table.item(row, 6)
        if control_item is None:
            return
        role = control_item.data(Qt.UserRole)
        if not overwrite and role == 'manual':
            return
        point_name = self._point_cell_text(row, 0)
        point_code = self._point_cell_text(row, 4)
        role_from_code = point_role(point_name, point_code)
        is_control = role_from_code == "control" or self._is_control_name(
            point_name)
        is_icp = role_from_code == "icp"
        if icp_item is not None and is_icp:
            icp_item.setCheckState(Qt.Checked)
            icp_item.setData(Qt.UserRole, 'auto')
            is_control = False
        control_item.setCheckState(Qt.Checked if is_control else Qt.Unchecked)
        control_item.setData(Qt.UserRole, 'auto')

    def _is_icp_name(self, name):
        return name.strip().upper().startswith("ICP")

    def apply_control_point_rule(self, show_message=True):
        """Apply the configured prefix rule to control-point flags."""
        overwrite = self.control_rule_mode_combo.currentText().startswith("Overwrite")
        self.point_table_updating = True
        for row in range(self.point_config_table.rowCount()):
            self._apply_control_rule_to_row(row, overwrite=overwrite)
        self.point_table_updating = False
        self.point_table_dirty = True
        self.update_point_preview()
        if show_message:
            QMessageBox.information(
                self, "Control Rule", "Control point rule applied.")

    def _reset_point_table_highlights(self):
        for row in range(self.point_config_table.rowCount()):
            for col in range(self.point_config_table.columnCount()):
                item = self.point_config_table.item(row, col)
                if item:
                    item.setBackground(QColor('white'))

    def _mark_invalid_cell(self, row, col):
        item = self.point_config_table.item(row, col)
        if item:
            item.setBackground(QColor('#FFCDD2'))

    def validate_point_table(self, show_message=True):
        """Validate required names, numeric coordinates, duplicates, and flags."""
        self._reset_point_table_highlights()
        if self.control_rule_enabled_check.isChecked():
            self.apply_control_point_rule(show_message=False)

        errors = []
        seen_names = {}
        for row in range(self.point_config_table.rowCount()):
            name = self._point_cell_text(row, 0)
            if not name:
                errors.append(f"Row {row + 1}: point name is required.")
                self._mark_invalid_cell(row, 0)
            else:
                key = name.upper()
                if key in seen_names:
                    errors.append(
                        f"Rows {seen_names[key] + 1} and {row + 1}: duplicated point name '{name}'.")
                    self._mark_invalid_cell(row, 0)
                    self._mark_invalid_cell(seen_names[key], 0)
                seen_names[key] = row

            for col, label in ((1, 'X'), (2, 'Y'), (3, 'Z')):
                try:
                    self._parse_coordinate(self._point_cell_text(row, col))
                except ValueError:
                    errors.append(f"Row {row + 1}: {label} must be numeric.")
                    self._mark_invalid_cell(row, col)

        self.update_point_preview(errors)
        if show_message:
            if errors:
                QMessageBox.warning(self, "Point Validation",
                                    "\n".join(errors[:12]))
            else:
                QMessageBox.information(
                    self, "Point Validation", "Point table is valid.")
        return not errors

    def point_table_to_points(self):
        """Convert the editable table into the point dictionaries used by processing."""
        points = []
        order_index = self._current_coordinate_order()
        for row in range(self.point_config_table.rowCount()):
            control_item = self.point_config_table.item(row, 5)
            icp_item = self.point_config_table.item(row, 6)
            point_name = self._point_cell_text(row, 0)
            point_code = self._point_cell_text(row, 4)
            role_from_code = point_role(point_name, point_code)
            x = self._point_cell_text(row, 1)
            y = self._point_cell_text(row, 2)
            z = self._point_cell_text(row, 3)
            # Apply coordinate order mapping from GUI setting
            x_val, y_val, z_val = self._apply_coordinate_order(
                x, y, z, order_index)
            is_icp = bool(icp_item and icp_item.checkState() ==
                          Qt.Checked) or role_from_code == "icp"
            is_control = (
                bool(control_item and control_item.checkState() == Qt.Checked)
                or role_from_code == "control"
            ) and not is_icp
            points.append({
                'name': point_name,
                'x': x_val,
                'y': y_val,
                'z': z_val,
                'point_code': point_code or ("ICP" if is_icp else ("CONTROL" if is_control else "SURVEY")),
                'is_control_point': is_control,
                'is_icp': is_icp,
                'comment': self._point_cell_text(row, 7)
            })
        return points

    def commit_point_table_to_current_station(self, show_message=True):
        return self._commit_point_table_to_station(
            self.point_station_combo.currentText(), show_message=show_message)

    def _commit_point_table_to_station(self, station_name, show_message=True):
        """Write the visible point table back into the loaded station structure."""
        if self.point_config_table.rowCount() == 0:
            return True
        if not self.validate_point_table(show_message=show_message):
            return False

        points = self.point_table_to_points()
        if not hasattr(self.survey_data, 'stations'):
            self.survey_data.stations = {}
        station_name = station_name or "Manual Station"
        existing = self.survey_data.stations.get(station_name, {})
        # Store coordinate order for consistency in later reads
        coord_order = self.coordinate_order_combo.currentText()
        self.survey_data.stations[station_name] = {
            **existing,
            'name': station_name,
            'number': existing.get('number', len(self.survey_data.stations) + 1),
            'coords': np.array([[p['x'], p['y'], p['z']] for p in points], dtype=float),
            'all_ids': [p['name'] for p in points],
            'ctrl_ids': [p['name'] for p in points if p['is_control_point'] and not p['is_icp']],
            'icp_ids': [p['name'] for p in points if p['is_icp']],
            'point_codes': {p['name']: p['point_code'] for p in points},
            'point_comments': {p['name']: p['comment'] for p in points if p['comment']},
            'coordinate_order': coord_order,
        }
        self.data_type = "station_data"
        self.point_table_dirty = False
        self.update_tables()
        self.update_statistics()
        self._update_data_type_display()
        if show_message:
            QMessageBox.information(
                self, "Station Updated", f"Point table applied to {station_name}.")
        return True

    def prepare_points_for_processing(self):
        """Validate and synchronize point edits before running a method."""
        if not hasattr(self, 'point_config_table') or self.point_config_table.rowCount() == 0:
            return True
        return self.commit_point_table_to_current_station(show_message=False)

    def update_point_preview(self, validation_errors=None):
        if not hasattr(self, 'point_preview_table'):
            return
        validation_errors = validation_errors or []
        self.point_preview_table.setRowCount(
            self.point_config_table.rowCount())
        for row in range(self.point_config_table.rowCount()):
            name = self._point_cell_text(row, 0) or f"Row {row + 1}"
            control_item = self.point_config_table.item(row, 5)
            icp_item = self.point_config_table.item(row, 6)
            is_control = bool(
                control_item and control_item.checkState() == Qt.Checked)
            is_icp = bool(icp_item and icp_item.checkState() == Qt.Checked)
            role = 'Independent Check' if is_icp else (
                'Control Point' if is_control else 'Observed Point')
            source = control_item.data(
                Qt.UserRole) if control_item else 'empty'
            comment = self._point_cell_text(row, 7)
            row_errors = [
                err for err in validation_errors if f"Row {row + 1}:" in err]
            status = "Needs review" if row_errors else "Ready"
            values = [name, role, str(
                source or 'empty').title(), comment, status]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 1 and is_icp:
                    item.setBackground(QColor('#E1BEE7'))
                elif col == 1 and is_control:
                    item.setBackground(QColor('#BBDEFB'))
                elif col == 4 and status == "Needs review":
                    item.setBackground(QColor('#FFCDD2'))
                elif col == 4:
                    item.setBackground(QColor('#C8E6C9'))
                self.point_preview_table.setItem(row, col, item)
        self.point_preview_table.resizeColumnsToContents()

    def _create_point_configuration_tab(self):
        """Create the visual point table used before processing."""
        tab = QWidget()
        layout = QVBoxLayout()

        station_group = QGroupBox("Station and Control Point Rule")
        station_layout = QVBoxLayout()

        station_row = QHBoxLayout()
        station_row.addWidget(QLabel("Station:"))
        self.point_station_combo = QComboBox()
        self.point_station_combo.currentTextChanged.connect(
            self._on_point_station_changed)
        station_row.addWidget(self.point_station_combo, 1)

        self.apply_table_btn = QPushButton("Apply Table to Station")
        self.apply_table_btn.clicked.connect(
            lambda: self.commit_point_table_to_current_station())
        station_row.addWidget(self.apply_table_btn)
        station_layout.addLayout(station_row)

        rule_layout = QHBoxLayout()
        self.control_rule_enabled_check = QCheckBox(
            "Auto-detect control points")
        self.control_rule_enabled_check.setChecked(True)
        rule_layout.addWidget(self.control_rule_enabled_check)

        rule_layout.addWidget(QLabel("Prefixes:"))
        self.control_prefix_input = QLineEdit("CP, CTRL")
        self.control_prefix_input.setToolTip(
            "Comma, semicolon, or space separated prefixes. Matching ignores case.")
        rule_layout.addWidget(self.control_prefix_input, 1)

        self.control_rule_mode_combo = QComboBox()
        self.control_rule_mode_combo.addItems([
            "Only empty or automatic flags",
            "Overwrite all flags"
        ])
        rule_layout.addWidget(self.control_rule_mode_combo)
        station_layout.addLayout(rule_layout)

        station_group.setLayout(station_layout)
        layout.addWidget(station_group)

        action_layout = QHBoxLayout()
        edit_layout = QHBoxLayout()
        self.add_point_btn = QPushButton("＋ Add Point")
        self.add_point_btn.clicked.connect(lambda: self.add_point_row())
        self.add_point_btn.setMinimumHeight(30)
        self.add_point_btn.setMinimumWidth(115)
        edit_layout.addWidget(self.add_point_btn)

        self.remove_point_btn = QPushButton("− Remove Selected")
        self.remove_point_btn.clicked.connect(
            lambda: self.remove_selected_points())
        self.remove_point_btn.setMinimumHeight(30)
        self.remove_point_btn.setMinimumWidth(145)
        edit_layout.addWidget(self.remove_point_btn)

        self.clear_points_btn = QPushButton("⌫ Clear Table")
        self.clear_points_btn.clicked.connect(lambda: self.clear_point_table())
        self.clear_points_btn.setMinimumHeight(30)
        self.clear_points_btn.setMinimumWidth(120)
        edit_layout.addWidget(self.clear_points_btn)

        validate_layout = QHBoxLayout()
        self.apply_rule_btn = QPushButton("◎ Apply Control Rule")
        self.apply_rule_btn.clicked.connect(
            lambda: self.apply_control_point_rule())
        self.apply_rule_btn.setMinimumHeight(30)
        self.apply_rule_btn.setMinimumWidth(155)
        validate_layout.addWidget(self.apply_rule_btn)

        self.validate_points_btn = QPushButton("✓ Validate Points")
        self.validate_points_btn.clicked.connect(
            lambda: self.validate_point_table())
        self.validate_points_btn.setMinimumHeight(30)
        self.validate_points_btn.setMinimumWidth(140)
        self.validate_points_btn.setStyleSheet(
            "font-weight: bold; color: #1976D2;")
        validate_layout.addWidget(self.validate_points_btn)
        action_layout.addLayout(edit_layout)
        action_layout.addStretch(1)
        action_layout.addLayout(validate_layout)
        layout.addLayout(action_layout)

        self.point_config_table = QTableWidget()
        self.point_config_table.setColumnCount(8)
        self.point_config_table.setHorizontalHeaderLabels([
            "Point Name", "X", "Y", "Z", "Point Code",
            "Is Control Point", "ICP Check", "Comment"
        ])
        self.point_config_table.setAlternatingRowColors(True)
        self.point_config_table.setMinimumHeight(330)
        self.point_config_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded)
        self.point_config_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded)
        self.point_config_table.itemChanged.connect(
            self._on_point_table_item_changed)
        layout.addWidget(self.point_config_table, 3)

        preview_group = QGroupBox("Validation Preview")
        preview_layout = QVBoxLayout()
        self.point_preview_table = QTableWidget()
        self.point_preview_table.setColumnCount(5)
        self.point_preview_table.setHorizontalHeaderLabels([
            "Point", "Role", "Classification", "Comment", "Status"
        ])
        self.point_preview_table.setAlternatingRowColors(True)
        self.point_preview_table.setMinimumHeight(160)
        self.point_preview_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded)
        self.point_preview_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded)
        preview_layout.addWidget(self.point_preview_table)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group, 1)

        tab.setLayout(layout)
        return tab

    def _selected_station_delimiter(self):
        text = self.station_delimiter_combo.currentText()
        if text.startswith("Auto"):
            return "auto"
        if text.startswith("Semicolon"):
            return "semicolon"
        if text.startswith("Comma"):
            return "comma"
        if text.startswith("Space"):
            return "space"
        return "tab"

    def _current_coordinate_order(self):
        """Get current coordinate order mapping from combo box selection."""
        order = self.coordinate_order_combo.currentText()
        mapping = {"X,Y,Z": 0, "Y,X,Z": 1, "X,Z,Y": 2,
                   "Z,X,Y": 3, "Y,Z,X": 4, "Z,Y,X": 5}
        return mapping.get(order, 0)

    def _apply_coordinate_order(self, x, y, z, order_index):
        """Apply coordinate order mapping based on GUI selection.

        Input columns are always X,Y,Z (cols 1,2,3 of the table).
        This maps them to the internal representation based on the order setting.
        """
        if order_index == 0:  # X,Y,Z - no change
            return float(x), float(y), float(z)
        elif order_index == 1:  # Y,X,Z - swap X and Y
            return float(y), float(x), float(z)
        elif order_index == 2:  # X,Z,Y - swap Y and Z
            return float(x), float(z), float(y)
        elif order_index == 3:  # Z,X,Y - rotate left
            return float(z), float(x), float(y)
        elif order_index == 4:  # Y,Z,X - rotate right
            return float(y), float(z), float(x)
        elif order_index == 5:  # Z,Y,X - reverse
            return float(z), float(y), float(x)
        return float(x), float(y), float(z)

    def load_station_folder(self):
        """Load data from station folder with automatic format detection."""
        folder_name = QFileDialog.getExistingDirectory(
            self, "Select Station Folder", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if folder_name:
            try:
                # Validate folder exists and has files
                folder_path = Path(folder_name)
                txt_files = list(folder_path.glob("*.TXT")) + \
                    list(folder_path.glob("*.txt"))

                if not txt_files:
                    QMessageBox.warning(self, "No Station Files",
                                        f"No .TXT files found in the selected folder:\n{folder_name}\n\n"
                                        "Please select a folder containing station files with .TXT extension.")
                    return

                # Check if files have valid content
                if len(txt_files) == 0:
                    QMessageBox.warning(self, "Empty Folder",
                                        "The selected folder appears to be empty or contains no valid station files.")
                    return

                reader = DataManager()

                # Show progress message
                QMessageBox.information(self, "Loading Data",
                                        f"Loading station data from {len(txt_files)} files...\n"
                                        "This may take a few moments for large datasets.")

                # Load station data from folder
                self.survey_data = reader.load_station_folder(
                    folder_name,
                    delimiter=self._selected_station_delimiter(),
                    coordinate_order=self.coordinate_order_combo.currentText(),
                )
                self.data_type = "station_data"
                self.station_folder_path = folder_name

                # Validate loaded data
                if not hasattr(self.survey_data, 'stations') or not self.survey_data.stations:
                    QMessageBox.warning(self, "No Valid Data",
                                        "No valid station data could be loaded from the selected files.\n"
                                        "Please check that the files contain properly formatted coordinate data.")
                    return

                # Show detailed success message
                stations = getattr(self.survey_data, 'stations', {})
                control_points = getattr(
                    self.survey_data, 'control_points', {})
                measurements = len(self.survey_data.measurements)

                success_msg = QMessageBox()
                success_msg.setIcon(QMessageBox.Information)
                success_msg.setWindowTitle("Station Data Loaded Successfully")
                success_msg.setText(
                    "Station data has been loaded and processed successfully!")
                success_msg.setDetailedText(
                    f"Folder: {folder_name}\n"
                    f"Stations: {len(stations)}\n"
                    f"Control Points: {len(control_points)}\n"
                    f"Measurements: {measurements}\n"
                    f"Files Processed: {len(txt_files)}\n\n"
                    "The data is now ready for adjustment calculations."
                )
                success_msg.exec()

                self.update_tables()
                self.update_statistics()
                self._update_data_type_display()
                self.refresh_point_station_selector()

                self.data_loaded_signal.emit(True)

            except FileNotFoundError as e:
                QMessageBox.critical(self, "File Error",
                                     f"The selected folder or files could not be found:\n{str(e)}")
            except PermissionError as e:
                QMessageBox.critical(self, "Permission Error",
                                     f"Permission denied accessing the selected folder:\n{str(e)}")
            except ValueError as e:
                QMessageBox.critical(self, "Data Format Error",
                                     f"The station files contain invalid data format:\n{str(e)}\n\n"
                                     "Please check that the files contain properly formatted coordinate data.")
            except Exception as e:
                QMessageBox.critical(self, "Unexpected Error",
                                     f"An unexpected error occurred while loading station data:\n{str(e)}\n\n"
                                     "Please check the file format and try again.")

    def clear_data(self):
        """Clear all data."""
        self.survey_data = SurveyData()
        self.data_type = "unknown"
        self.station_folder_path = ""
        # Reset table to default measurement format
        self.measurement_table.setColumnCount(8)
        self.measurement_table.setHorizontalHeaderLabels([
            "From Point", "To Point", "Hz Angle", "V Angle",
            "Distance", "Inst Height", "Target Height"
        ])
        self.update_tables()
        self.update_statistics()
        self.refresh_point_station_selector()

        self.data_loaded_signal.emit(False)

    def update_tables(self):
        """Update the data table with current station data."""
        if self.data_type == "station_data":
            self._update_station_data_table()
        else:
            self._update_measurements_table()

    def _update_measurements_table(self):
        """Update table for traditional angle/distance measurements."""
        # Setup table headers for measurements
        self.measurement_table.setColumnCount(8)
        self.measurement_table.setHorizontalHeaderLabels([
            "From Point", "To Point", "Hz Angle", "V Angle",
            "Distance", "Inst Height", "Target Height"
        ])

        # Update measurements table
        self.measurement_table.setRowCount(len(self.survey_data.measurements))
        for row, measurement in enumerate(self.survey_data.measurements):
            self.measurement_table.setItem(
                row, 0, QTableWidgetItem(measurement['from_point']))
            self.measurement_table.setItem(
                row, 1, QTableWidgetItem(measurement['to_point']))
            self.measurement_table.setItem(row, 2, QTableWidgetItem(
                f"{np.degrees(measurement['angle_hz']):.4f}"))
            self.measurement_table.setItem(row, 3, QTableWidgetItem(
                f"{np.degrees(measurement['angle_v']):.4f}"))
            self.measurement_table.setItem(
                row, 4, QTableWidgetItem(f"{measurement['distance']:.4f}"))
            self.measurement_table.setItem(row, 5, QTableWidgetItem(
                f"{measurement['height_instrument']:.4f}"))
            self.measurement_table.setItem(row, 6, QTableWidgetItem(
                f"{measurement['height_target']:.4f}"))

    def _update_station_data_table(self):
        """Update table for station data (control points and survey points from multiple stations)."""
        # Setup table headers for station data
        self.measurement_table.setColumnCount(7)
        self.measurement_table.setHorizontalHeaderLabels([
            "Point ID", "Type", "Station", "X", "Y", "Z", "Status"
        ])

        # Collect all points from all stations
        all_points = []

        if hasattr(self.survey_data, 'stations'):
            for station_name, station_data in self.survey_data.stations.items():
                if 'coords' in station_data:
                    coords_arr = station_data['coords']
                    all_ids = station_data.get('all_ids', [])
                    ctrl_ids = set(station_data.get('ctrl_ids', []))
                    icp_ids = set(station_data.get('icp_ids', []))

                    for i, point_id in enumerate(all_ids):
                        if i < len(coords_arr):
                            coords = coords_arr[i]
                            is_ctrl = point_id in ctrl_ids
                            is_icp = point_id in icp_ids
                            all_points.append({
                                'id': point_id,
                                'type': 'ICP' if is_icp else ('Control' if is_ctrl else 'Survey'),
                                'station': station_name,
                                'x': coords[0],
                                'y': coords[1],
                                'z': coords[2],
                                'status': 'Independent Check' if is_icp else ('Fixed' if is_ctrl else 'To Adjust')
                            })

        # Sort by station, then by type, then by point ID
        all_points.sort(key=lambda p: (p['station'], p['type'], p['id']))

        # Update station data table
        self.measurement_table.setRowCount(len(all_points))
        for row, point in enumerate(all_points):
            # Point ID
            id_item = QTableWidgetItem(point['id'])
            if point['type'].startswith('Control'):
                # Light blue for control points
                id_item.setBackground(QColor('#E3F2FD'))
            else:
                # Light orange for survey points
                id_item.setBackground(QColor('#FFF3E0'))
            self.measurement_table.setItem(row, 0, id_item)

            # Type
            type_item = QTableWidgetItem(point['type'])
            self.measurement_table.setItem(row, 1, type_item)

            # Station
            station_item = QTableWidgetItem(point['station'])
            station_item.setBackground(QColor('#F5F5F5'))  # Light gray
            self.measurement_table.setItem(row, 2, station_item)

            # Coordinates
            self.measurement_table.setItem(
                row, 3, QTableWidgetItem(f"{point['x']:.4f}"))
            self.measurement_table.setItem(
                row, 4, QTableWidgetItem(f"{point['y']:.4f}"))
            self.measurement_table.setItem(
                row, 5, QTableWidgetItem(f"{point['z']:.4f}"))

            # Status
            status_item = QTableWidgetItem(point['status'])
            if point['status'] == 'Independent Check':
                status_item.setBackground(QColor('#E1BEE7'))
            elif point['status'] == 'Fixed':
                status_item.setBackground(QColor('#C8E6C9'))  # Light green
            else:
                status_item.setBackground(QColor('#FFECB3'))  # Light yellow
            self.measurement_table.setItem(row, 6, status_item)

        # Resize columns to fit content
        self.measurement_table.resizeColumnsToContents()
        # Set minimum column widths
        self.measurement_table.setColumnWidth(
            0, max(80, self.measurement_table.columnWidth(0)))  # Point ID
        self.measurement_table.setColumnWidth(
            1, max(100, self.measurement_table.columnWidth(1)))  # Type
        self.measurement_table.setColumnWidth(
            2, max(80, self.measurement_table.columnWidth(2)))  # Station

    def _update_data_type_display(self):
        """Update the interface to reflect the current data type."""
        if self.data_type == "station_data":
            self.stats_group.setTitle("Station Data Statistics")
            self.data_type_label.setText("Data Type: Station Coordinates")
            self.data_type_label.setStyleSheet(
                "font-weight: bold; color: #FF8C00;")  # Orange for station data
            stations = getattr(self.survey_data, 'stations', {})
            total_survey_points = 0
            for data in stations.values():
                if 'all_ids' in data:
                    total_survey_points += len(data['all_ids']) - \
                        len(data.get('ctrl_ids', []))
                else:
                    total_survey_points += len(data.get('survey_points', {}))
            self.survey_points_label.setText(
                f"Stations: {len(stations)} | Survey Points: {total_survey_points}")
            self.measurement_count_label.setText(
                f"Measurements: {len(self.survey_data.measurements)}")
            source = self.station_folder_path or "not loaded"
            self.source_folder_label.setText(f"Source: {source}")
        elif self.data_type == "measurements":
            self.stats_group.setTitle("Survey Measurements Statistics")
            self.data_type_label.setText("Data Type: Angles & Distances")
            self.data_type_label.setStyleSheet(
                "font-weight: bold; color: #4169E1;")  # Blue for measurements
            self.survey_points_label.setText(
                f"Survey Points: {len(self.survey_data.all_points)}")
            self.measurement_count_label.setText(
                f"Measurements: {len(self.survey_data.measurements)}")
            self.source_folder_label.setText("Source: measurement dataset")
        else:
            self.stats_group.setTitle("Survey Data Statistics")
            self.data_type_label.setText("Data Type: Unknown")
            self.data_type_label.setStyleSheet(
                "font-weight: bold; color: #666;")
            self.survey_points_label.setText("Survey Points: 0")
            self.measurement_count_label.setText("Measurements: 0")
            self.source_folder_label.setText("Source: not loaded")

    def update_statistics(self):
        """Update the TFS statistics display."""
        self._update_data_type_display()


class ResultsWidget(QWidget):
    """Widget for displaying adjustment results."""

    def __init__(self):
        super().__init__()
        self.results = {}
        self.current_audit_output_dir = None
        self.current_audit_method_key = None
        self.init_ui()

    def init_ui(self):
        """Initialize the results UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.results_tabs = QTabWidget()
        self.results_tabs.setMinimumWidth(820)
        self.results_tabs.setDocumentMode(True)
        self.results_tabs.setUsesScrollButtons(True)
        self.results_tabs.setElideMode(Qt.ElideRight)
        self.results_tabs.setStyleSheet("""
            QTabWidget::pane { border-top: 1px solid #d0d0d0; }
            QTabBar::tab {
                min-width: 76px;
                min-height: 24px;
                padding: 3px 8px;
                font-size: 11px;
            }
        """)
        self.current_uncertainties = {}

        self.results_side_panel = QScrollArea()
        self.results_side_panel.setWidgetResizable(True)
        self.results_side_panel.setFrameShape(QScrollArea.NoFrame)
        self.results_side_panel.setMinimumWidth(210)
        self.results_side_panel.setMaximumWidth(285)
        self.results_side_panel.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding)

        self.results_splitter = QSplitter(Qt.Horizontal)
        self.results_splitter.setChildrenCollapsible(False)
        self.results_splitter.addWidget(self.results_side_panel)
        self.results_splitter.addWidget(self.results_tabs)
        self.results_splitter.setStretchFactor(0, 0)
        self.results_splitter.setStretchFactor(1, 1)
        self.results_splitter.setSizes([235, 1180])

        self.clear_results()
        layout.addWidget(self.results_splitter, 1)

        export_group = QGroupBox("Export and review")
        export_layout = QHBoxLayout()
        export_layout.setContentsMargins(8, 6, 8, 6)
        export_layout.setSpacing(6)

        self.export_coords_btn = QPushButton("Export Coordinates")
        self.export_coords_btn.setMinimumHeight(32)
        self.export_coords_btn.setMinimumWidth(140)
        self.export_coords_btn.clicked.connect(self.export_coordinates)
        export_layout.addWidget(self.export_coords_btn)

        self.export_dxf_btn = QPushButton("Export DXF")
        self.export_dxf_btn.setMinimumHeight(32)
        self.export_dxf_btn.setMinimumWidth(115)
        self.export_dxf_btn.setToolTip(
            "Export transformed control and irradiated points to DXF layers.")
        self.export_dxf_btn.clicked.connect(self.export_dxf)
        export_layout.addWidget(self.export_dxf_btn)

        self.export_matrices_btn = QPushButton("Export Matrix")
        self.export_matrices_btn.setMinimumHeight(32)
        self.export_matrices_btn.setMinimumWidth(125)
        self.export_matrices_btn.setToolTip(
            "Choose one available matrix from the current result and export it as CSV.")
        self.export_matrices_btn.clicked.connect(self.export_selected_matrix)
        export_layout.addWidget(self.export_matrices_btn)

        self.open_html_btn = QPushButton("Open HTML Output")
        self.open_html_btn.setMinimumHeight(32)
        self.open_html_btn.setMinimumWidth(150)
        self.open_html_btn.setToolTip(
            "Open the selected-method 3D or RMSE HTML output when available.")
        self.open_html_btn.clicked.connect(self.open_primary_html_output)
        export_layout.addWidget(self.open_html_btn)

        self.focused_review_btn = QPushButton("Focused Review")
        self.focused_review_btn.setMinimumHeight(32)
        self.focused_review_btn.setMinimumWidth(135)
        self.focused_review_btn.setToolTip(
            "Open a compact review window with only the key result panels.")
        self.focused_review_btn.clicked.connect(
            self.open_focused_results_review_dialog)
        export_layout.addWidget(self.focused_review_btn)

        export_layout.addStretch(1)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        self.setLayout(layout)

    def _message_widget(self, message):
        widget = QWidget()
        layout = QVBoxLayout()
        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        layout.addWidget(label, 1)
        widget.setLayout(layout)
        return widget

    def _set_summary_panel(self, widget):
        if widget is None:
            widget = self._message_widget(
                "Run a processing method to view its available outputs.")
        self.results_side_panel.setWidget(widget)

    def _polish_table(self, table, stretch_last=True):
        """Apply consistent spacing and resizing to result tables."""
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setSortingEnabled(False)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(24)
        table.horizontalHeader().setStretchLastSection(stretch_last)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def _selected_analyzer_summary(self, results):
        output_dir = results.get("advanced_output_dir")
        method_key = results.get("advanced_method_key")
        if not output_dir or not method_key:
            return {}
        return load_method_summary(
            output_dir,
            method_key,
            prefer_removed=bool(results.get("outlier_removal_applied")),
        )

    def _results_summary_widget(self, results, stations_data):
        if not results:
            return None

        stations = list(stations_data or [])
        adjustable = [s for s in stations if not s.get('is_reference')]
        successful = [
            s for s in adjustable
            if s.get('coords') is not None or s.get('success')
        ]
        failed = [
            s for s in adjustable
            if s.get('coords') is None and not s.get('success')
        ]

        rmse_values = [
            float(s.get('rmse')) for s in stations
            if self._is_valid_number(s.get('rmse'))
        ]
        scale_values = []
        for station in stations:
            metrics = self._safe_dict(station.get('quality_metrics'))
            scale = station.get('scale', metrics.get('scale'))
            transform = station.get('transform')
            if scale is None and isinstance(transform, tuple) and transform:
                scale = transform[0]
            if self._is_valid_number(scale):
                scale_values.append(float(scale))

        coords_count = len(
            self._extract_coords_from_transformed_stations(stations))
        cov = self._matrix_to_array(results.get('covariance'))
        cov_trace = np.trace(
            cov) if cov is not None and cov.ndim == 2 else None
        analyzer_summary = self._selected_analyzer_summary(results)
        mean_rmse = analyzer_summary.get(
            "mean_rmse",
            results.get("mean_rmse", results.get(
                "rmse", np.mean(rmse_values) if rmse_values else None)),
        )
        max_rmse = analyzer_summary.get(
            "max_residual",
            analyzer_summary.get(
                "max_rmse",
                results.get("max_rmse", np.max(rmse_values)
                            if rmse_values else None),
            ),
        )

        rows = [
            ("Processing method", results.get('method', 'N/A')),
            ("Registration type", results.get(
                'registration_type', results.get('type', 'N/A'))),
            (
                "Outlier handling",
                "automatic removal in analyzer outputs" if results.get(
                    "outlier_removal_applied") else "review only",
            ),
            ("Stations adjusted", f"{len(successful)} / {len(adjustable)}"),
            ("Stations failed", len(failed)),
            ("Adjusted points shown", coords_count),
            ("Mean RMSE (m)", mean_rmse),
            ("Max point RMSE (m)", max_rmse),
            ("ICP RMSE (m)", analyzer_summary.get("icp_rmse")),
            ("ICP max residual (m)", analyzer_summary.get("icp_max_residual")),
            ("ICP points", analyzer_summary.get("icp_points")),
            ("Mean scale", np.mean(scale_values) if scale_values else None),
            ("Scale standard deviation", np.std(
                scale_values) if scale_values else None),
            ("Sigma0", results.get('sigma0')),
            ("Covariance trace", cov_trace),
        ]
        if results.get('advanced_html_path'):
            rows.append(
                ("Embedded 3D HTML", Path(results['advanced_html_path']).name))
        if results.get('advanced_rmse_html_path'):
            rows.append(("RMSE-colored 3D HTML",
                        Path(results['advanced_rmse_html_path']).name))

        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        title = QLabel("Results")
        title.setStyleSheet(
            "font-weight: bold; color: #202124; font-size: 15px;")
        layout.addWidget(title)

        summary_group = QGroupBox("Processing summary")
        summary_layout = QFormLayout()
        summary_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        summary_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        summary_layout.setLabelAlignment(Qt.AlignLeft)
        summary_layout.setFormAlignment(Qt.AlignTop)
        summary_layout.setHorizontalSpacing(10)
        summary_layout.setVerticalSpacing(5)
        for name, value in rows:
            key_label = QLabel(str(name))
            key_label.setWordWrap(True)
            value_label = QLabel(self._format_cell(value))
            value_label.setWordWrap(True)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value_label.setStyleSheet(
                "font-family: Consolas, monospace; color: #202124;")
            summary_layout.addRow(key_label, value_label)
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        if failed:
            failed_group = QGroupBox("Stations needing attention")
            failed_layout = QVBoxLayout()
            failed_names = [
                s.get('name', s.get('station', 'Station'))
                for s in failed[:10]
            ]
            failed_label = QLabel(", ".join(failed_names))
            failed_label.setWordWrap(True)
            failed_layout.addWidget(failed_label)
            failed_group.setLayout(failed_layout)
            layout.addWidget(failed_group)

        actions_group = QGroupBox("Article outputs")
        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(6)
        has_action = False
        if results.get('advanced_html_path'):
            open_btn = QPushButton("Show selected-method 3D HTML")
            open_btn.setMinimumHeight(28)
            open_btn.clicked.connect(
                lambda: self._show_embedded_html_dialog(
                    results['advanced_html_path'])
            )
            actions_layout.addWidget(open_btn)
            has_action = True
        if results.get('advanced_rmse_html_path'):
            open_rmse_btn = QPushButton("Show selected-method RMSE HTML")
            open_rmse_btn.setMinimumHeight(28)
            open_rmse_btn.clicked.connect(
                lambda: self._show_embedded_html_dialog(
                    results['advanced_rmse_html_path'])
            )
            actions_layout.addWidget(open_rmse_btn)
            has_action = True
        if not has_action:
            actions_note = QLabel(
                "Selected-method HTML outputs will appear here when enabled.")
            actions_note.setWordWrap(True)
            actions_layout.addWidget(actions_note)
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        layout.addStretch(1)

        widget.setLayout(layout)
        return widget

    def clear_results(self):
        """Clear previous method outputs before a new processing run."""
        self.results = {}
        self.current_uncertainties = {}
        self.results_tabs.clear()
        if hasattr(self, 'audit_widget'):
            self.audit_widget = None
        if hasattr(self, 'results_side_panel'):
            self._set_summary_panel(
                self._message_widget(
                    "Run a processing method to view its available outputs.")
            )
        self.results_tabs.addTab(
            self._message_widget(
                "Run a processing method to view its available outputs."),
            "No Results",
        )

    def _safe_dict(self, value):
        """Return a dictionary-like value or an empty dictionary."""
        return value if isinstance(value, dict) else {}

    def _is_valid_number(self, value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        return np.isfinite(numeric)

    def _format_number(self, value, decimals=6):
        if not self._is_valid_number(value):
            return "N/A"
        return f"{float(value):.{decimals}f}"

    def _first_defined(self, *values):
        """Return the first value that was actually provided."""
        for value in values:
            if value is not None:
                return value
        return None

    def _station_display_name(self, station):
        if not isinstance(station, dict):
            return "Station"
        return station.get('name') or station.get('station') or "Station"

    def setup_coordinates_table(self, table=None):
        """Setup the coordinates table."""
        table = table or self.coordinates_table
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels(
            ["Point ID", "X", "Y", "Z", "Sigma X", "Sigma Y", "Sigma Z"])
        table.setAlternatingRowColors(True)

    def setup_residuals_table(self):
        """Setup the residuals table."""
        self.residuals_table.setColumnCount(7)
        self.residuals_table.setHorizontalHeaderLabels(
            ["Observation ID", "Observed Value", "Adjusted Value",
             "Residual (v)", "Normalized Res.", "Std Dev", "Info"])
        self.residuals_table.setAlternatingRowColors(True)

    def display_results(self, results: Dict):
        """Display the adjustment results."""
        self.results = results
        self.results_tabs.clear()

        # Adjusted coordinates with available standard deviations.
        final_uncertainties = {}
        self.current_uncertainties = {}

        stations_data = results.get(
            'transformed_stations', results.get('stations')) or []

        summary_widget = self._results_summary_widget(results, stations_data)
        self._set_summary_panel(summary_widget)

        # Statistics first: this is the main article-facing summary.
        if 'registration_type' in results:
            stats_text = self._format_network_adjustment_stats(
                results) if results['registration_type'] == 'network' else self._format_sequential_registration_stats(results)
            self.stats_text = QTextEdit()
            self.stats_text.setReadOnly(True)
            self.stats_text.setText(stats_text)
            self.results_tabs.addTab(self.stats_text, "Quality")

        method_outputs = self._method_outputs_widget(results)
        if method_outputs is not None:
            self.results_tabs.addTab(method_outputs, "Stations")

        if stations_data:  # For sequential/network
            coords = self._extract_coords_from_transformed_stations(
                stations_data)

            for station_data in stations_data:
                if 'point_uncertainties' in station_data:
                    station_name = station_data.get(
                        'name', station_data.get('station', 'Unknown'))
                    for pt_id, unc in station_data['point_uncertainties'].items():
                        if isinstance(unc, np.ndarray) and unc.shape == (3, 3):
                            unique_id = f"{pt_id} ({station_name})"
                            # Store the 3x3 covariance matrix
                            final_uncertainties[unique_id] = unc

            self.current_uncertainties = final_uncertainties
            original_coords = self._extract_coords_from_original_stations(
                stations_data)
            if original_coords:
                initial_table = self._coordinates_table_widget(
                    original_coords,
                    {},
                )
                self.results_tabs.addTab(initial_table, "Coordinates Initial")

            if coords:
                self.coordinates_table = self._coordinates_table_widget(
                    coords,
                    final_uncertainties,
                )
                final_name = "Coordinates Final" if original_coords else "Coordinates"
                self.results_tabs.addTab(self.coordinates_table, final_name)

        if self.results_tabs.count() == 0:
            self.results_tabs.addTab(
                self._message_widget(
                    "This method finished, but it did not return displayable outputs."),
                "No Outputs",
            )

    def _method_outputs_widget(self, results):
        transformed_stations = list(results.get('transformed_stations') or [])
        result_stations = list(results.get('stations') or [])
        stations = transformed_stations or result_stations
        result_by_name = {
            self._station_display_name(station): station
            for station in result_stations
            if isinstance(station, dict)
        }
        rows = []
        for station in stations:
            station_name = self._station_display_name(station)
            source_station = result_by_name.get(station_name, {})
            metrics = {
                **self._safe_dict(source_station.get('quality_metrics')),
                **self._safe_dict(station.get('quality_metrics')),
            }
            transform = self._first_defined(
                station.get('transform'), source_station.get('transform'))
            if transform is None:
                transform = (None, None, None)

            scale = self._first_defined(
                station.get('scale'), metrics.get('scale'), source_station.get('scale'))
            if scale is None and isinstance(transform, tuple) and len(transform) >= 1:
                scale = transform[0]

            rotation = self._first_defined(
                station.get('rotation'),
                station.get('rotation_matrix'),
                metrics.get('rotation_matrix'),
                source_station.get('rotation'),
                source_station.get('rotation_matrix'),
            )
            if rotation is None and isinstance(transform, tuple) and len(transform) >= 2:
                rotation = transform[1]
            rotation_arr = self._matrix_to_array(rotation)
            if rotation_arr is not None and rotation_arr.shape != (3, 3):
                rotation_arr = None

            translation = self._first_defined(
                station.get('translation'),
                metrics.get('translation'),
                source_station.get('translation'),
            )
            if translation is None and isinstance(transform, tuple) and len(transform) >= 3:
                translation = transform[2]

            rotation_det = self._first_defined(
                metrics.get('rotation_determinant'),
                station.get('rotation_determinant'),
                source_station.get('rotation_determinant'),
            )
            orthogonality_error = self._first_defined(
                metrics.get('orthogonality_error'),
                station.get('orthogonality_error'),
                source_station.get('orthogonality_error'),
            )
            if rotation_arr is not None:
                if rotation_det is None:
                    rotation_det = float(np.linalg.det(rotation_arr))
                if orthogonality_error is None:
                    orthogonality_error = float(
                        np.linalg.norm(rotation_arr.T @ rotation_arr - np.eye(3)))

            rmse = self._first_defined(
                station.get('rmse'), metrics.get('rmse'), source_station.get('rmse'))
            n_control = self._first_defined(
                metrics.get('n_control_points'),
                station.get('n_control_points'),
                station.get('n_points'),
                source_station.get('n_control_points'),
                source_station.get('n_points'),
            )
            if n_control is None and station.get('ctrl_ids') is not None:
                n_control = len(station.get('ctrl_ids') or [])
            n_icp = self._first_defined(
                station.get('n_icp_points'),
                source_station.get('n_icp_points'),
            )
            if n_icp is None and station.get('icp_ids') is not None:
                n_icp = len(station.get('icp_ids') or [])
            excluded = self._first_defined(
                station.get('excluded_outliers'),
                source_station.get('excluded_outliers'),
                metrics.get('excluded_outliers'),
            )
            if isinstance(excluded, (list, tuple, set)):
                excluded = ", ".join(str(item) for item in excluded)
            rows.append({
                "Station": station_name,
                "Status": "Reference" if station.get('is_reference') or source_station.get('is_reference') else ("Success" if station.get('coords') is not None or station.get('success') or source_station.get('success') else "Failed"),
                "RMSE (m)": rmse,
                "Mean residual (m)": metrics.get('mean_residual'),
                "Max residual (m)": metrics.get('max_residual'),
                "ICP RMSE (m)": self._first_defined(station.get('icp_rmse'), source_station.get('icp_rmse')),
                "Scale": scale,
                "Rotation det.": rotation_det,
                "Orthogonality error": orthogonality_error,
                "Control points": n_control,
                "ICP points": n_icp,
                "Time (s)": self._first_defined(station.get('time'), station.get('elapsed_time_solver'), metrics.get('elapsed_time_solver'), source_station.get('time'), source_station.get('elapsed_time_solver')),
                "Translation": self._format_vector(translation),
                "Excluded": excluded,
            })
        if not rows:
            return None
        table = QTableWidget()
        headers = list(rows[0].keys())
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            for col, header in enumerate(headers):
                table.setItem(row, col, QTableWidgetItem(
                    self._format_cell(item.get(header))))
        self._polish_table(table)
        table.resizeColumnsToContents()
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(table)
        widget.setLayout(layout)
        return widget

    def _format_cell(self, value):
        if value is None:
            return "N/A"
        try:
            if isinstance(value, float) or np.isscalar(value):
                numeric = float(value)
                if np.isnan(numeric):
                    return "N/A"
                return f"{numeric:.6g}"
        except Exception:
            pass
        return str(value)

    def _format_vector(self, value):
        if value is None:
            return "N/A"
        try:
            arr = np.asarray(value, dtype=float).ravel()
        except Exception:
            return str(value)
        if arr.size == 0:
            return "N/A"
        return ", ".join(f"{v:.6g}" for v in arr[:3])

    def _coordinates_table_widget(self, coords, uncertainties=None):
        table = QTableWidget()
        self.setup_coordinates_table(table)
        table.setRowCount(len(coords))
        uncertainties = uncertainties or {}
        for row, (point_id, coord) in enumerate(sorted(coords.items())):
            coord = np.asarray(coord, dtype=float)
            table.setItem(row, 0, QTableWidgetItem(str(point_id)))
            table.setItem(row, 1, QTableWidgetItem(f"{coord[0]:.4f}"))
            table.setItem(row, 2, QTableWidgetItem(f"{coord[1]:.4f}"))
            table.setItem(row, 3, QTableWidgetItem(f"{coord[2]:.4f}"))

            sigmas = self._point_sigmas_from_covariance(
                uncertainties.get(point_id))
            for col, value in enumerate(sigmas, start=4):
                text = f"{value:.4f}" if self._is_valid_number(
                    value) else "N/A"
                table.setItem(row, col, QTableWidgetItem(text))

        self._polish_table(table)
        table.resizeColumnsToContents()
        return table

    def _point_sigmas_from_covariance(self, covariance):
        if not isinstance(covariance, np.ndarray) or covariance.shape != (3, 3):
            return [None, None, None]
        diag = np.diag(covariance).astype(float)
        if np.any(~np.isfinite(diag)) or np.any(diag < 0):
            return [None, None, None]
        sigmas = np.sqrt(diag)
        # Values larger than this are usually parameter-covariance artifacts,
        # not reliable coordinate precision for a point.
        if np.any(~np.isfinite(sigmas)) or np.any(sigmas > 1.0):
            return [None, None, None]
        return sigmas.tolist()

    def _extract_coords_from_transformed_stations(self, transformed_stations):
        """Extracts point IDs and coordinates from the transformed_stations list."""
        all_coords = {}
        for station_data in transformed_stations:
            if station_data.get('coords') is not None:
                coords_array = station_data['coords']
                all_ids = station_data.get('all_ids', [])
                station_name = station_data.get(
                    'name', station_data.get('station', 'Unknown'))
                for i, point_id in enumerate(all_ids):
                    if i < len(coords_array):
                        unique_id = f"{point_id} ({station_name})"
                        all_coords[unique_id] = coords_array[i]
        return all_coords

    def _extract_coords_from_original_stations(self, transformed_stations):
        """Extracts point IDs and original coordinates from the transformed_stations list."""
        all_coords = {}
        for station_data in transformed_stations:
            if station_data.get('coords_original') is not None:
                coords_array = station_data['coords_original']
                all_ids = station_data.get('all_ids', [])
                station_name = station_data.get(
                    'name', station_data.get('station', 'Unknown'))
                for i, point_id in enumerate(all_ids):
                    if i < len(coords_array):
                        unique_id = f"{point_id} ({station_name})"
                        all_coords[unique_id] = coords_array[i]
        return all_coords

    def _collect_dxf_points(self, transformed_stations):
        """Collect transformed points with station and role metadata for CAD export."""
        points = []
        for station_data in transformed_stations or []:
            coords_array = station_data.get('coords')
            if coords_array is None:
                continue
            coords_array = np.asarray(coords_array, dtype=float)
            if coords_array.ndim != 2 or coords_array.shape[1] < 3:
                continue
            all_ids = list(station_data.get('all_ids', []))
            station_name = station_data.get(
                'name', station_data.get('station', 'Unknown'))
            ctrl_ids = {str(point_id) for point_id in station_data.get('ctrl_ids', [])}
            icp_ids = {str(point_id) for point_id in station_data.get('icp_ids', [])}
            point_codes = station_data.get('point_codes', {}) or {}

            for index, point_id in enumerate(all_ids):
                if index >= len(coords_array):
                    continue
                coord = np.asarray(coords_array[index, :3], dtype=float)
                if not np.all(np.isfinite(coord)):
                    continue
                point_id_text = str(point_id)
                role = point_role(
                    point_id_text,
                    point_codes.get(point_id, point_codes.get(point_id_text)),
                )
                is_control = (
                    point_id_text in ctrl_ids
                    or point_id_text in icp_ids
                    or role in {"control", "icp"}
                )
                points.append({
                    "point_id": point_id_text,
                    "station": str(station_name),
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "z": float(coord[2]),
                    "role": "control" if is_control else "irradiated",
                    "label": f"{point_id_text} ({station_name})",
                })
        return points

    def _dxf_export_options(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("DXF Export Options")
        dialog.setMinimumWidth(360)
        layout = QVBoxLayout()

        form = QFormLayout()
        symbol_combo = QComboBox()
        symbol_combo.addItem("CAD point", "point")
        symbol_combo.addItem("Cross", "cross")
        symbol_combo.addItem("Circle", "circle")
        symbol_combo.addItem("Square", "square")
        form.addRow("Point symbol:", symbol_combo)

        scale_spin = QDoubleSpinBox()
        scale_spin.setDecimals(3)
        scale_spin.setRange(0.001, 10000.0)
        scale_spin.setSingleStep(0.05)
        scale_spin.setValue(0.20)
        scale_spin.setSuffix(" m")
        form.addRow("Symbol scale:", scale_spin)

        label_check = QCheckBox("Export point labels")
        label_check.setChecked(True)
        form.addRow("Labels:", label_check)
        layout.addLayout(form)

        note = QLabel(
            "Control/ICP points and irradiated points will be exported on separate DXF layers."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_btn = QPushButton("Cancel")
        export_btn = QPushButton("Export")
        export_btn.setDefault(True)
        cancel_btn.clicked.connect(dialog.reject)
        export_btn.clicked.connect(dialog.accept)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(export_btn)
        layout.addLayout(buttons)

        dialog.setLayout(layout)
        if dialog.exec() != QDialog.Accepted:
            return None
        return {
            "symbol": symbol_combo.currentData(),
            "scale": float(scale_spin.value()),
            "labels": bool(label_check.isChecked()),
        }

    def _dxf_text(self, value):
        return str(value).replace("\r", " ").replace("\n", " ").strip()

    def _dxf_float(self, value):
        return f"{float(value):.6f}"

    def _dxf_pair(self, code, value):
        return [str(code), self._dxf_text(value)]

    def _dxf_layer(self, name, color):
        return [
            "0", "LAYER",
            "2", name,
            "70", "0",
            "62", str(color),
            "6", "CONTINUOUS",
        ]

    def _dxf_point_entity(self, layer, x, y, z):
        return [
            "0", "POINT",
            "8", layer,
            "10", self._dxf_float(x),
            "20", self._dxf_float(y),
            "30", self._dxf_float(z),
        ]

    def _dxf_line_entity(self, layer, start, end):
        return [
            "0", "LINE",
            "8", layer,
            "10", self._dxf_float(start[0]),
            "20", self._dxf_float(start[1]),
            "30", self._dxf_float(start[2]),
            "11", self._dxf_float(end[0]),
            "21", self._dxf_float(end[1]),
            "31", self._dxf_float(end[2]),
        ]

    def _dxf_circle_entity(self, layer, x, y, z, radius):
        return [
            "0", "CIRCLE",
            "8", layer,
            "10", self._dxf_float(x),
            "20", self._dxf_float(y),
            "30", self._dxf_float(z),
            "40", self._dxf_float(radius),
        ]

    def _dxf_text_entity(self, layer, x, y, z, height, label):
        return [
            "0", "TEXT",
            "8", layer,
            "10", self._dxf_float(x),
            "20", self._dxf_float(y),
            "30", self._dxf_float(z),
            "40", self._dxf_float(height),
            "1", self._dxf_text(label),
        ]

    def _dxf_symbol_entities(self, layer, x, y, z, symbol, scale):
        half = max(float(scale), 0.001) / 2.0
        if symbol == "point":
            return self._dxf_point_entity(layer, x, y, z)
        if symbol == "circle":
            return self._dxf_circle_entity(layer, x, y, z, half)
        if symbol == "square":
            corners = [
                (x - half, y - half, z),
                (x + half, y - half, z),
                (x + half, y + half, z),
                (x - half, y + half, z),
            ]
            lines = []
            for start, end in zip(corners, corners[1:] + corners[:1]):
                lines.extend(self._dxf_line_entity(layer, start, end))
            return lines
        lines = []
        lines.extend(self._dxf_line_entity(
            layer, (x - half, y, z), (x + half, y, z)))
        lines.extend(self._dxf_line_entity(
            layer, (x, y - half, z), (x, y + half, z)))
        return lines

    def _write_dxf_file(self, file_name, points, symbol, scale, include_labels):
        layers = {
            "CONTROL_POINTS": 1,
            "IRRADIATED_POINTS": 7,
            "CONTROL_LABELS": 3,
            "IRRADIATED_LABELS": 8,
        }
        lines = [
            "0", "SECTION", "2", "HEADER",
            "9", "$ACADVER", "1", "AC1009",
            "0", "ENDSEC",
            "0", "SECTION", "2", "TABLES",
            "0", "TABLE", "2", "LAYER", "70", str(len(layers)),
        ]
        for layer, color in layers.items():
            lines.extend(self._dxf_layer(layer, color))
        lines.extend(["0", "ENDTAB", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"])

        label_height = max(float(scale) * 0.70, 0.001)
        label_offset = max(float(scale) * 0.65, 0.001)
        for point in points:
            layer = "CONTROL_POINTS" if point["role"] == "control" else "IRRADIATED_POINTS"
            label_layer = "CONTROL_LABELS" if point["role"] == "control" else "IRRADIATED_LABELS"
            x, y, z = point["x"], point["y"], point["z"]
            lines.extend(self._dxf_symbol_entities(layer, x, y, z, symbol, scale))
            if include_labels:
                lines.extend(self._dxf_text_entity(
                    label_layer,
                    x + label_offset,
                    y + label_offset,
                    z,
                    label_height,
                    point["label"],
                ))

        lines.extend(["0", "ENDSEC", "0", "EOF"])
        Path(file_name).write_text("\n".join(lines) + "\n", encoding="ascii", errors="ignore")

    def export_dxf(self):
        """Export transformed control and irradiated points to a simple DXF file."""
        stations_data = self.results.get(
            'transformed_stations', self.results.get('stations'))
        points = self._collect_dxf_points(stations_data)
        if not points:
            QMessageBox.warning(
                self, "Warning", "No transformed points are available for DXF export.")
            return

        options = self._dxf_export_options()
        if not options:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export DXF", "", "DXF Files (*.dxf);;All Files (*)"
        )
        if not file_name:
            return
        if not file_name.lower().endswith(".dxf"):
            file_name += ".dxf"

        try:
            self._write_dxf_file(
                file_name,
                points,
                options["symbol"],
                options["scale"],
                options["labels"],
            )
            n_control = sum(1 for point in points if point["role"] == "control")
            n_irradiated = len(points) - n_control
            QMessageBox.information(
                self,
                "Success",
                "DXF exported successfully.\n\n"
                f"Control/ICP points: {n_control}\n"
                f"Irradiated points: {n_irradiated}",
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Error", f"Failed to export DXF: {str(exc)}")

    def _format_sequential_registration_stats(self, results):
        """Format statistics for Sequential Registration."""
        method_name = results.get('method', 'Sequential Registration')
        stats_text = f"""
{method_name} - Sequential Registration Summary
{'='*60}

Processing Order: {' -> '.join([s['name'] for s in results['stations']])}

Per-Station Results:
{'Station':<10} | {'Status':<10} | {'RMSE (m)':<12} | {'Time (s)':<10} | {'Scale':<10}
{'-'*60}
"""
        total_rmse = []
        total_time = []
        for station in results['stations']:
            metrics = self._safe_dict(station.get('quality_metrics'))
            status = "SUCCESS" if station.get(
                'coords') is not None else "FAILED"
            rmse = self._format_number(station.get('rmse'), 6)
            time_val = self._format_number(station.get('time'), 4)
            scale = self._format_number(metrics.get('scale'), 6)

            station_name = station.get(
                'name', station.get('station', 'Station'))
            stats_text += f"{station_name:<10} | {status:<10} | {rmse:<12} | {time_val:<10} | {scale:<10}\n"
            if self._is_valid_number(station.get('rmse')):
                total_rmse.append(station['rmse'])
            if self._is_valid_number(station.get('time')):
                total_time.append(station['time'])

        stats_text += f"{'='*60}\n"
        if total_rmse:
            stats_text += f"Overall Average RMSE: {np.mean(total_rmse):.6f} m\n"
            stats_text += f"Overall Total Time: {np.sum(total_time):.4f} s\n"
        else:
            stats_text += "No successful registrations to calculate overall statistics.\n"

        return stats_text

    def _format_network_adjustment_stats(self, results):
        """Format statistics for Network Adjustment."""
        method_name = results.get('method', 'Network Adjustment')
        stats_text = f"""
{method_name} - Network Adjustment Summary
{'='*60}

Type: {results.get('type', 'N/A')}
Stations in Network: {results.get('n_stations', 'N/A')}
Control Points in Network: {results.get('n_control_points', 'N/A')}
Reference Station: {results.get('reference_station', 'N/A')}
Total Computation Time: {self._format_number(results.get('time'), 4)} s

"""
        if self._is_valid_number(results.get('sigma0')):
            stats_text += f"Global Sigma0: {float(results['sigma0']):.6f} m\n"
            sigma0_sq = results.get('sigma0_sq')
            sigma0_sq_text = f"{float(sigma0_sq):.6e}" if self._is_valid_number(
                sigma0_sq) else "N/A"
            stats_text += f"Global Sigma0^2: {sigma0_sq_text}\n\n"

        cov = self._matrix_to_array(results.get('covariance'))
        if cov is not None and cov.ndim == 2:
            std_devs = np.sqrt(np.abs(np.diag(cov)))
            stats_text += "Parameter Standard Deviations (A Posteriori):\n"
            stats_text += "-"*60 + "\n"
            n_stations = len(results['stations'])
            if len(std_devs) == n_stations * 8:
                for i, station in enumerate(results['stations']):
                    st_devs = std_devs[i*8:(i+1)*8]
                    station_name = station.get(
                        'station', station.get('name', 'Station'))
                    stats_text += f"{station_name}:\n"
                    stats_text += f"  Scale: {st_devs[0]:.6e}\n"
                    stats_text += f"  Quaternions (q0,q1,q2,q3): {st_devs[1]:.6e}, {st_devs[2]:.6e}, {st_devs[3]:.6e}, {st_devs[4]:.6e}\n"
                    stats_text += f"  Translations (X,Y,Z): {st_devs[5]:.6f}m, {st_devs[6]:.6f}m, {st_devs[7]:.6f}m\n\n"
            else:
                stats_text += "Covariance matrix size does not match expected parameters.\n\n"

        stats_text += f"""
Per-Station Results:
{'Station':<10} | {'Status':<10} | {'RMSE (m)':<12} | {'Scale':<10} | {'N CPs':<8}
{'-'*60}
"""
        total_rmse = []
        for station in results['stations']:
            status = "SUCCESS" if station.get('success') else "FAILED"
            rmse = self._format_number(station.get('rmse'), 6)
            scale = self._format_number(station.get('scale'), 6)
            n_points = station.get('n_points', 'N/A')

            station_name = station.get(
                'station', station.get('name', 'Station'))
            stats_text += f"{station_name:<10} | {status:<10} | {rmse:<12} | {scale:<10} | {n_points:<8}\n"
            if self._is_valid_number(station.get('rmse')):
                total_rmse.append(station['rmse'])

        stats_text += f"{'='*60}\n"
        if total_rmse:
            stats_text += f"Overall Average RMSE: {np.mean(total_rmse):.6f} m\n"
        else:
            stats_text += "No successful station adjustments to calculate overall statistics.\n"

        return stats_text

    def export_coordinates(self):
        """Export coordinates to CSV."""
        uncertainties = getattr(self, 'current_uncertainties', {})
        stations_data = self.results.get(
            'transformed_stations', self.results.get('stations'))
        if not stations_data:
            QMessageBox.warning(
                self, "Warning", "No transformed stations to export!")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export Coordinates", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_name:
            try:
                coords_to_export = self._extract_coords_from_transformed_stations(
                    stations_data)
                data = []
                for point_id, coord in sorted(coords_to_export.items()):
                    row = {
                        'point_id': point_id,
                        'x': coord[0],
                        'y': coord[1],
                        'z': coord[2]
                    }
                    sigmas = self._point_sigmas_from_covariance(
                        uncertainties.get(point_id))
                    row['sigma_x'] = sigmas[0]
                    row['sigma_y'] = sigmas[1]
                    row['sigma_z'] = sigmas[2]

                    data.append(row)

                df = pd.DataFrame(data)
                df.to_csv(file_name, index=False)

                QMessageBox.information(
                    self, "Success", "Coordinates exported successfully!")

            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to export coordinates: {str(e)}")

    def create_matrices_widget(self):
        """Create the adjustment matrices widget."""
        widget = QWidget()
        layout = QVBoxLayout()
        self.matrices_tabs = QTabWidget()
        self.matrices_tabs.setStyleSheet("""
            QTabWidget::pane { border-top: 1px solid #d0d0d0; }
            QTabBar::tab {
                min-width: 80px;
                min-height: 22px;
                padding: 2px 8px;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.matrices_tabs)
        widget.setLayout(layout)
        return widget

    def update_matrices(self, results, matrix_dict=None):
        """Update matrix tables with compact numeric previews."""
        if not hasattr(self, 'matrices_tabs'):
            return
        self.matrices_tabs.clear()

        matrix_dict = matrix_dict if matrix_dict is not None else self._matrix_dict_for_result(
            results)
        self._add_matrix_summary_tab(matrix_dict)

        for name, mat in matrix_dict.items():
            tab = QWidget()
            layout = QVBoxLayout()
            mat_arr = self._matrix_to_array(mat)
            if mat_arr is None or mat_arr.size == 0:
                msg = QLabel(
                    f"The matrix '{name}' is not available for the selected method.")
                msg.setAlignment(Qt.AlignCenter)
                layout.addWidget(msg)
            elif mat_arr.ndim == 2:
                labels = self._matrix_labels_for_name(results, name, mat_arr)
                note = self._matrix_explanation_label(name, labels)
                if note is not None:
                    layout.addWidget(note)
                layout.addWidget(self._matrix_table_preview(mat_arr, labels))
                btn = QPushButton(f"Export {name} (CSV)")
                btn.clicked.connect(lambda checked, m=mat_arr,
                                    n=name: self.export_matrix(m, n))
                layout.addWidget(btn)
            else:
                layout.addWidget(self._matrix_table_preview(mat_arr))
                btn = QPushButton(f"Export {name} (CSV)")
                btn.clicked.connect(lambda checked, m=mat_arr,
                                    n=name: self.export_matrix(m, n))
                layout.addWidget(btn)
            tab.setLayout(layout)
            self.matrices_tabs.addTab(tab, name)

    def _matrix_dict_for_result(self, results):
        """Return available matrix-like outputs from the active method."""
        candidates = {
            "Design Matrix A": results.get('design_matrix'),
            "Weight Matrix P": results.get('weight_matrix'),
            "Normal Matrix N": results.get('normal_matrix'),
            "Observation Vector L": results.get('observation_vector'),
            "Vector U": results.get('vector_u'),
            "Vector X": results.get('vector_x'),
            "Residual Vector v": results.get('observation_residuals', results.get('residuals')),
            "Covariance (Parameters)": results.get('covariance'),
            "Covariance (Observations)": results.get('observation_covariance'),
            "Covariance (Residuals)": results.get('covariance_residuals'),
            "Correlation Matrix": results.get('correlation_matrix'),
            "Q Matrix": results.get('Q_matrix'),
        }
        for station in results.get('stations', []):
            station_name = station.get(
                'name', station.get('station', 'Station'))
            qm = self._safe_dict(station.get('quality_metrics'))
            candidates[f"{station_name} Rotation Matrix"] = qm.get(
                'rotation_matrix', station.get('rotation'))
            candidates[f"{station_name} Translation Vector"] = qm.get(
                'translation', station.get('translation'))
            candidates[f"{station_name} Covariance"] = qm.get('covariance')
            candidates[f"{station_name} Correlation Matrix"] = qm.get(
                'correlation_matrix')
            candidates[f"{station_name} Q Matrix"] = qm.get('Q_matrix')

        available = {}
        for name, value in candidates.items():
            arr = self._matrix_to_array(value)
            if arr is not None and arr.size > 0:
                available[name] = arr
        cov = available.get("Covariance (Parameters)")
        if cov is not None and cov.ndim == 2 and cov.shape[0] == cov.shape[1]:
            if "Correlation Matrix" not in available:
                available["Correlation Matrix (derived)"] = self._correlation_from_covariance(
                    cov)
            if "Normal Matrix N" not in available:
                available["Normal Matrix N (pseudo-inverse covariance)"] = np.linalg.pinv(
                    cov)
        return available

    def _matrix_to_array(self, value):
        """Convert matrix-like values to a numeric numpy array."""
        if value is None:
            return None
        try:
            arr = np.asarray(value)
        except Exception:
            return None
        if arr.size == 0:
            return None
        if arr.dtype == object:
            try:
                arr = arr.astype(float)
            except Exception:
                return None
        return arr.astype(float, copy=False)

    def _correlation_from_covariance(self, covariance):
        diag = np.clip(np.diag(covariance).astype(float), 0.0, None)
        std = np.sqrt(diag)
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = covariance / np.outer(std, std)
        corr[~np.isfinite(corr)] = 0.0
        return corr

    def _add_matrix_summary_tab(self, matrix_dict):
        tab = QWidget()
        layout = QVBoxLayout()
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(
            ["Matrix", "Shape", "Norm", "Min", "Max", "Type"])
        table.setRowCount(max(1, len(matrix_dict)))
        if not matrix_dict:
            table.setItem(0, 0, QTableWidgetItem("No matrices available"))
        for row, (name, arr) in enumerate(matrix_dict.items()):
            values = [
                name,
                " x ".join(str(dim) for dim in arr.shape),
                f"{np.linalg.norm(np.nan_to_num(arr)):.6e}",
                f"{np.nanmin(arr):.6e}",
                f"{np.nanmax(arr):.6e}",
                "matrix" if arr.ndim == 2 else "vector",
            ]
            for col, value in enumerate(values):
                table.setItem(row, col, QTableWidgetItem(value))
        self._polish_table(table)
        table.resizeColumnsToContents()
        layout.addWidget(table)
        tab.setLayout(layout)
        self.matrices_tabs.addTab(tab, "Summary")

    def _matrix_table_preview(self, mat_arr, labels=None):
        table = QTableWidget()
        if mat_arr.ndim == 1:
            max_rows = min(len(mat_arr), 500)
            table.setRowCount(max_rows)
            table.setColumnCount(1)
            table.setHorizontalHeaderLabels(["Value"])
            if labels and len(labels) >= max_rows:
                table.setVerticalHeaderLabels(
                    [str(label) for label in labels[:max_rows]])
            for i in range(max_rows):
                table.setItem(i, 0, QTableWidgetItem(f"{mat_arr[i]:.6e}"))
        else:
            max_rows = min(mat_arr.shape[0], 120)
            max_cols = min(mat_arr.shape[1], 40)
            table.setRowCount(max_rows)
            table.setColumnCount(max_cols)
            if labels:
                shown_cols = [str(label) for label in labels[:max_cols]]
                shown_rows = [str(label) for label in labels[:max_rows]]
                if len(shown_cols) == max_cols:
                    table.setHorizontalHeaderLabels(shown_cols)
                if len(shown_rows) == max_rows:
                    table.setVerticalHeaderLabels(shown_rows)
            for i in range(max_rows):
                for j in range(max_cols):
                    table.setItem(i, j, QTableWidgetItem(
                        f"{mat_arr[i, j]:.6e}"))
        self._polish_table(table)
        table.resizeColumnsToContents()
        return table

    def _matrix_labels_for_name(self, results, name, mat_arr):
        if mat_arr is None or mat_arr.ndim != 2 or mat_arr.shape[0] != mat_arr.shape[1]:
            return []
        if name != "Covariance (Parameters)":
            return []
        labels = list(results.get("covariance_param_names") or [])
        if len(labels) == mat_arr.shape[0]:
            return labels
        stations = list(results.get("stations", []) or [])
        if not stations or mat_arr.shape[0] % len(stations) != 0:
            return []
        block = mat_arr.shape[0] // len(stations)
        if block == 8:
            base = ["scale", "q0", "q1", "q2", "q3", "tx", "ty", "tz"]
        elif block == 7:
            base = ["scale", "rx", "ry", "rz", "tx", "ty", "tz"]
        else:
            base = [f"p{i + 1}" for i in range(block)]
        labels = []
        for station in stations:
            station_name = station.get(
                "station", station.get("name", "station"))
            labels.extend([f"{station_name}.{param}" for param in base])
        return labels

    def _matrix_explanation_label(self, name, labels):
        if name != "Covariance (Parameters)":
            return None
        text = (
            "Parameter covariance: rows and columns represent the estimated parameters "
            "for each station. scale is the station scale factor; q0-q3 are quaternion "
            "rotation components; rx-ry-rz are Euler rotation components when used; "
            "tx-ty-tz are translations in metres. Diagonal values are variances and "
            "off-diagonal values are covariances between parameters."
        )
        if labels:
            text += f" Showing {len(labels)} labelled parameters."
        note = QLabel(text)
        note.setWordWrap(True)
        note.setStyleSheet("color: #444; font-size: 11px;")
        return note

    def display_audit_outputs(self, output_dir, method_key):
        """Load advanced analyzer outputs into the main Results tabs."""
        output_dir = Path(output_dir)
        self.current_audit_output_dir = output_dir
        self.current_audit_method_key = method_key
        if hasattr(self, 'audit_widget'):
            index = self.results_tabs.indexOf(self.audit_widget)
            if index >= 0:
                self.results_tabs.removeTab(index)

        self.audit_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        audit_tabs = QTabWidget()
        audit_tabs.setDocumentMode(True)
        audit_tabs.setUsesScrollButtons(True)
        audit_tabs.setElideMode(Qt.ElideRight)
        audit_tabs.setStyleSheet("""
            QTabWidget::pane { border-top: 1px solid #d0d0d0; }
            QTabBar::tab {
                min-width: 76px;
                min-height: 21px;
                padding: 2px 7px;
                font-size: 11px;
            }
        """)
        layout.addWidget(audit_tabs)
        self.audit_widget.setLayout(layout)

        dashboard = self._results_dashboard_widget(output_dir, method_key)
        if dashboard is not None:
            audit_tabs.addTab(dashboard, "Overview")

        # Detailed metric tables are intentionally not shown in the main GUI.
        # They remain available in the exported CSV files for specialist review.

        # Keep the prepared widget available for focused/research use only.

        # Residual details are represented in the Overview and Outliers tabs.

        outlier_review = self._outlier_review_widget(output_dir, method_key)
        if outlier_review is not None:
            audit_tabs.addTab(outlier_review, "Outliers")

        diagnostics = self._diagnostic_plots_widget(output_dir, method_key)
        if diagnostics is not None:
            audit_tabs.addTab(diagnostics, "Plots")

        html_outputs = self._html_outputs_widget(output_dir, method_key)
        if html_outputs is not None:
            audit_tabs.addTab(html_outputs, "3D/HTML")
        else:
            plotly_note = output_dir / "plotly_html_not_generated.txt"
            if plotly_note.exists():
                audit_tabs.addTab(
                    self._message_widget(plotly_note.read_text(
                        encoding="utf-8", errors="replace")),
                    "Plotly Status",
                )

        if audit_tabs.count() > 0:
            self.results_tabs.addTab(self.audit_widget, "Analyzer")

    def display_advanced_error(self, message):
        """Show analyzer generation errors without hiding the basic method results."""
        if not message:
            return
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(str(message))
        self.results_tabs.addTab(text, "Analyzer Error")

    def _read_filtered_csv(self, output_dir, filename, method_key=None):
        path = Path(output_dir) / filename
        if not path.exists():
            return pd.DataFrame()
        df = self._filtered_csv(path, method_key)
        return df if df is not None else pd.DataFrame()

    def _metric_card_widget(self, label, value, note=""):
        card = QLabel(f"<b>{value}</b><br>{label}" + (f"<br><span>{note}</span>" if note else ""))
        card.setAlignment(Qt.AlignCenter)
        card.setMinimumHeight(58)
        card.setWordWrap(True)
        card.setStyleSheet(
            "background:#f8fafc; border:1px solid #d8dee6; border-radius:6px; "
            "padding:8px; color:#334155;"
        )
        return card

    def _format_mm_metric(self, value):
        if not self._is_valid_number(value):
            return "N/A"
        return f"{float(value) * 1000.0:.2f} mm"

    def _results_dashboard_widget(self, output_dir, method_key):
        station_df = self._read_filtered_csv(output_dir, "station_outputs.csv", method_key)
        residual_df = self._residual_review_dataframe(output_dir, method_key)
        outlier_df = self._read_filtered_csv(output_dir, "outlier_flags.csv", method_key)
        method_df = self._read_filtered_csv(output_dir, "method_summary.csv", method_key)

        if station_df.empty and residual_df.empty and method_df.empty:
            return None

        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel(f"Validated result overview: {method_key}")
        title.setStyleSheet("font-weight: bold; color: #202124; font-size: 14px;")
        layout.addWidget(title)
        note = QLabel(
            "This view keeps only the operational indicators needed for review. "
            "Detailed research tables remain exported as CSV files in the output folder."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        ok = station_df
        if not ok.empty and "success" in ok.columns:
            ok = ok[ok["success"].map(lambda value: self._truthy(value) if isinstance(value, str) else bool(value))]
        cards = QHBoxLayout()
        stations_success = len(ok) if not ok.empty else "N/A"
        mean_rmse = np.nan
        max_rmse = np.nan
        icp_rmse = np.nan
        if not ok.empty:
            if "rmse" in ok.columns:
                rmse_series = pd.to_numeric(ok["rmse"], errors="coerce")
                mean_rmse = rmse_series.mean()
                max_rmse = rmse_series.max()
            if "icp_rmse" in ok.columns:
                icp_series = pd.to_numeric(ok["icp_rmse"], errors="coerce")
                if icp_series.notna().any():
                    icp_rmse = float(np.sqrt(np.nanmean(icp_series ** 2)))
        if not method_df.empty:
            first = method_df.iloc[0]
            mean_rmse = first.get("mean_rmse", mean_rmse)
            max_rmse = first.get("max_rmse", max_rmse)
            icp_rmse = first.get("icp_rmse", icp_rmse)
        outlier_count = 0
        if not outlier_df.empty and "is_outlier_candidate" in outlier_df.columns:
            outlier_count = int(outlier_df["is_outlier_candidate"].map(self._truthy).sum())

        cards.addWidget(self._metric_card_widget("Stations solved", stations_success))
        cards.addWidget(self._metric_card_widget("Mean RMSE", self._format_mm_metric(mean_rmse)))
        cards.addWidget(self._metric_card_widget("Max RMSE", self._format_mm_metric(max_rmse)))
        cards.addWidget(self._metric_card_widget("ICP RMSE", self._format_mm_metric(icp_rmse)))
        cards.addWidget(self._metric_card_widget("Outlier candidates", outlier_count))
        layout.addLayout(cards)

        split = QSplitter(Qt.Horizontal)
        if not ok.empty:
            station_cols = [
                col for col in (
                    "station", "run_label", "success", "rmse", "icp_rmse",
                    "icp_max_residual", "excluded_outliers", "n_control_points",
                    "n_icp_points"
                )
                if col in ok.columns
            ]
            station_view = ok[station_cols].copy()
            for col in ("rmse", "icp_rmse", "icp_max_residual"):
                if col in station_view.columns:
                    station_view[col] = pd.to_numeric(station_view[col], errors="coerce") * 1000.0
                    station_view.rename(columns={col: f"{col}_mm"}, inplace=True)
            station_box = QGroupBox("Station summary")
            station_layout = QVBoxLayout()
            station_layout.addWidget(self._csv_preview_widget(
                Path(output_dir) / "station_outputs.csv",
                df=station_view.head(80),
            ))
            station_box.setLayout(station_layout)
            split.addWidget(station_box)

        if not residual_df.empty:
            residual_cols = [
                col for col in (
                    "role", "station", "point", "point_rmse_mm", "v_norm_mm",
                    "vx_mm", "vy_mm", "vz_mm", "is_outlier_candidate",
                    "detection_class", "baarda_score", "mad_score"
                )
                if col in residual_df.columns
            ]
            worst = residual_df[residual_cols].head(40).copy()
            residual_box = QGroupBox("Worst residuals")
            residual_layout = QVBoxLayout()
            residual_layout.addWidget(self._csv_preview_widget(
                Path(output_dir) / "residuals.csv",
                df=worst,
            ))
            residual_box.setLayout(residual_layout)
            split.addWidget(residual_box)

        if split.count() > 0:
            split.setSizes([560, 560])
            layout.addWidget(split, 1)

        widget.setLayout(layout)
        return widget

    def open_focused_results_review_dialog(self):
        output_dir = self.current_audit_output_dir
        method_key = self.current_audit_method_key
        if not output_dir or not Path(output_dir).exists():
            QMessageBox.information(
                self, "Focused Review",
                "Run a method with diagnostics before opening focused review.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Focused Review - {method_key}")
        dialog.resize(1250, 820)
        layout = QVBoxLayout()
        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        dashboard = self._results_dashboard_widget(output_dir, method_key)
        if dashboard is not None:
            tabs.addTab(dashboard, "Overview")
        outliers = self._outlier_review_widget(output_dir, method_key)
        if outliers is not None:
            tabs.addTab(outliers, "Outliers")
        plots = self._diagnostic_plots_widget(output_dir, method_key)
        if plots is not None:
            tabs.addTab(plots, "Plots")
        html = self._html_outputs_widget(output_dir, method_key)
        if html is not None:
            tabs.addTab(html, "3D/HTML")
        layout.addWidget(tabs, 1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)
        dialog.setLayout(layout)
        dialog.exec()

    def _truthy(self, value):
        return str(value).strip().lower() in {"true", "1", "yes", "y"}

    def _normalise_point_column(self, df):
        df = df.copy()
        if "point" not in df.columns and "point_id" in df.columns:
            df["point"] = df["point_id"]
        if "station" in df.columns:
            df["station"] = df["station"].astype(str)
        if "point" in df.columns:
            df["point"] = df["point"].astype(str)
        return df

    def _residual_review_dataframe(self, output_dir, method_key=None, controls_only=False):
        frames = []
        residual_path = output_dir / "residuals.csv"
        if residual_path.exists():
            df = self._filtered_csv(residual_path, method_key)
            if df is not None and not df.empty:
                df = self._normalise_point_column(df)
                df["source_table"] = "residuals.csv"
                if "role" not in df.columns:
                    df["role"] = "CP"
                frames.append(df)

        if not controls_only:
            icp_path = output_dir / "independent_check_points.csv"
            if icp_path.exists():
                icp = self._filtered_csv(icp_path, method_key)
                if icp is not None and not icp.empty:
                    icp = self._normalise_point_column(icp)
                    icp["source_table"] = "independent_check_points.csv"
                    icp["role"] = "ICP"
                    icp["is_control_point"] = False
                    frames.append(icp)

        if not frames:
            return pd.DataFrame()
        review = pd.concat(frames, ignore_index=True, sort=False)
        if method_key and "method" in review.columns:
            method_rows = review[review["method"].astype(str) == str(method_key)]
            if not method_rows.empty:
                review = method_rows.copy()

        for col in ("vx", "vy", "vz", "v_norm"):
            if col in review.columns:
                review[col] = pd.to_numeric(review[col], errors="coerce")
        if "v_norm" not in review.columns or review["v_norm"].isna().all():
            if {"vx", "vy", "vz"}.issubset(review.columns):
                review["v_norm"] = np.sqrt(
                    review["vx"].fillna(0) ** 2 +
                    review["vy"].fillna(0) ** 2 +
                    review["vz"].fillna(0) ** 2
                )
        if "v_norm" not in review.columns:
            review["v_norm"] = np.nan
        for col in ("v_norm", "vx", "vy", "vz"):
            if col in review.columns:
                review[f"{col}_mm"] = review[col] * 1000.0
        point_key_cols = [
            col for col in ("method", "mode", "run_label", "station", "point")
            if col in review.columns
        ]
        if point_key_cols and "v_norm" in review.columns:
            point_stats = (
                review.groupby(point_key_cols, dropna=False)["v_norm"]
                .apply(lambda values: float(np.sqrt(np.nanmean(pd.to_numeric(values, errors="coerce") ** 2))))
                .reset_index(name="point_rmse")
            )
            review = review.merge(point_stats, on=point_key_cols, how="left")
            review["point_rmse_mm"] = review["point_rmse"] * 1000.0

        flags_path = output_dir / "outlier_flags.csv"
        if flags_path.exists():
            try:
                flags = pd.read_csv(flags_path)
                flags = self._normalise_point_column(flags)
                if method_key and "method" in flags.columns:
                    flags = flags[flags["method"].astype(str) == str(method_key)].copy()
                key_cols = [
                    col for col in ("method", "mode", "run_label", "station", "point")
                    if col in review.columns and col in flags.columns
                ]
                flag_cols = [
                    col for col in (
                        "is_outlier_candidate", "detection_class",
                        "outlier_score", "n_outlier_flags", "baarda_score",
                        "mad_score", "critical_component", "baarda_available",
                        "baarda_unavailable_reason", "mad_warning",
                    )
                    if col in flags.columns
                ]
                if key_cols and flag_cols:
                    flags = flags[key_cols + flag_cols].drop_duplicates(key_cols)
                    review = review.merge(
                        flags, on=key_cols, how="left", suffixes=("", "_flag"))
            except Exception:
                pass

        if "is_outlier_candidate" not in review.columns:
            review["is_outlier_candidate"] = False
        review["is_outlier_candidate"] = review["is_outlier_candidate"].map(self._truthy)

        if controls_only:
            if "role" in review.columns:
                review = review[~review["role"].astype(str).str.upper().eq("ICP")].copy()
            if "is_control_point" in review.columns:
                review = review[review["is_control_point"].map(
                    lambda value: self._truthy(value) if isinstance(value, str) else bool(value)
                )].copy()

        role_series = (
            review["role"]
            if "role" in review.columns
            else pd.Series("CP", index=review.index)
        )
        review["__role_order"] = role_series.astype(str).str.upper().eq("ICP").astype(int)
        sort_metric = "point_rmse" if "point_rmse" in review.columns else "v_norm"
        review = review.sort_values(
            ["__role_order", "is_outlier_candidate", sort_metric],
            ascending=[True, False, False],
        )
        return review.drop(columns=["__role_order"], errors="ignore")

    def _residual_review_widget(self, output_dir, method_key):
        review = self._residual_review_dataframe(output_dir, method_key)
        if review.empty:
            return None
        columns = [
            col for col in (
                "method", "run_label", "role", "station", "point",
                "point_rmse_mm", "v_norm_mm", "vx_mm", "vy_mm", "vz_mm",
                "is_outlier_candidate", "detection_class",
                "baarda_score", "mad_score", "critical_component",
            )
            if col in review.columns
        ]
        shown = review[columns].head(300).copy() if columns else review.head(300).copy()
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        title = QLabel("Residual review")
        title.setStyleSheet(
            "font-weight: bold; color: #202124; font-size: 13px;")
        layout.addWidget(title)
        note = QLabel(
            "Rows are ordered by control/check role, outlier flag and residual norm. "
            "Residual components are shown in millimetres when available."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addWidget(self._csv_preview_widget(
            output_dir / "residuals.csv", df=shown), 1)
        widget.setLayout(layout)
        return widget

    def _manual_exclusion_specs_from_table(self, table):
        specs = []
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item is None or item.checkState() != Qt.Checked:
                continue
            spec = item.data(Qt.UserRole)
            if spec:
                specs.append(str(spec))
        return sorted(set(specs))

    def _set_highlighted_outlier_rows_checked(self, table, checked=True):
        rows = sorted({index.row() for index in table.selectedIndexes()})
        for row in rows:
            item = table.item(row, 0)
            if item is not None:
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def _update_manual_outlier_action_state(self, table, count_label, run_button):
        selected_count = len(self._manual_exclusion_specs_from_table(table))
        count_label.setText(f"Selected for removal: {selected_count}")
        run_button.setEnabled(selected_count > 0)

    def _select_outlier_flagged_rows(self, table, flag_col, count_label, run_button):
        for row in range(table.rowCount()):
            flag_item = table.item(row, flag_col)
            check_item = table.item(row, 0)
            if check_item is not None:
                check_item.setCheckState(
                    Qt.Checked
                    if flag_item is not None and flag_item.text().strip().lower() == "yes"
                    else Qt.Unchecked
                )
        self._update_manual_outlier_action_state(table, count_label, run_button)

    def _clear_manual_outlier_selection(self, table, count_label, run_button):
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item is not None:
                item.setCheckState(Qt.Unchecked)
        self._update_manual_outlier_action_state(table, count_label, run_button)

    def _request_manual_outlier_rerun(self, output_dir, method_key, table):
        specs = self._manual_exclusion_specs_from_table(table)
        if not specs:
            QMessageBox.information(
                self, "Manual outlier review",
                "Select at least one station-point row before reprocessing.")
            return
        answer = QMessageBox.question(
            self,
            "Manual outlier review",
            "Run the selected method again excluding these station-point observations?\n\n"
            + "\n".join(specs[:20])
            + ("\n..." if len(specs) > 20 else ""),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        window = self.window()
        if hasattr(window, "run_outlier_removal_from_results"):
            window.run_outlier_removal_from_results(
                str(output_dir), method_key, manual_exclusions=specs)

    def _outlier_review_widget(self, output_dir, method_key):
        flags_path = output_dir / "outlier_flags.csv"
        config_path = output_dir / "run_configuration.json"
        outlier_action = "report"
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as f:
                    outlier_action = str(json.load(f).get(
                        "outlier_action", "report"))
            except Exception:
                outlier_action = "report"
        if not flags_path.exists():
            return None
        try:
            flags_df = pd.read_csv(flags_path)
        except Exception:
            return None
        if flags_df.empty or "is_outlier_candidate" not in flags_df.columns:
            return None

        review_df = self._residual_review_dataframe(
            output_dir, method_key, controls_only=True)
        if review_df.empty:
            return None
        selected_candidates = review_df.head(250).copy()
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)

        title = QLabel("Outlier removal audit" if outlier_action ==
                       "remove" else "Outlier review before removal")
        title.setStyleSheet(
            "font-weight: bold; color: #202124; font-size: 13px;")
        layout.addWidget(title)

        table_title = QLabel("Manual removal review")
        table_title.setStyleSheet("font-weight: bold; color: #334155;")
        layout.addWidget(table_title)
        table = QTableWidget()
        headers = [
            "Remove?", "Run", "Station", "Point", "Point RMSE mm",
            "|v| mm", "vx mm", "vy mm", "vz mm", "Flag", "Class",
            "Baarda |w|", "MAD score", "Critical"
        ]
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(selected_candidates))
        for row, (_, item) in enumerate(selected_candidates.iterrows()):
            station = str(item.get("station", ""))
            point = str(item.get("point", ""))
            flagged = bool(item.get("is_outlier_candidate", False))
            check_item = QTableWidgetItem("")
            check_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            check_item.setCheckState(Qt.Checked if flagged else Qt.Unchecked)
            check_item.setData(Qt.UserRole, f"{station}:{point}")
            table.setItem(row, 0, check_item)
            values = [
                item.get("run_label", ""),
                station,
                point,
                item.get("point_rmse_mm", np.nan),
                item.get("v_norm_mm", np.nan),
                item.get("vx_mm", np.nan),
                item.get("vy_mm", np.nan),
                item.get("vz_mm", np.nan),
                "yes" if flagged else "",
                item.get("detection_class", ""),
                item.get("baarda_score", np.nan),
                item.get("mad_score", np.nan),
                item.get("critical_component", ""),
            ]
            for col, value in enumerate(values, start=1):
                cell = QTableWidgetItem(self._format_cell(value))
                if flagged:
                    cell.setBackground(QColor("#fff3cd"))
                table.setItem(row, col, cell)
        self._polish_table(table)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.resizeColumnsToContents()
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        table.setToolTip(
            "Mark rows in the first column, or select rows and use 'Mark highlighted rows'. "
            "Point RMSE is computed from the residual norm for each station-point observation."
        )
        layout.addWidget(table, 3)

        action_box = QGroupBox(
            "Manual outlier removal")
        action_layout = QVBoxLayout()
        if outlier_action == "remove":
            action_note = QLabel(
                "This output already contains an outlier-removal pass. You can still select rows below and run a new manual pass in a separate output folder."
            )
        else:
            action_note = QLabel(
                "Select the observations to remove after checking RMSE, residual components and Baarda/MAD scores. "
                "The original output folder is preserved; the manual pass is written to a separate folder."
            )
        action_note.setWordWrap(True)
        action_layout.addWidget(action_note)

        button_row = QHBoxLayout()
        selection_count_label = QLabel()
        selection_count_label.setMinimumWidth(150)
        selection_count_label.setStyleSheet("font-weight: bold; color: #334155;")
        button_row.addWidget(selection_count_label)

        mark_highlighted_btn = QPushButton("Mark highlighted rows")
        mark_highlighted_btn.setMinimumHeight(32)
        button_row.addWidget(mark_highlighted_btn)

        select_flagged_btn = QPushButton("Select flagged")
        select_flagged_btn.setMinimumHeight(32)
        button_row.addWidget(select_flagged_btn)

        clear_btn = QPushButton("Clear selection")
        clear_btn.setMinimumHeight(32)
        button_row.addWidget(clear_btn)

        remove_btn = QPushButton("Run again without selected points")
        remove_btn.setMinimumHeight(36)
        remove_btn.setStyleSheet(
            "font-weight: bold; background-color: #2E7D32; color: white; "
            "border-radius: 5px; padding: 8px;"
        )
        remove_btn.clicked.connect(
            lambda: self._request_manual_outlier_rerun(output_dir, method_key, table)
        )
        button_row.addWidget(remove_btn)
        button_row.addStretch()
        action_layout.addLayout(button_row)
        action_box.setLayout(action_layout)
        layout.addWidget(action_box)

        mark_highlighted_btn.clicked.connect(
            lambda: (
                self._set_highlighted_outlier_rows_checked(table, True),
                self._update_manual_outlier_action_state(table, selection_count_label, remove_btn),
            )
        )
        select_flagged_btn.clicked.connect(
            lambda: self._select_outlier_flagged_rows(
                table, 9, selection_count_label, remove_btn)
        )
        clear_btn.clicked.connect(
            lambda: self._clear_manual_outlier_selection(
                table, selection_count_label, remove_btn)
        )
        table.itemChanged.connect(
            lambda *_args: self._update_manual_outlier_action_state(
                table, selection_count_label, remove_btn)
        )
        self._update_manual_outlier_action_state(
            table, selection_count_label, remove_btn)

        widget.setLayout(layout)
        return widget

    def _output_availability_widget(self, output_dir, method_key):
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setUsesScrollButtons(True)
        added = False

        file_rows = []
        file_descriptions = {
            "method_summary.csv": "Per-method RMSE, ICP and outlier counts.",
            "station_outputs.csv": "Station-level transformations and fit statistics.",
            "residuals.csv": "Control residual vectors and residual norms.",
            "independent_check_points.csv": "Independent check-point residuals.",
            "outlier_flags.csv": "Baarda/MAD candidate flags and scores.",
            "outlier_summary.csv": "Outlier counts by method, run and station.",
        }
        for filename, description in file_descriptions.items():
            path = output_dir / filename
            if not path.exists():
                continue
            rows = ""
            if path.suffix.lower() == ".csv":
                try:
                    rows = len(pd.read_csv(path))
                except Exception:
                    rows = "N/A"
            file_rows.append({
                "Output": filename,
                "Rows": rows,
                "Description": description,
            })
        if file_rows:
            tabs.addTab(
                self._csv_preview_widget(
                    output_dir / "outputs", df=pd.DataFrame(file_rows)),
                "Files",
            )
            added = True

        return tabs if added else None

    def _filtered_csv(self, path, method_key):
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        if df.empty:
            return df
        if "method" in df.columns:
            filtered = df[df["method"].astype(str) == str(method_key)]
            if not filtered.empty:
                return filtered
        text_columns = [
            col for col in df.columns
            if df[col].dtype == object or str(df[col].dtype).startswith("string")
        ]
        if text_columns and method_key:
            mask = pd.Series(False, index=df.index)
            for col in text_columns:
                mask = mask | df[col].astype(str).str.contains(
                    str(method_key), regex=False, na=False)
            filtered = df[mask]
            if not filtered.empty:
                return filtered
        return df

    def _selected_method_overview_widget(self, output_dir, method_key):
        path = output_dir / "accuracy_quality_full.csv"
        if not path.exists():
            return None
        df = self._filtered_csv(path, method_key)
        if df is None or df.empty:
            return None
        priority = [
            "GCP_m_x", "GCP_m_y", "GCP_m_z", "GCP_m_p",
            "GCP_sigma_X", "GCP_sigma_Y", "GCP_sigma_Z",
            "sigma_0_mm", "tr_Q_x", "lambda", "lambda_av",
            "orthogonality_deviation_norm", "omega", "phi", "kappa",
            "sigma_AH2", "sigma_HD2", "error_ellipse_2d", "vertical_error_surface_3d",
        ]
        if "variable" in df.columns:
            selected = df[df["variable"].isin(priority)].copy()
            if selected.empty:
                selected = df.head(80).copy()
            else:
                selected["scope_order"] = selected["scope"].astype(
                    str).ne("method_summary").astype(int)
                selected["order"] = selected["variable"].map(
                    {name: idx for idx, name in enumerate(priority)})
                selected = selected.sort_values(["scope_order", "scope", "order"]).drop(
                    columns=["scope_order", "order"])
        else:
            selected = df.head(80).copy()

        widget = QWidget()
        layout = QVBoxLayout()
        title = QLabel(f"Selected method: {method_key}")
        title.setStyleSheet(
            "font-weight: bold; color: #202124; font-size: 13px;")
        layout.addWidget(title)
        subtitle = QLabel(
            "Only outputs available for this processing method are shown. Blue means comparatively better; red means comparatively worse; gray means N/D.")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        preview = self._csv_preview_widget(path, df=selected)
        layout.addWidget(preview, 1)
        widget.setLayout(layout)
        return widget

    def _method_summary_widget(self, output_dir, method_key=None):
        path = output_dir / "accuracy_quality_summary.csv"
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        if df.empty:
            return None
        if method_key and "method" in df.columns:
            filtered = df[df["method"].astype(str) == str(method_key)].copy()
            if not filtered.empty:
                df = filtered
        preferred = [
            "method", "GCP_m_p", "GCP_m_x", "GCP_m_y", "GCP_m_z",
            "tr_Q_x", "lambda", "lambda_av", "orthogonality_deviation_norm",
            "omega", "phi", "kappa",
        ]
        columns = [col for col in preferred if col in df.columns]
        if columns:
            df = df[columns]
        widget = QWidget()
        layout = QVBoxLayout()
        title_text = (
            f"{method_key} - article quality summary"
            if method_key else
            "Article quality summary"
        )
        title = QLabel(title_text)
        title.setStyleSheet(
            "font-weight: bold; color: #202124; font-size: 13px;")
        layout.addWidget(title)
        note = QLabel(
            "The table keeps the article metrics for the selected method: residual quality, "
            "scale diagnostics, orientation and covariance indicators when available."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addWidget(self._csv_preview_widget(path, df=df), 1)
        widget.setLayout(layout)
        return widget

    def _diagnostic_plots_widget(self, output_dir, method_key=None):
        generated, messages = generate_gui_diagnostic_plots(
            output_dir, method_key)

        if not generated and not messages:
            return None

        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)
        title = QLabel("Diagnostic plots")
        title.setStyleSheet(
            "font-weight: bold; color: #202124; font-size: 13px;")
        layout.addWidget(title)
        note = QLabel(
            "These plots are generated from the analyzer CSV/NPZ outputs for the selected method."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        if generated:
            preview_tabs = QTabWidget()
            preview_tabs.setDocumentMode(True)
            preview_tabs.setUsesScrollButtons(True)
            preview_tabs.setElideMode(Qt.ElideRight)
            for item in generated:
                path = Path(item.get("path", ""))
                label = item.get("label", path.stem or "Plot")
                if not path.exists():
                    continue
                suffix = path.suffix.lower()
                if suffix in {".png", ".jpg", ".jpeg"}:
                    tab_widget = self._embedded_matplotlib_file_widget(path)
                elif suffix == ".html":
                    tab_widget = self._embedded_html_file_widget(
                        path, item.get("description", ""))
                else:
                    continue
                preview_tabs.addTab(tab_widget, label[:28])

            if preview_tabs.count() > 0:
                layout.addWidget(preview_tabs, 1)

        if messages:
            msg = QTextBrowser()
            msg.setPlainText("\n".join(messages))
            layout.addWidget(msg, 1)
        widget.setLayout(layout)
        return widget

    def _embedded_matplotlib_file_widget(self, image_path):
        """Display a generated Matplotlib image inside a Qt canvas."""
        image_path = Path(image_path)
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(3)

        if FigureCanvas is not None and Figure is not None and mpl_image is not None:
            try:
                fig = Figure(figsize=(10, 7), tight_layout=True)
                canvas = FigureCanvas(fig)
                canvas.setSizePolicy(QSizePolicy.Expanding,
                                     QSizePolicy.Expanding)
                ax = fig.add_subplot(111)
                ax.imshow(mpl_image.imread(str(image_path)))
                ax.axis("off")
                layout.addWidget(canvas, 1)
                widget.setLayout(layout)
                return widget
            except Exception as exc:
                layout.addWidget(QLabel(
                    f"Could not render the Matplotlib canvas for {image_path.name}: {exc}"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        image_label = QLabel()
        pixmap = QPixmap(str(image_path))
        if not pixmap.isNull():
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setMinimumSize(pixmap.size())
            scroll.setWidget(image_label)
            layout.addWidget(scroll, 1)
        else:
            note = QLabel(f"Could not load image file: {image_path}")
            note.setWordWrap(True)
            layout.addWidget(note)
        widget.setLayout(layout)
        return widget

    def _embedded_html_file_widget(self, html_path, description=""):
        """Display a local Plotly/HTML output inside the GUI when WebEngine exists."""
        html_path = Path(html_path)
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        if description:
            note = QLabel(description)
            note.setWordWrap(True)
            layout.addWidget(note)

        if QWebEngineView is not None:
            view = QWebEngineView()
            view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            view.setUrl(QUrl.fromLocalFile(str(html_path.resolve())))
            layout.addWidget(view, 1)
        else:
            fallback = QTextBrowser()
            fallback.setSizePolicy(QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
            fallback.setPlainText(
                "PyQtWebEngine is not installed, so this interactive HTML "
                "cannot be embedded in the GUI on this Python environment.\n\n"
                f"File ready:\n{html_path}\n\n"
                "Install PyQtWebEngine to show Plotly outputs inside the application."
            )
            layout.addWidget(fallback, 1)
            open_btn = QPushButton("Open in Browser")
            open_btn.clicked.connect(
                lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(html_path))))
            layout.addWidget(open_btn)

        widget.setLayout(layout)
        return widget

    def _show_embedded_html_dialog(self, html_path):
        html_path = Path(html_path)
        if not html_path.exists():
            QMessageBox.warning(
                self, "HTML Output", f"HTML file was not found:\n{html_path}")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(html_path.name)
        dialog.resize(1100, 760)
        layout = QVBoxLayout()
        layout.addWidget(
            self._embedded_html_file_widget(
                html_path, self._html_output_description(html_path)),
            1,
        )
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        dialog.setLayout(layout)
        dialog.exec()

    def _per_method_overview_tabs(self, output_dir):
        path = output_dir / "accuracy_quality_full.csv"
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        if df.empty or "method" not in df.columns:
            return None
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setUsesScrollButtons(True)
        tabs.setElideMode(Qt.ElideRight)
        tabs.setStyleSheet("""
            QTabWidget::pane { border-top: 1px solid #d0d0d0; }
            QTabBar::tab {
                min-width: 74px;
                min-height: 21px;
                padding: 2px 7px;
                font-size: 11px;
            }
        """)
        methods = sorted(str(m) for m in df["method"].dropna().unique())
        priority = [
            "GCP_m_x", "GCP_m_y", "GCP_m_z", "GCP_m_p",
            "GCP_sigma_X", "GCP_sigma_Y", "GCP_sigma_Z",
            "sigma_0_mm", "tr_Q_x", "lambda", "lambda_av",
            "orthogonality_deviation_norm", "omega", "phi", "kappa",
            "m_omega", "m_phi", "m_kappa", "error_ellipse_2d",
            "vertical_error_surface_3d",
        ]
        for method in methods:
            method_df = df[df["method"].astype(str) == method].copy()
            if "scope" in method_df.columns:
                summary = method_df[method_df["scope"].astype(
                    str) == "method_summary"].copy()
                if not summary.empty:
                    method_df = summary
            if "variable" in method_df.columns:
                selected = method_df[method_df["variable"].isin(
                    priority)].copy()
                if not selected.empty:
                    selected["order"] = selected["variable"].map(
                        {name: idx for idx, name in enumerate(priority)})
                    method_df = selected.sort_values(
                        "order").drop(columns=["order"])
            tabs.addTab(self._csv_preview_widget(
                path, df=method_df.head(80)), method[:28])
        return tabs if tabs.count() else None

    def _csv_preview_widget(self, path, df=None):
        widget = QWidget()
        layout = QVBoxLayout()
        if df is None:
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                layout.addWidget(QLabel(f"Could not read {path.name}: {exc}"))
                widget.setLayout(layout)
                return widget
        table = QTableWidget()
        max_rows = min(len(df), 1000)
        max_cols = min(len(df.columns), 80)
        table.setRowCount(max_rows)
        table.setColumnCount(max_cols)
        table.setHorizontalHeaderLabels(
            [str(c) for c in df.columns[:max_cols]])
        indicator_col = df.columns.get_loc(
            "indicator") if "indicator" in df.columns[:max_cols] else None
        for i in range(max_rows):
            indicator = str(df.iat[i, indicator_col]
                            ) if indicator_col is not None else ""
            for j in range(max_cols):
                item = QTableWidgetItem(str(df.iat[i, j]))
                if indicator in {"better", "good"}:
                    item.setBackground(QColor("#d8ecff"))
                elif indicator in {"worse", "bad"}:
                    item.setBackground(QColor("#ffd9d9"))
                elif indicator == "not_available":
                    item.setBackground(QColor("#eeeeee"))
                table.setItem(i, j, item)
        self._polish_table(table)
        table.resizeColumnsToContents()
        layout.addWidget(
            QLabel(f"{path.name}: showing {max_rows} of {len(df)} rows"))
        layout.addWidget(table)
        open_btn = QPushButton("Open CSV")
        open_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))))
        layout.addWidget(open_btn)
        widget.setLayout(layout)
        return widget

    def _arrays_inventory_widget(self, arrays_path, method_key=None):
        widget = QWidget()
        layout = QVBoxLayout()
        arrays = np.load(arrays_path)
        array_keys = [
            key for key in arrays.files
            if not method_key or str(method_key) in key
        ]
        if not array_keys:
            return None
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Array", "Shape", "Preview"])
        table.setRowCount(len(array_keys))
        for row, key in enumerate(array_keys):
            arr = arrays[key]
            preview = ", ".join(f"{float(v):.3e}" for v in arr.ravel()[
                                :4]) if arr.size else ""
            table.setItem(row, 0, QTableWidgetItem(key))
            table.setItem(row, 1, QTableWidgetItem(
                " x ".join(map(str, arr.shape))))
            table.setItem(row, 2, QTableWidgetItem(preview))
        table.setAlternatingRowColors(True)
        self._polish_table(table)
        table.resizeColumnsToContents()
        layout.addWidget(
            QLabel(f"Complete numeric arrays: {arrays_path.name}"))
        layout.addWidget(table)
        widget.setLayout(layout)
        return widget

    def _html_outputs_widget(self, output_dir, method_key):
        outputs = []
        seen = set()

        def add_output(label, path, description):
            path = Path(path)
            if not path.exists():
                return False
            key = str(path.resolve())
            if key in seen:
                return False
            seen.add(key)
            outputs.append((label, path, description))
            return True

        method_dir = output_dir / "plotly_3d_by_method"
        selected_3d_added = False
        selected_rmse_added = False
        if method_key:
            selected_3d_added = add_output(
                "Selected method 3D",
                method_dir / f"{method_key}_3d.html",
                "Point cloud, residual vectors and error ellipsoids for the selected method.",
            )
            selected_rmse_added = add_output(
                "Selected method RMSE colors",
                method_dir / f"{method_key}_rmse_3d.html",
                "Same 3D view, with marker color scaled by point RMSE in millimetres.",
            )
        if not selected_3d_added:
            add_output(
                "Selected method 3D",
                output_dir / "selected_method_3d.html",
                "Point cloud, residual vectors and error ellipsoids for the selected method.",
            )
        if not selected_rmse_added:
            add_output(
                "Selected method RMSE colors",
                output_dir / "selected_method_rmse_3d.html",
                "Same 3D view, with marker color scaled by point RMSE in millimetres.",
            )
        add_output(
            "Accuracy HTML",
            output_dir / "accuracy_quality_summary.html",
            "Article quality report for the selected methodology, with color indicators for metric interpretation.",
        )

        if not outputs:
            return None

        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        title = QLabel("Selected-method HTML outputs")
        title.setStyleSheet(
            "font-weight: bold; color: #202124; font-size: 13px;")
        layout.addWidget(title)
        note = QLabel(
            "Selected-method Plotly/HTML views are embedded below when PyQtWebEngine "
            "is available in this Python environment."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        preview_tabs = QTabWidget()
        preview_tabs.setDocumentMode(True)
        preview_tabs.setUsesScrollButtons(True)
        preview_tabs.setElideMode(Qt.ElideRight)
        tab_by_path = {}
        for label, path, description in outputs:
            preview_tabs.addTab(
                self._embedded_html_file_widget(path, description),
                label[:28],
            )
            tab_by_path[str(path)] = preview_tabs.count() - 1

        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(
            ["Output", "File", "Description", "View"])
        table.setRowCount(len(outputs))
        for row, (label, path, description) in enumerate(outputs):
            table.setItem(row, 0, QTableWidgetItem(label))
            table.setItem(row, 1, QTableWidgetItem(path.name))
            table.setItem(row, 2, QTableWidgetItem(description))
            btn = QPushButton("View")
            btn.clicked.connect(
                lambda checked=False, p=str(path): preview_tabs.setCurrentIndex(
                    tab_by_path.get(p, 0))
            )
            table.setCellWidget(row, 3, btn)
        self._polish_table(table)
        table.resizeColumnsToContents()
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        table.setMaximumHeight(150)
        table.setMinimumHeight(105)
        layout.addWidget(table, 1)
        if preview_tabs.count() > 0:
            layout.addWidget(preview_tabs, 7)
        widget.setLayout(layout)
        return widget

    def _html_output_description(self, html_path):
        name = Path(html_path).name.lower()
        if name == "accuracy_quality_summary.html":
            return "Article quality HTML for the selected methodology, with color indicators for metric interpretation."
        if "rmse" in name:
            return "3D Plotly HTML where point color represents point RMSE in millimetres."
        return "3D Plotly HTML generated with point clouds, residual vectors and error ellipsoids when available."

    def _html_link_widget(self, html_path):
        return self._embedded_html_file_widget(
            html_path, self._html_output_description(html_path))

    def export_selected_matrix(self):
        """Export one matrix from the current result without entering matrix tabs."""
        matrix_dict = self._matrix_dict_for_result(self.results)
        if not matrix_dict:
            QMessageBox.warning(
                self, "Export Matrix", "No matrix is available for the current result.")
            return
        names = list(matrix_dict.keys())
        selected, ok = QInputDialog.getItem(
            self,
            "Export Matrix",
            "Choose the matrix to export:",
            names,
            0,
            False,
        )
        if ok and selected:
            self.export_matrix(matrix_dict[selected], selected)

    def open_primary_html_output(self):
        """Open the most relevant HTML output generated by the analyzer."""
        html_candidates = [
            self.results.get("advanced_rmse_html_path"),
            self.results.get("advanced_html_path"),
        ]
        output_dir = self.results.get("advanced_output_dir")
        if output_dir:
            output_dir = Path(output_dir)
            html_candidates.extend([
                output_dir / "selected_method_rmse_3d.html",
                output_dir / "selected_method_3d.html",
                output_dir / "accuracy_quality_summary.html",
            ])
        for candidate in html_candidates:
            if not candidate:
                continue
            path = Path(candidate)
            if path.exists():
                self._show_embedded_html_dialog(path)
                return
        if output_dir and Path(output_dir).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir)))
            return
        QMessageBox.warning(
            self, "HTML Output", "No HTML output is available for the current result.")

    def export_matrix(self, mat, name):
        """Export matrix data to CSV."""
        file_name, _ = QFileDialog.getSaveFileName(
            self, f"Export {name}", "", "CSV Files (*.csv);;All Files (*)")
        if file_name:
            try:
                pd.DataFrame(mat).to_csv(file_name, index=False, header=False)
                QMessageBox.information(
                    self, "Success", f"{name} exported successfully!")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to export {name}: {str(e)}")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.worker_thread = None
        self.outlier_worker_thread = None
        self.logger = logging.getLogger(__name__)
        self.last_processing_options = {}
        self.last_advanced_data_folder = ""
        self.processing_busy = False

        self.init_ui()

    def init_ui(self):
        """Initialize the main UI."""
        self.setWindowTitle("GeoNet Adjust")
        self.setGeometry(100, 100, 1400, 900)

        # Start maximized
        self.showMaximized()

        # Create central widget
        central_widget = QWidget()
        central_widget.setObjectName("appRoot")
        self.setCentralWidget(central_widget)
        self.setStyleSheet(self._application_stylesheet())

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)

        self._setup_workflow_header(layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimumHeight(6)
        self.progress_bar.setMaximumHeight(6)
        layout.addWidget(self.progress_bar)

        self._setup_main_tabs(layout)
        central_widget.setLayout(layout)
        self.statusBar().showMessage("Ready")
        self.logo_label = self._create_logo_widget(22)
        if self.logo_label:
            self.statusBar().addPermanentWidget(self.logo_label, 0)
        self._create_menu_bar()
        self._update_workflow_state()

    def _application_stylesheet(self):
        return """
            QWidget#appRoot {
                background: #f5f7fa;
                color: #202124;
                font-size: 12px;
            }
            QFrame#workflowHeader {
                background: white;
                border: 1px solid #d8dee6;
                border-radius: 8px;
            }
            QLabel#appTitle {
                font-size: 18px;
                font-weight: 700;
                color: #0f172a;
            }
            QLabel#appSubtitle {
                color: #5f6368;
            }
            QLabel#workflowStatus {
                color: #334155;
                background: #f8fafc;
                border: 1px solid #d8dee6;
                border-radius: 5px;
                padding: 5px 8px;
            }
            QPushButton#stepButton {
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                background: #ffffff;
                color: #334155;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton#stepButton[stepState="current"] {
                background: #1565c0;
                color: white;
                border-color: #1565c0;
            }
            QPushButton#stepButton[stepState="done"] {
                background: #e8f5e9;
                color: #1b5e20;
                border-color: #a5d6a7;
            }
            QPushButton#stepButton[stepState="pending"] {
                background: #ffffff;
                color: #64748b;
            }
            QPushButton#stepButton:disabled {
                background: #f1f5f9;
                color: #94a3b8;
                border-color: #e2e8f0;
            }
            QPushButton#workflowNavButton {
                min-height: 30px;
                padding: 6px 12px;
                border-radius: 5px;
                border: 1px solid #cbd5e1;
                background: #ffffff;
            }
            QPushButton#workflowNavButton:hover {
                background: #eef4ff;
            }
            QGroupBox {
                background: white;
                border: 1px solid #d8dee6;
                border-radius: 7px;
                margin-top: 14px;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #334155;
                font-weight: 700;
            }
            QTableWidget {
                background: white;
                gridline-color: #e2e8f0;
                selection-background-color: #dbeafe;
                selection-color: #0f172a;
            }
            QHeaderView::section {
                background: #f1f5f9;
                color: #334155;
                padding: 5px;
                border: 1px solid #e2e8f0;
                font-weight: 600;
            }
            QTabWidget::pane {
                border: 1px solid #d8dee6;
                background: white;
            }
            QTabBar::tab {
                min-height: 24px;
                padding: 5px 10px;
            }
            QTabBar::tab:selected {
                color: #1565c0;
                font-weight: 700;
            }
        """

    def _create_menu_bar(self):
        """Create a small application menu for common window actions."""
        menubar = self.menuBar()
        menubar.clear()

        file_menu = menubar.addMenu("&File")
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

        view_menu = menubar.addMenu("&View")
        input_action = view_menu.addAction("Input")
        input_action.setShortcut("Ctrl+1")
        input_action.triggered.connect(
            lambda: self._switch_workflow_page(self.input_tab_index))
        processing_action = view_menu.addAction("Processing")
        processing_action.setShortcut("Ctrl+2")
        processing_action.triggered.connect(
            lambda: self._switch_workflow_page(self.processing_tab_index))
        results_action = view_menu.addAction("Results")
        results_action.setShortcut("Ctrl+3")
        results_action.triggered.connect(
            lambda: self._switch_workflow_page(self.results_tab_index))

        help_menu = menubar.addMenu("&Help")
        about_action = help_menu.addAction("About GeoNet Adjust")
        about_action.triggered.connect(self.show_about)

    def _setup_workflow_header(self, parent_layout):
        """Create visible navigation for the data -> processing -> results flow."""
        header = QFrame()
        header.setObjectName("workflowHeader")
        layout = QHBoxLayout()
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("GeoNet Adjust")
        title.setObjectName("appTitle")
        subtitle = QLabel(
            "Input data, run one validated method, then review residuals, outliers and exports."
        )
        subtitle.setObjectName("appSubtitle")
        subtitle.setWordWrap(True)
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        layout.addLayout(title_layout, 2)

        self.workflow_step_buttons = {}
        for key, text, tooltip in (
            ("input", "1  Input Data", "Load station files and validate point roles."),
            ("processing", "2  Processing", "Select method, parameters and outlier workflow."),
            ("results", "3  Results", "Review results, diagnostics and exports."),
        ):
            button = QPushButton(text)
            button.setObjectName("stepButton")
            button.setCheckable(True)
            button.setMinimumHeight(38)
            button.setToolTip(tooltip)
            button.clicked.connect(
                lambda checked=False, page=key: self._switch_workflow_key(page))
            self.workflow_step_buttons[key] = button
            layout.addWidget(button)

        nav_layout = QVBoxLayout()
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_buttons = QHBoxLayout()
        self.workflow_back_btn = QPushButton("Back")
        self.workflow_back_btn.setObjectName("workflowNavButton")
        self.workflow_back_btn.clicked.connect(self._previous_workflow_page)
        nav_buttons.addWidget(self.workflow_back_btn)

        self.workflow_next_btn = QPushButton("Next")
        self.workflow_next_btn.setObjectName("workflowNavButton")
        self.workflow_next_btn.clicked.connect(self._next_workflow_page)
        nav_buttons.addWidget(self.workflow_next_btn)
        nav_layout.addLayout(nav_buttons)

        self.workflow_status_label = QLabel("Load station data to start.")
        self.workflow_status_label.setObjectName("workflowStatus")
        self.workflow_status_label.setMinimumWidth(265)
        nav_layout.addWidget(self.workflow_status_label)
        layout.addLayout(nav_layout, 1)

        header.setLayout(layout)
        parent_layout.addWidget(header)

    def _setup_main_tabs(self, parent_layout):
        """Create workflow pages controlled by the menu bar."""
        self.main_tabs = QStackedWidget()
        self.input_tab_index = self.main_tabs.addWidget(
            self._create_input_tab())
        self.processing_tab_index = self.main_tabs.addWidget(
            self._create_processing_tab())
        self.results_tab_index = self.main_tabs.addWidget(
            self._create_results_tab())
        self.main_tabs.currentChanged.connect(self._update_workflow_state)
        parent_layout.addWidget(self.main_tabs, 1)

    def _workflow_index_by_key(self, key):
        return {
            "input": getattr(self, "input_tab_index", 0),
            "processing": getattr(self, "processing_tab_index", 1),
            "results": getattr(self, "results_tab_index", 2),
        }.get(key, getattr(self, "input_tab_index", 0))

    def _workflow_key_by_index(self, index):
        return {
            getattr(self, "input_tab_index", 0): "input",
            getattr(self, "processing_tab_index", 1): "processing",
            getattr(self, "results_tab_index", 2): "results",
        }.get(index, "input")

    def _switch_workflow_key(self, key):
        self._switch_workflow_page(self._workflow_index_by_key(key))

    def _switch_workflow_page(self, index):
        if hasattr(self, "main_tabs"):
            self.main_tabs.setCurrentIndex(index)
            labels = {
                self.input_tab_index: "Input",
                self.processing_tab_index: "Processing",
                self.results_tab_index: "Results",
            }
            self.statusBar().showMessage(f"View: {labels.get(index, '')}")
            self._update_workflow_state()

    def _has_loaded_station_data(self):
        if not hasattr(self, "data_input"):
            return False
        stations = getattr(getattr(self.data_input, "survey_data", None), "stations", {})
        return bool(stations)

    def _has_displayed_results(self):
        return bool(getattr(getattr(self, "results_widget", None), "results", {}))

    def _workflow_counts(self):
        stations = {}
        if hasattr(self, "data_input"):
            stations = getattr(self.data_input.survey_data, "stations", {}) or {}
        point_count = 0
        control_count = 0
        icp_count = 0
        for station in stations.values():
            point_count += len(station.get("all_ids", []))
            control_count += len(station.get("ctrl_ids", []))
            icp_count += len(station.get("icp_ids", []))
        return len(stations), point_count, control_count, icp_count

    def _set_step_state(self, key, state):
        button = getattr(self, "workflow_step_buttons", {}).get(key)
        if button is None:
            return
        button.setProperty("stepState", state)
        button.setChecked(state == "current")
        button.style().unpolish(button)
        button.style().polish(button)

    def _update_workflow_state(self, *_args):
        if not hasattr(self, "main_tabs"):
            return
        current_index = self.main_tabs.currentIndex()
        current_key = self._workflow_key_by_index(current_index)
        has_data = self._has_loaded_station_data()
        has_results = self._has_displayed_results()

        states = {
            "input": "current" if current_key == "input" else ("done" if has_data else "pending"),
            "processing": "current" if current_key == "processing" else ("done" if has_results else "pending"),
            "results": "current" if current_key == "results" else ("done" if has_results else "pending"),
        }
        for key, state in states.items():
            self._set_step_state(key, state)

        if hasattr(self, "workflow_step_buttons"):
            self.workflow_step_buttons["processing"].setEnabled(True)
            self.workflow_step_buttons["results"].setEnabled(has_results)

        if hasattr(self, "workflow_back_btn"):
            self.workflow_back_btn.setEnabled(
                current_key != "input" and not self.processing_busy)
        if hasattr(self, "workflow_next_btn"):
            if current_key == "input":
                self.workflow_next_btn.setText("Next: Processing")
                self.workflow_next_btn.setEnabled(has_data and not self.processing_busy)
            elif current_key == "processing":
                self.workflow_next_btn.setText("Next: Results")
                self.workflow_next_btn.setEnabled(has_results and not self.processing_busy)
            else:
                self.workflow_next_btn.setText("Next")
                self.workflow_next_btn.setEnabled(False)

        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(has_data and not self.processing_busy)

        if hasattr(self, "workflow_status_label"):
            if self.processing_busy:
                status = "Processing is running. Results will open automatically."
            elif current_key == "input":
                if has_data:
                    stations, points, controls, icps = self._workflow_counts()
                    status = f"Loaded: {stations} stations, {points} points, {controls} controls, {icps} ICP."
                else:
                    status = "Load a station folder before processing."
            elif current_key == "processing":
                method = (
                    self._current_method_payload().get("method_key", "method")
                    if hasattr(self, "adjustment_method_combo")
                    else "method"
                )
                status = f"Ready to run {method}." if has_data else "Load data first."
            else:
                status = "Review outputs and export CSV/DXF/HTML." if has_results else "No result is available yet."
            self.workflow_status_label.setText(status)

        if hasattr(self, "processing_readiness_label"):
            method = (
                self._current_method_payload().get("method_key", "method")
                if hasattr(self, "adjustment_method_combo")
                else "method"
            )
            if self.processing_busy:
                readiness = "Running adjustment. Wait until the result tabs are updated."
            elif has_data:
                stations, points, controls, icps = self._workflow_counts()
                removal_mode = (
                    "automatic outlier removal"
                    if getattr(self, "auto_remove_outliers_check", None) is not None
                    and self.auto_remove_outliers_check.isChecked()
                    else "outlier review only"
                )
                readiness = (
                    f"Ready: {method} will run with {stations} stations, "
                    f"{points} points, {controls} control points and {icps} ICP points. "
                    f"Outlier mode: {removal_mode}."
                )
            else:
                readiness = "Load and validate a station folder before running an adjustment."
            if hasattr(self, "baarda_threshold_spin") and hasattr(self, "mad_threshold_spin") and has_data:
                readiness += (
                    f" Criteria: Baarda |w|>{self.baarda_threshold_spin.value():.2f}, "
                    f"MAD z>{self.mad_threshold_spin.value():.2f}."
                )
            self.processing_readiness_label.setText(readiness)

    def _previous_workflow_page(self):
        current_key = self._workflow_key_by_index(self.main_tabs.currentIndex())
        if current_key == "results":
            self._switch_workflow_key("processing")
        elif current_key == "processing":
            self._switch_workflow_key("input")

    def _next_workflow_page(self):
        current_key = self._workflow_key_by_index(self.main_tabs.currentIndex())
        if current_key == "input" and self._has_loaded_station_data():
            self._switch_workflow_key("processing")
        elif current_key == "processing" and self._has_displayed_results():
            self._switch_workflow_key("results")

    def _create_input_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        self.data_input = DataInputWidget()
        self.data_input.data_loaded_signal.connect(self._on_data_loaded)
        scroll.setWidget(self.data_input)
        layout.addWidget(scroll, 1)
        tab.setLayout(layout)
        return tab

    def _create_processing_tab(self):
        tab = QWidget()
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(8, 8, 8, 8)
        outer_layout.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        body = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 10, 4)
        layout.setSpacing(8)

        method_group = QGroupBox("Processing Method / Article Acronym")
        method_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #1976D2; }")
        method_group_layout = QVBoxLayout()

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method acronym:"))
        self.adjustment_method_combo = QComboBox()
        for method in GUI_METHODS:
            self.adjustment_method_combo.addItem(method["label"], method)
        self.adjustment_method_combo.setMinimumHeight(32)
        self.adjustment_method_combo.setMinimumWidth(420)
        self.adjustment_method_combo.setToolTip(
            "Select the processing method acronym used in the article.")
        method_layout.addWidget(self.adjustment_method_combo, 1)
        method_group_layout.addLayout(method_layout)

        self.selected_method_label = QLabel()
        self.selected_method_label.setWordWrap(True)
        self.selected_method_label.setMinimumHeight(76)
        self.selected_method_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.selected_method_label.setStyleSheet("""
            QLabel {
                background-color: #E8F5E9;
                color: #1B5E20;
                border: 1px solid #A5D6A7;
                border-radius: 4px;
                padding: 10px;
                font-weight: bold;
            }
        """)
        method_group_layout.addWidget(self.selected_method_label)
        method_group.setLayout(method_group_layout)
        layout.addWidget(method_group)

        self.method_options_group = self._create_method_options_group()
        layout.addWidget(self.method_options_group)

        run_group = QGroupBox("Run")
        run_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #2E7D32; }")
        run_layout = QVBoxLayout()
        action_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run Adjustment")
        self.run_btn.clicked.connect(self.run_adjustment)
        self.run_btn.setMinimumHeight(46)
        self.run_btn.setMinimumWidth(180)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #256D2A;
            }
            QPushButton:pressed {
                background-color: #1B5E20;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.run_btn.setToolTip("Run least squares adjustment on loaded data")
        action_layout.addWidget(self.run_btn, 2)

        self.advanced_outputs_check = QCheckBox(
            "Generate selected-method diagnostics")
        self.advanced_outputs_check.setChecked(True)
        self.advanced_outputs_check.setMinimumHeight(34)
        self.advanced_outputs_check.setMinimumWidth(240)
        self.advanced_outputs_check.setToolTip(
            "Generate article tables, outlier review and Plotly HTML only for the selected method"
        )
        action_layout.addWidget(self.advanced_outputs_check, 2)
        run_layout.addLayout(action_layout)

        self.processing_readiness_label = QLabel()
        self.processing_readiness_label.setWordWrap(True)
        self.processing_readiness_label.setStyleSheet("""
            QLabel {
                background-color: #F8FAFC;
                border: 1px solid #D8DEE6;
                border-radius: 5px;
                padding: 8px;
                color: #334155;
            }
        """)
        run_layout.addWidget(self.processing_readiness_label)

        run_group.setLayout(run_layout)
        layout.addWidget(run_group)
        layout.addStretch()

        body.setLayout(layout)
        scroll.setWidget(body)
        outer_layout.addWidget(scroll, 1)

        self.adjustment_method_combo.currentIndexChanged.connect(
            self._on_method_changed)
        self._on_method_changed(self.adjustment_method_combo.currentIndex())

        tab.setLayout(outer_layout)
        return tab

    def _create_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.results_widget = ResultsWidget()
        layout.addWidget(self.results_widget, 1)
        tab.setLayout(layout)
        return tab

    def _create_method_options_group(self):
        group = QGroupBox("Processing Variables")
        group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #2E7D32; margin-top: 8px; }")
        layout = QFormLayout()
        layout.setLabelAlignment(Qt.AlignRight)

        self.control_points_input = QLineEdit()
        self.control_points_input.setPlaceholderText(
            "Example: CP1 CP2 CP3 or leave blank for all")
        self.control_points_input.setToolTip(
            "Optional control point IDs used by the selected-method article diagnostics.")
        layout.addRow("Control points:", self.control_points_input)

        self.min_common_cp_spin = QSpinBox()
        self.min_common_cp_spin.setRange(3, 50)
        self.min_common_cp_spin.setValue(4)
        self.min_common_cp_spin.setToolTip(
            "Minimum common control points required between stations.")
        layout.addRow("Minimum common CPs:", self.min_common_cp_spin)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 1000)
        self.max_iter_spin.setValue(50)
        self.max_iter_spin.setToolTip(
            "Maximum iterations for iterative methods.")
        layout.addRow("Max iterations:", self.max_iter_spin)

        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setDecimals(12)
        self.tol_spin.setRange(1e-12, 1e-2)
        self.tol_spin.setSingleStep(1e-8)
        self.tol_spin.setValue(1e-8)
        self.tol_spin.setToolTip(
            "Numerical convergence tolerance for iterative methods.")
        layout.addRow("Tolerance:", self.tol_spin)

        self.apriori_sigma_spin = QDoubleSpinBox()
        self.apriori_sigma_spin.setDecimals(3)
        self.apriori_sigma_spin.setRange(0.001, 1000.0)
        self.apriori_sigma_spin.setSingleStep(0.1)
        self.apriori_sigma_spin.setValue(3.0)
        self.apriori_sigma_spin.setSuffix(" mm")
        self.apriori_sigma_spin.setToolTip(
            "A priori standard deviation used only for the global chi-square variance test. "
            "The adjustment still uses unit weights P=I."
        )
        layout.addRow("A priori sigma:", self.apriori_sigma_spin)

        outlier_row = QWidget()
        outlier_layout = QHBoxLayout()
        outlier_layout.setContentsMargins(0, 0, 0, 0)
        self.outlier_checks = {}
        for key, label in (
            ("baarda", "Baarda"),
            ("mad", "MAD"),
        ):
            check = QCheckBox(label)
            check.setChecked(True)
            check.setToolTip(
                "Used by the analyzer outlier workflow when the required residual diagnostics are available."
            )
            check.stateChanged.connect(lambda *_args: self._update_workflow_state())
            self.outlier_checks[key] = check
            outlier_layout.addWidget(check)
        outlier_layout.addStretch()
        outlier_row.setLayout(outlier_layout)
        layout.addRow("Outlier tests:", outlier_row)

        outlier_thresholds = QWidget()
        threshold_layout = QHBoxLayout()
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.setSpacing(8)

        threshold_layout.addWidget(QLabel("Baarda |w|:"))
        self.baarda_threshold_spin = QDoubleSpinBox()
        self.baarda_threshold_spin.setDecimals(2)
        self.baarda_threshold_spin.setRange(0.50, 10.00)
        self.baarda_threshold_spin.setSingleStep(0.10)
        self.baarda_threshold_spin.setValue(3.00)
        self.baarda_threshold_spin.setToolTip(
            "Critical standardized residual threshold for Baarda Data Snooping.")
        self.baarda_threshold_spin.valueChanged.connect(
            lambda *_args: self._update_workflow_state())
        threshold_layout.addWidget(self.baarda_threshold_spin)

        threshold_layout.addWidget(QLabel("MAD z:"))
        self.mad_threshold_spin = QDoubleSpinBox()
        self.mad_threshold_spin.setDecimals(2)
        self.mad_threshold_spin.setRange(0.50, 10.00)
        self.mad_threshold_spin.setSingleStep(0.10)
        self.mad_threshold_spin.setValue(3.50)
        self.mad_threshold_spin.setToolTip(
            "Robust modified z-score threshold for MAD residual screening.")
        self.mad_threshold_spin.valueChanged.connect(
            lambda *_args: self._update_workflow_state())
        threshold_layout.addWidget(self.mad_threshold_spin)
        threshold_layout.addStretch(1)
        outlier_thresholds.setLayout(threshold_layout)
        layout.addRow("Outlier limits:", outlier_thresholds)

        self.auto_remove_outliers_check = QCheckBox(
            "Remove outliers automatically during processing")
        self.auto_remove_outliers_check.setToolTip(
            "When enabled, the selected outlier tests are applied, flagged points are removed, and the selected method is rerun iteratively."
        )
        self.auto_remove_outliers_check.stateChanged.connect(
            lambda *_args: self._update_workflow_state())
        layout.addRow("Outlier removal:", self.auto_remove_outliers_check)

        outlier_note = QLabel(
            "Scale handling and quaternion constraints are defined by the selected method. "
            "With automatic removal disabled, Baarda/MAD only flag candidate points for review in Results."
        )
        outlier_note.setWordWrap(True)
        outlier_note.setStyleSheet("color: #555; font-size: 11px;")
        layout.addRow("", outlier_note)

        self.method_options_note = QLabel()
        self.method_options_note.setWordWrap(True)
        self.method_options_note.setStyleSheet("color: #555; font-size: 11px;")
        layout.addRow("", self.method_options_note)

        group.setLayout(layout)
        return group

    def _current_method_payload(self):
        payload = self.adjustment_method_combo.currentData()
        if isinstance(payload, dict):
            return payload
        text = self.adjustment_method_combo.currentText()
        if ": " in text:
            registration_type, method_class_name = text.split(": ", 1)
            method_key = self._method_key_for_analyzer(
                registration_type.lower(), method_class_name)
            return {
                "label": text,
                "registration_type": registration_type.lower(),
                "class_name": method_class_name,
                "method_key": method_key or method_class_name,
            }
        return {}

    def _on_method_changed(self, *_args):
        payload = self._current_method_payload()
        registration_type = payload.get("registration_type", "")
        method_class_name = payload.get("class_name", "")
        method_key = payload.get("method_key", method_class_name)
        method_class = globals().get(method_class_name)
        meta = method_metadata(method_class or method_key)
        article_name = meta.get(
            "article_name", payload.get("label", method_key))
        self.selected_method_label.setText(
            f"Highlighted method: {method_key} | {article_name}"
        )
        is_iterative = method_class_name in {
            "Tao2020_CMTLS_Rigorous",
            "CMTLS",
            "N_CUQ_LM",
        }
        self.max_iter_spin.setEnabled(is_iterative)
        self.tol_spin.setEnabled(is_iterative)
        notes = {
            "SVD-P": "Inputs: coordinate pairs and minimum common CPs. Scale is fixed to 1 in the article workflow.",
            "SCH-Q": "Inputs: coordinate pairs and minimum common CPs. Scale is fixed to 1 in the article workflow.",
            "Q-ST": "Inputs: coordinate pairs, pairwise distances and minimum common CPs. Scale is fixed to 1 to preserve the validated Q-ST configuration.",
            "CMTLS": "Inputs: coordinate pairs, rigorous CMTLS iterations and tolerance.",
            "TFS": "Inputs: coordinate pairs and minimum common CPs. Scale is fixed to 1 by the Total Free Station model.",
            "N-CUQ-LM": "Network inputs: station list, common CPs, iterations and tolerance. Unit-quaternion constraint is part of the method.",
        }
        reference = meta.get("reference", "")
        note = notes.get(method_key, "Inputs depend on the selected method.")
        self.method_options_note.setText(
            note + (f"\nReference: {reference}" if reference else ""))
        self._update_workflow_state()

    def _current_processing_options(self):
        outlier_methods = [
            key for key, check in self.outlier_checks.items()
            if check.isChecked()
        ]
        icp_points = sorted({
            str(point_id)
            for station in getattr(self.data_input.survey_data, "stations", {}).values()
            for point_id in station.get("icp_ids", [])
        })
        return {
            "control_points": self.control_points_input.text().strip(),
            "independent_check_points": icp_points,
            "min_common_cp": self.min_common_cp_spin.value(),
            "fixed_scale": True,
            "max_iter": self.max_iter_spin.value(),
            "tol": self.tol_spin.value(),
            "weight_constraint": 1e5,
            "apriori_sigma_mm": self.apriori_sigma_spin.value(),
            "use_quaternion_constraint": True,
            "outlier_methods": outlier_methods or ["none"],
            "outlier_sigma": self.baarda_threshold_spin.value(),
            "outlier_mad": self.mad_threshold_spin.value(),
            "auto_remove_outliers": self.auto_remove_outliers_check.isChecked(),
            "outlier_removal_iterations": 5,
            "outlier_max_remove_per_iteration": 1,
        }

    def _create_logo_widget(self, size=34):
        """
        Create a clickable logo widget that opens the GPHidro website.

        Returns:
            QLabel: Configured logo widget, or None if logo file not found
        """
        try:
            # Get the path to the logo file
            logo_path = os.path.join(os.path.dirname(
                os.path.dirname(os.path.dirname(__file__))), 'gphidro.png')

            if not os.path.exists(logo_path):
                self.logger.warning(f"Logo file not found: {logo_path}")
                return None

            # Create pixmap from logo file
            pixmap = QPixmap(logo_path)

            scaled_pixmap = pixmap.scaled(
                size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Create label with the logo
            logo_label = QLabel()
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setToolTip("Click to visit GPHidro website")
            # Change cursor to hand when hovering
            logo_label.setCursor(Qt.PointingHandCursor)

            # Make the logo clickable
            logo_label.mousePressEvent = self._open_gphidro_website

            # Add some styling
            logo_label.setStyleSheet("""
                QLabel {
                    border: 1px solid #ddd;
                    border-radius: 3px;
                    padding: 1px;
                    background-color: white;
                }
                QLabel:hover {
                    border: 1px solid #1976D2;
                    background-color: #f5f5f5;
                }
            """)

            return logo_label

        except Exception as e:
            self.logger.error(f"Failed to create logo widget: {e}")
            return None

    def _open_gphidro_website(self, event):
        """
        Open the GPHidro website in the default web browser.
        Does NOT close the application.

        Args:
            event: Mouse press event (ignored)
        """
        try:
            url = QUrl("https://gphidro.com.br/")
            QDesktopServices.openUrl(url)
            self.logger.info("GPHidro website opened successfully")
        except Exception as e:
            self.logger.error(f"Failed to open GPHidro website: {e}")
            QMessageBox.warning(
                self, "Error", "Failed to open website. Please visit https://gphidro.com.br/ manually.")
            # Do NOT close the application

    def _on_data_loaded(self, has_data: bool):
        """Move the workflow forward after data is loaded."""
        if has_data and hasattr(self, 'main_tabs'):
            self._switch_workflow_key("processing")
        self._update_workflow_state()

    def run_adjustment(self):
        """Run the TFS adjustment calculation."""
        if hasattr(self.data_input, 'prepare_points_for_processing'):
            if not self.data_input.prepare_points_for_processing():
                if hasattr(self, 'main_tabs'):
                    self.main_tabs.setCurrentIndex(self.input_tab_index)
                QMessageBox.warning(
                    self, "Point Input", "Please fix the point table before running processing.")
                return

        # Ensure station data is loaded for these methods
        if not hasattr(self.data_input.survey_data, 'stations') or not self.data_input.survey_data.stations:
            if hasattr(self, 'main_tabs'):
                self.main_tabs.setCurrentIndex(self.input_tab_index)
            QMessageBox.warning(
                self, "Warning", "Please load station data from a folder first.")
            return

        # The new registration methods expect a list of station dictionaries, not the SurveyData object directly.
        # The DataInputWidget's survey_data.stations attribute is a dictionary of stations.
        stations_list = list(self.data_input.survey_data.stations.values())
        if not stations_list:
            if hasattr(self, 'main_tabs'):
                self.main_tabs.setCurrentIndex(self.input_tab_index)
            QMessageBox.warning(
                self, "Warning", "No valid station data found in the loaded folder.")
            return

        self.results_widget.clear_results()
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.processing_busy = True
        self.run_btn.setEnabled(False)
        self.statusBar().showMessage("Running registration...")
        self._update_workflow_state()

        payload = self._current_method_payload()
        registration_type = str(payload.get("registration_type", "")).lower()
        method_class_name = payload.get("class_name")
        method_key = payload.get("method_key") or self._method_key_for_analyzer(
            registration_type, method_class_name)
        if not registration_type or not method_class_name:
            QMessageBox.warning(
                self, "Warning", "Please select a valid processing method.")
            self.progress_bar.setVisible(False)
            self.processing_busy = False
            self.run_btn.setEnabled(True)
            self._update_workflow_state()
            return

        # Start worker thread
        self.worker_thread = WorkerThread(self.data_input.survey_data)
        self.worker_thread.registration_type = registration_type
        self.worker_thread.method_class_name = method_class_name
        self.worker_thread.method_display_name = method_key or method_class_name
        self.last_processing_options = self._current_processing_options()
        auto_remove_outliers = bool(
            self.last_processing_options.get("auto_remove_outliers"))
        self.worker_thread.advanced_outputs_enabled = (
            (
                auto_remove_outliers or
                (
                    getattr(self, 'advanced_outputs_check', None) is not None and
                    self.advanced_outputs_check.isChecked()
                )
            ) and
            method_key is not None
        )
        self.worker_thread.advanced_data_folder = getattr(
            self.data_input, 'station_folder_path', '')
        self.worker_thread.advanced_method_key = method_key or ""
        self.worker_thread.advanced_output_dir = str(
            Path(__file__).resolve().parents[2] / "results" / "article_new_nomenclature_outputs")
        self.last_advanced_data_folder = getattr(
            self.data_input, 'station_folder_path', '')
        self.worker_thread.processing_options = dict(
            self.last_processing_options)
        self.worker_thread.outlier_action = (
            "remove" if auto_remove_outliers else "report"
        )

        # Connect signals
        self.worker_thread.finished.connect(self.on_calculation_finished)
        self.worker_thread.error.connect(self.on_calculation_error)
        self.worker_thread.progress.connect(self.on_calculation_progress)
        self.worker_thread.start()

    def on_calculation_progress(self, message: str):
        """Handle calculation progress updates."""
        self.statusBar().showMessage(message)

    def on_calculation_finished(self, results: Dict):
        """Handle successful calculation completion."""
        self.progress_bar.setVisible(False)
        self.processing_busy = False
        self.run_btn.setEnabled(True)

        # Display results
        # The results dictionary will now contain 'registration_type'
        self.results_widget.display_results(results)
        if results.get('advanced_output_dir') and results.get('advanced_method_key'):
            self.results_widget.display_audit_outputs(
                results['advanced_output_dir'], results['advanced_method_key'])
        elif results.get('advanced_error'):
            self.results_widget.display_advanced_error(
                results['advanced_error'])
            self.logger.warning(
                "Advanced output generation failed: %s", results['advanced_error'])

        self.statusBar().showMessage("Registration completed!")
        if hasattr(self, 'main_tabs'):
            self._switch_workflow_key("results")
        self._update_workflow_state()

        QMessageBox.information(
            self, "Success", "Adjustment completed successfully!")

    def run_outlier_removal_from_results(self, source_output_dir, method_key, manual_exclusions=None):
        """Run analyzer again with outlier removal only after inline user confirmation."""
        manual_exclusions = [
            str(item).strip()
            for item in (manual_exclusions or [])
            if str(item).strip()
        ]
        data_folder = self.last_advanced_data_folder or getattr(
            self.data_input, 'station_folder_path', '')
        if not data_folder or not os.path.isdir(data_folder):
            self.statusBar().showMessage("Cannot remove outliers: data folder is not available.")
            return
        if not method_key:
            self.statusBar().showMessage(
                "Cannot remove outliers: selected method is not available.")
            return

        source_output_dir = Path(source_output_dir)
        suffix = "_manual_outliers_removed" if manual_exclusions else "_outliers_removed"
        output_base = source_output_dir.with_name(
            f"{source_output_dir.name}{suffix}")
        output_dir = output_base
        counter = 2
        while output_dir.exists():
            output_dir = output_base.with_name(f"{output_base.name}_{counter}")
            counter += 1
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.processing_busy = True
        self.run_btn.setEnabled(False)
        if manual_exclusions:
            self.statusBar().showMessage(
                f"Reprocessing after manual exclusion of {len(manual_exclusions)} point(s)...")
        else:
            self.statusBar().showMessage("Reprocessing with confirmed outlier removal...")

        self.outlier_worker_thread = AnalyzerWorkerThread(
            data_folder=data_folder,
            output_dir=output_dir,
            method_key=method_key,
            processing_options=self.last_processing_options or self._current_processing_options(),
            outlier_action="remove",
            manual_exclusions=manual_exclusions,
        )
        self.outlier_worker_thread.progress.connect(
            self.on_calculation_progress)
        self.outlier_worker_thread.finished.connect(
            self.on_outlier_removal_finished)
        self.outlier_worker_thread.error.connect(self.on_outlier_removal_error)
        self.outlier_worker_thread.start()
        self._update_workflow_state()

    def on_outlier_removal_finished(self, results):
        self.progress_bar.setVisible(False)
        self.processing_busy = False
        self.run_btn.setEnabled(True)
        self.statusBar().showMessage("Outlier-removal analyzer outputs generated.")
        if results.get('advanced_output_dir') and results.get('advanced_method_key'):
            self.results_widget.display_audit_outputs(
                results['advanced_output_dir'], results['advanced_method_key'])
        if hasattr(self, 'main_tabs'):
            self._switch_workflow_key("results")
        self._update_workflow_state()

    def on_outlier_removal_error(self, message):
        self.progress_bar.setVisible(False)
        self.processing_busy = False
        self.run_btn.setEnabled(True)
        self.statusBar().showMessage("Outlier removal was not completed.")
        self.results_widget.display_advanced_error(
            f"Outlier removal failed:\n{message}")
        self._update_workflow_state()

    def on_calculation_error(self, error_msg: str):
        """Handle calculation errors."""
        self.progress_bar.setVisible(False)
        self.processing_busy = False
        self.run_btn.setEnabled(True)
        self.statusBar().showMessage("Error occurred")
        self._update_workflow_state()

        QMessageBox.critical(self, "Error", f"Adjustment failed: {error_msg}")

    def _method_key_for_analyzer(self, registration_type, method_name):
        if registration_type == "network" and method_name == "N_CUQ_LM":
            return "N-CUQ-LM"
        if registration_type == "network":
            return None
        return {
            "Method1_SequentialSVD": "SVD-P",
            "SVD_P": "SVD-P",
            "Schut1973_SpaceTime": "SCH-Q",
            "SCH_Q": "SCH-Q",
            "Stepien2020_QST": "Q-ST",
            "Q_ST": "Q-ST",
            "Tao2020_CMTLS_Rigorous": "CMTLS",
            "CMTLS": "CMTLS",
            "TotalFreeStation_TFS": "TFS",
            "TFS": "TFS",
        }.get(method_name)

    def show_about(self):
        """Show the about dialog."""
        about_dialog = AboutDialog(self)
        about_dialog.exec()  # Use exec() for modal dialog

    def refresh_data_display(self):
        """Refresh the data display after changes."""
        self.data_input.update_tables()
        self.data_input.update_statistics()


def launch_gui():
    """Launch the graphical user interface."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Set application properties
    app.setApplicationName("GeoNet Adjust")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("Victor Carvalho")

    # Create and show main window
    window = MainWindow()
    window.showMaximized()

    sys.exit(app.exec_())


if __name__ == "__main__":
    launch_gui()
