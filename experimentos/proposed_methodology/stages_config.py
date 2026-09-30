"""Declarative stage registry and configuration for experimental orchestration."""

import os
from dataclasses import dataclass, field
import numpy as np

from editorial_config import ITERATIVE_DATASET_ORDER, dataset_display_name

# Default randomized runs per task
DEFAULT_RUNS = 32


@dataclass(frozen=True)
class Stage:
    """Specification of an evaluation stage across thesis chapters."""

    stage_id: str
    chapter: str  # "CorrectedModel" or "ImprovedModel"
    model_tag: str  # Focal model introduced in this stage
    models: list[str]  # All models evaluated in this stage's tables
    control_model: str  # Baseline for pairwise Wilcoxon signed-rank tests
    artifact_dir: str  # Output directory name inside monografia/{chapter}/
    label_prefix: str  # Prefix for LaTeX labels (e.g. "orderdep_")
    datasets: list[str] = field(default_factory=lambda: list(ITERATIVE_DATASET_ORDER))


# -----------------------------------------------------------------------------
# Declarative Stages Registry
# -----------------------------------------------------------------------------

# Chapter 3: Model Correction Stages
STAGE_ORDER_DEPENDENCY = Stage(
    stage_id="order_dependency",
    chapter="CorrectedModel",
    model_tag="MicroTEDAClusCorrected_orderdepend",
    models=[
        "MicroTEDAClus",
        "MicroTEDAClusCorrected_orderdepend",
    ],
    control_model="MicroTEDAClus",
    artifact_dir="artifacts_order_dependency",
    label_prefix="orderdep_",
)

STAGE_DIMENSION_PENALTY = Stage(
    stage_id="dimension_penalty",
    chapter="CorrectedModel",
    model_tag="MicroTEDAClusCorrected_dimpen",
    models=[
        "MicroTEDAClus",
        "MicroTEDAClusCorrected_orderdepend",
        "MicroTEDAClusCorrected_dimpen",
    ],
    control_model="MicroTEDAClusCorrected_orderdepend",
    artifact_dir="artifacts_dimension_penalty",
    label_prefix="dimpen_",
)

STAGE_TYPICALITY_MATH = Stage(
    stage_id="typicality_math",
    chapter="CorrectedModel",
    model_tag="MicroTEDAClusCorrected_math",
    models=[
        "MicroTEDAClus",
        "MicroTEDAClusCorrected_orderdepend",
        "MicroTEDAClusCorrected_dimpen",
        "MicroTEDAClusCorrected_math",
    ],
    control_model="MicroTEDAClusCorrected_dimpen",
    artifact_dir="artifacts_math",
    label_prefix="math_",
)

STAGE_DECAY = Stage(
    stage_id="decay",
    chapter="CorrectedModel",
    model_tag="MicroTEDAClusCorrected_decay",
    models=[
        "MicroTEDAClus",
        "MicroTEDAClusCorrected_orderdepend",
        "MicroTEDAClusCorrected_dimpen",
        "MicroTEDAClusCorrected_math",
        "MicroTEDAClusCorrected_decay",
    ],
    control_model="MicroTEDAClusCorrected_math",
    artifact_dir="artifacts_decay",
    label_prefix="decay_",
)

STAGE_MACROCLUSTER_ACTIVATION = Stage(
    stage_id="macrocluster_activation",
    chapter="CorrectedModel",
    model_tag="MicroTEDAClusCorrected_macro",
    models=[
        "MicroTEDAClus",
        "MicroTEDAClusCorrected_orderdepend",
        "MicroTEDAClusCorrected_dimpen",
        "MicroTEDAClusCorrected_math",
        "MicroTEDAClusCorrected_decay",
        "MicroTEDAClusCorrected_macro",
    ],
    control_model="MicroTEDAClusCorrected_decay",
    artifact_dir="artifacts_macrocluster_activation",
    label_prefix="macro_",
)

STAGE_CORRECTED_FINAL = Stage(
    stage_id="corrected_final",
    chapter="CorrectedModel",
    model_tag="MicroTEDAClusCorrected_final",
    models=[
        "MicroTEDAClus",
        "MicroTEDAClusCorrected_orderdepend",
        "MicroTEDAClusCorrected_dimpen",
        "MicroTEDAClusCorrected_math",
        "MicroTEDAClusCorrected_decay",
        "MicroTEDAClusCorrected_macro",
        "MicroTEDAClusCorrected_final",
    ],
    control_model="MicroTEDAClusCorrected_macro",
    artifact_dir="artifacts_corrected_final",
    label_prefix="final_",
)

# Chapter 4: Improved Model Stages
STAGE_BASE_IMPROVED = Stage(
    stage_id="base_improved",
    chapter="ImprovedModel",
    model_tag="MicroTEDAClusImproved_base",
    models=[
        "MicroTEDAClus",
        "MicroTEDAClusCorrected_final",
        "MicroTEDAClusImproved_base",
    ],
    control_model="MicroTEDAClusCorrected_final",
    artifact_dir="artifacts_base_improved",
    label_prefix="base_imp_",
)

STAGE_M_FUNCTION = Stage(
    stage_id="m_function",
    chapter="ImprovedModel",
    model_tag="MicroTEDAClusImproved_reparam",
    models=[
        "MicroTEDAClus",
        "MicroTEDAClusImproved_base",
        "MicroTEDAClusImproved_param_orig",
        "MicroTEDAClusImproved_reparam",
        "MicroTEDAClusImproved_const_up",
        "MicroTEDAClusImproved_scalar_m",
        "MicroTEDAClusImproved_const_m",
    ],
    control_model="MicroTEDAClusImproved_base",
    artifact_dir="artifacts_m_function",
    label_prefix="mfunc_",
)

ALL_STAGES = [
    STAGE_ORDER_DEPENDENCY,
    STAGE_DIMENSION_PENALTY,
    STAGE_TYPICALITY_MATH,
    STAGE_DECAY,
    STAGE_MACROCLUSTER_ACTIVATION,
    STAGE_CORRECTED_FINAL,
    STAGE_BASE_IMPROVED,
    STAGE_M_FUNCTION,
]

STAGE_BY_ID: dict[str, Stage] = {stage.stage_id: stage for stage in ALL_STAGES}


# -----------------------------------------------------------------------------
# Canonical LaTeX Headers and Plot Colors
# -----------------------------------------------------------------------------

MODEL_SHORT_HEADERS = {
    # Baseline
    "MicroTEDAClus": r"\modelname{baseline}",
    # Chapter 3 Corrected Models
    "MicroTEDAClusCorrected_orderdepend": r"\modelname{order-independent}",
    "MicroTEDAClusCorrected_dimpen": r"\modelname{no-dim-penalty}",
    "MicroTEDAClusCorrected_math": r"\modelname{corr-typicality-math}",
    "MicroTEDAClusCorrected_decay": r"\modelname{decay}",
    "MicroTEDAClusCorrected_macro": r"\modelname{macrocluster-activation}",
    "MicroTEDAClusCorrected_final": r"\modelname{corrected-final}",
    # Chapter 4 Improved Models
    "MicroTEDAClusImproved_base": r"\modelname{base-improved}",
    "MicroTEDAClusImproved_param_orig": r"\modelname{param-orig-m}",
    "MicroTEDAClusImproved_reparam": r"\modelname{reparam-m}",
    "MicroTEDAClusImproved_const_up": r"\modelname{const-up}",
    "MicroTEDAClusImproved_scalar_m": r"\modelname{scalar-m}",
    "MicroTEDAClusImproved_const_m": r"\modelname{const-m}",
}

MODEL_COLORS = {
    "MicroTEDAClus": "xkcd:teal",
    "MicroTEDAClusCorrected_orderdepend": "xkcd:orange",
    "MicroTEDAClusCorrected_dimpen": "xkcd:light green",
    "MicroTEDAClusCorrected_math": "xkcd:magenta",
    "MicroTEDAClusCorrected_decay": "xkcd:yellow",
    "MicroTEDAClusCorrected_macro": "xkcd:sky blue",
    "MicroTEDAClusCorrected_final": "xkcd:grey",
    "MicroTEDAClusImproved_base": "xkcd:lime green",
    "MicroTEDAClusImproved_param_orig": "xkcd:light purple",
    "MicroTEDAClusImproved_reparam": "xkcd:violet",
    "MicroTEDAClusImproved_const_up": "xkcd:dark green",
    "MicroTEDAClusImproved_scalar_m": "xkcd:turquoise",
    "MicroTEDAClusImproved_const_m": "xkcd:lavender",
}


# -----------------------------------------------------------------------------
# Statistical Metrics Metadata
# -----------------------------------------------------------------------------

READABLE_METRICS = {
    "ari_overall": "Overall Adjusted Rand Index",
    "ari_steadystate": "Steady-State Adjusted Rand Index",
    "ari_fifth": "Adjusted Rand Index at the Fifth Batch",
    "ari_diff": "Maximum ARI Drop",
    "time_predict_cum_ms": "Cumulative Prediction Time [ms]",
    "time_fit_cum_ms": "Cumulative Fitting Time [ms]",
    "memory_max_bytes": "Maximum Memory Footprint [B]",
    "memory_steadystate_bytes": "Steady-State Memory Footprint [B]",
    "mc_max": "Maximum Microcluster Count",
    "mc_steadystate": "Steady-State Microcluster Count",
}

STATS_METRICS = {
    "ari_overall": {"dir": "max", "round": lambda x: np.round(x, 3), "fmt": "{:.3f}"},
    "ari_steadystate": {
        "dir": "max",
        "round": lambda x: np.round(x, 3),
        "fmt": "{:.3f}",
    },
    "ari_fifth": {"dir": "max", "round": lambda x: np.round(x, 3), "fmt": "{:.3f}"},
    "ari_diff": {"dir": "min", "round": lambda x: np.round(x, 3), "fmt": "{:.3f}"},
    "time_predict_cum_ms": {
        "dir": "min",
        "round": lambda x: np.round(x, -1),
        "fmt": "{:.1f}",
    },
    "time_fit_cum_ms": {
        "dir": "min",
        "round": lambda x: np.round(x, -1),
        "fmt": "{:.1f}",
    },
    "memory_max_bytes": {
        "dir": "min",
        "round": lambda x: np.round(x / 1024) * 1024,
        "fmt": "{:.0f}",
    },
    "memory_steadystate_bytes": {
        "dir": "min",
        "round": lambda x: np.round(x / 1024) * 1024,
        "fmt": "{:.0f}",
    },
    "mc_max": {"dir": "min", "round": lambda x: np.round(x), "fmt": "{:.0f}"},
    "mc_steadystate": {"dir": "min", "round": lambda x: np.round(x), "fmt": "{:.0f}"},
}


# -----------------------------------------------------------------------------
# Dynamic Model Class Resolution
# -----------------------------------------------------------------------------


def get_model_registry() -> dict[str, tuple[type, type]]:
    """Dynamically resolve available model classes and their search spaces."""
    registry = {}

    # Baseline
    try:
        from models.MicroTEDAClus.MicroTEDAClus import (
            MicroTEDAClus,
            MicroTEDAClusParameterSearchSpace,
        )

        registry["MicroTEDAClus"] = (MicroTEDAClus, MicroTEDAClusParameterSearchSpace)
    except ImportError:
        pass

    # Chapter 3 Corrected Models
    try:
        from models.MicroTEDAClusCorrected.MicroTEDAClusCorrected_orderdepend import (
            MicroTEDAClusCorrectedOrderdepend,
            MicroTEDAClusCorrectedOrderdependParameterSearchSpace,
        )

        registry["MicroTEDAClusCorrected_orderdepend"] = (
            MicroTEDAClusCorrectedOrderdepend,
            MicroTEDAClusCorrectedOrderdependParameterSearchSpace,
        )
    except ImportError:
        pass

    try:
        from models.MicroTEDAClusCorrected.MicroTEDAClusCorrected_dimpen import (
            MicroTEDAClusCorrectedDimpen,
            MicroTEDAClusCorrectedDimpenParameterSearchSpace,
        )

        registry["MicroTEDAClusCorrected_dimpen"] = (
            MicroTEDAClusCorrectedDimpen,
            MicroTEDAClusCorrectedDimpenParameterSearchSpace,
        )
    except ImportError:
        pass

    try:
        from models.MicroTEDAClusCorrected.MicroTEDAClusCorrected_math import (
            MicroTEDAClusCorrectedMath,
            MicroTEDAClusCorrectedMathParameterSearchSpace,
        )

        registry["MicroTEDAClusCorrected_math"] = (
            MicroTEDAClusCorrectedMath,
            MicroTEDAClusCorrectedMathParameterSearchSpace,
        )
    except ImportError:
        pass

    try:
        from models.MicroTEDAClusCorrected.MicroTEDAClusCorrected_decay import (
            MicroTEDAClusCorrectedDecay,
            MicroTEDAClusCorrectedDecayParameterSearchSpace,
        )

        registry["MicroTEDAClusCorrected_decay"] = (
            MicroTEDAClusCorrectedDecay,
            MicroTEDAClusCorrectedDecayParameterSearchSpace,
        )
    except ImportError:
        pass

    try:
        from models.MicroTEDAClusCorrected.MicroTEDAClusCorrected_macro import (
            MicroTEDAClusCorrectedMacro,
            MicroTEDAClusCorrectedMacroParameterSearchSpace,
        )

        registry["MicroTEDAClusCorrected_macro"] = (
            MicroTEDAClusCorrectedMacro,
            MicroTEDAClusCorrectedMacroParameterSearchSpace,
        )
    except ImportError:
        pass

    try:
        from models.MicroTEDAClusCorrected.MicroTEDAClusCorrected_final import (
            MicroTEDAClusCorrectedFinal,
            MicroTEDAClusCorrectedFinalParameterSearchSpace,
        )

        registry["MicroTEDAClusCorrected_final"] = (
            MicroTEDAClusCorrectedFinal,
            MicroTEDAClusCorrectedFinalParameterSearchSpace,
        )
    except ImportError:
        pass

    # Chapter 4 Improved Models
    try:
        from models.MicroTEDAClusImproved.MicroTEDAClusImproved_base import (
            MicroTEDAClusImproved_base,
            MicroTEDAClusImproved_baseParameterSearchSpace,
        )

        registry["MicroTEDAClusImproved_base"] = (
            MicroTEDAClusImproved_base,
            MicroTEDAClusImproved_baseParameterSearchSpace,
        )
    except ImportError:
        pass

    try:
        from models.MicroTEDAClusImproved.m_func_investigation.MicroTEDAClusImproved_param_orig import (
            MicroTEDAClusImproved_param_orig,
            MicroTEDAClusImproved_param_origParameterSearchSpace,
        )

        registry["MicroTEDAClusImproved_param_orig"] = (
            MicroTEDAClusImproved_param_orig,
            MicroTEDAClusImproved_param_origParameterSearchSpace,
        )
    except ImportError:
        pass

    try:
        from models.MicroTEDAClusImproved.m_func_investigation.MicroTEDAClusImproved_reparam import (
            MicroTEDAClusImproved_reparam,
            MicroTEDAClusImproved_reparamParameterSearchSpace,
        )

        registry["MicroTEDAClusImproved_reparam"] = (
            MicroTEDAClusImproved_reparam,
            MicroTEDAClusImproved_reparamParameterSearchSpace,
        )
    except ImportError:
        pass

    try:
        from models.MicroTEDAClusImproved.m_func_investigation.MicroTEDAClusImproved_const_up import (
            MicroTEDAClusImproved_const_up,
            MicroTEDAClusImproved_const_upParameterSearchSpace,
        )

        registry["MicroTEDAClusImproved_const_up"] = (
            MicroTEDAClusImproved_const_up,
            MicroTEDAClusImproved_const_upParameterSearchSpace,
        )
    except ImportError:
        pass

    try:
        from models.MicroTEDAClusImproved.m_func_investigation.MicroTEDAClusImproved_scalar_m import (
            MicroTEDAClusImproved_scalar_m,
            MicroTEDAClusImproved_scalar_mParameterSearchSpace,
        )

        registry["MicroTEDAClusImproved_scalar_m"] = (
            MicroTEDAClusImproved_scalar_m,
            MicroTEDAClusImproved_scalar_mParameterSearchSpace,
        )
    except ImportError:
        pass

    try:
        from models.MicroTEDAClusImproved.m_func_investigation.MicroTEDAClusImproved_const_m import (
            MicroTEDAClusImproved_const_m,
            MicroTEDAClusImproved_const_mParameterSearchSpace,
        )

        registry["MicroTEDAClusImproved_const_m"] = (
            MicroTEDAClusImproved_const_m,
            MicroTEDAClusImproved_const_mParameterSearchSpace,
        )
    except ImportError:
        pass

    return registry
