"""Experiments documented in the Model Improvement chapter.

An experiment is a list of (variant, control) comparisons evaluated on the same
runs. The main experiment owns runs 0 to 93; the experiments of this chapter use
the runs that follow, so the final comparison is made on streams that took no
part in the development of the model:

    development  runs  94 to 125  every design choice was made on these runs
    validation   runs 126 to 219  the accepted models are confirmed on these runs
"""

from dataclasses import dataclass, field

from editorial_config import DATASET_ORDER

# The six non-scalability datasets of the main experiment and the two
# highest-dimensional scalability streams.
DATASETS = [*DATASET_ORDER, "SCL-32", "SCL-64"]

DEVELOPMENT_RUNS = range(94, 126)
VALIDATION_RUNS = range(126, 220)

# Models accepted at the end of each stage
CORRECTED = "c_nearest"
IMPROVED = "ic_single_noise_relative"
FINAL = "ir_nopen"

STAGE_NAMES = {
    "vectorized": "vectorized",
    CORRECTED: "corrected",
    IMPROVED: "improved",
    FINAL: "MicroTEDAClus+",
}


@dataclass(frozen=True)
class Experiment:
    key: str
    title: str
    question: str
    comparisons: tuple  # of (variant, control)
    runs: range = DEVELOPMENT_RUNS
    stage_names: dict = field(
        default_factory=dict
    )  # accepted models shown by their stage name

    @property
    def variants(self) -> list[str]:
        """Every variant the experiment needs, controls first."""
        seen = []
        for variant, control in self.comparisons:
            for key in (control, variant):
                if key not in seen:
                    seen.append(key)
        return seen


def _against(control, *variants):
    return tuple((v, control) for v in variants)


EXPERIMENTS = [
    Experiment(
        "optimization",
        "Implementation optimization",
        "Does the vectorized implementation preserve the quality of the original one, and at what cost?",
        _against("original", "vectorized"),
    ),
    Experiment(
        "corrections_single",
        "Corrections, one at a time",
        "Which single correction changes the quality of the vectorized model?",
        _against(
            "vectorized",
            "c_born",
            "c_peak",
            "c_volume",
            "c_mature",
            "c_nopen",
            "c_normtyp",
            "c_local",
            "c_dsum",
            "c_dmax",
            "c_nearest",
        ),
    ),
    Experiment(
        "corrections_combined",
        "Combined corrections",
        "Does adding another correction improve the model with the nearest membership?",
        _against(
            "c_nearest",
            "cc_nearest_peak",
            "cc_nearest_peak_mature",
            "cc_nearest_nopen",
            "cc_nearest_peak_nopen",
            "cc_nearest_peak_mature_nopen",
        ),
    ),
    Experiment(
        "search_range",
        "Search range of the variance limit",
        "Does the fixed search range of the variance limit constrain the model?",
        _against(CORRECTED, "c_nearest_wide"),
        stage_names=STAGE_NAMES,
    ),
    Experiment(
        "improvements_single",
        "Improvements, one at a time",
        "Which single change improves the corrected model?",
        _against(
            CORRECTED,
            "i_cap",
            "i_soft",
            "i_noact",
            "i_g0",
            "i_link",
            "i_recency",
            "i_recency_inactive",
            "i_forget",
            "i_noise",
            "i_single",
            "i_chi",
            "i_relative",
        ),
        stage_names={CORRECTED: "corrected"},
    ),
    Experiment(
        "m_function",
        "Reparameterizations of the m function",
        "Does a tunable m function improve the corrected model?",
        _against(CORRECTED, "m_reparam", "m_constant", "m_asymptote"),
        stage_names={CORRECTED: "corrected"},
    ),
    Experiment(
        "improvements_combined",
        "Combined improvements",
        "Which combination of the changes that helped improves the corrected model without regression?",
        _against(
            CORRECTED,
            "ic_single_cap",
            "ic_single_noise",
            "ic_single_relative",
            "ic_cap_noise",
            "ic_cap_relative",
            "ic_noise_relative",
            "ic_single_cap_noise",
            "ic_single_cap_relative",
            "ic_single_noise_relative",
            "ic_cap_noise_relative",
            "ic_single_cap_noise_relative",
        ),
        stage_names={CORRECTED: "corrected"},
    ),
    Experiment(
        "improvements_selection",
        "Direct comparisons between combinations",
        "Is the combination with the most gains improved upon by another one?",
        (
            ("ic_single_noise_relative", "ic_single_relative"),
            ("ic_single_noise_relative", "ic_noise_relative"),
            ("ic_single_noise_relative", "ic_cap_noise_relative"),
            ("ic_cap_noise_relative", "ic_noise_relative"),
            ("ic_single_cap_noise_relative", "ic_single_noise_relative"),
            ("ic_single_cap_noise_relative", "ic_cap_noise_relative"),
        ),
    ),
    Experiment(
        "refinements",
        "Refinements of the improved model",
        "Which further change improves the improved model?",
        _against(IMPROVED, "ir_nopen", "ir_labels", "ir_forget", "ir_recency"),
        stage_names={IMPROVED: "improved"},
    ),
    Experiment(
        "geometry",
        "Covariance of the microclusters",
        "Does a full covariance improve the isotropic model?",
        _against(FINAL, "cov_directional", "cov_bhattacharyya"),
        stage_names={FINAL: "MicroTEDAClus+"},
    ),
    Experiment(
        "density_revisited",
        "Order-independent densities in \\MicroTEDAClusPlus{}",
        "Do the order-independent densities, rejected as corrections, cost quality in MicroTEDAClus+?",
        _against(FINAL, "fd_peak", "fd_volume"),
        stage_names={FINAL: "MicroTEDAClus+"},
    ),
    Experiment(
        "validation",
        "Validation on held-out runs",
        "Do the accepted models hold on runs that did not guide any choice?",
        (
            (CORRECTED, "vectorized"),
            (IMPROVED, CORRECTED),
            (FINAL, IMPROVED),
            (FINAL, "vectorized"),
        ),
        runs=VALIDATION_RUNS,
        stage_names=STAGE_NAMES,
    ),
]

EXPERIMENT_BY_KEY = {e.key: e for e in EXPERIMENTS}
