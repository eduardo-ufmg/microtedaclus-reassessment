"""Experiments documented in the Model Improvement chapter.

An experiment is a list of (variant, control) comparisons evaluated on the same
runs. The main experiment owns runs 0 to 93; the experiments of this chapter use
the runs that follow, so the final comparison is made on streams that took no
part in the development of the model:

    development  runs  94 to 125  every design choice was made on these runs
    validation   runs 126 to 219  the accepted models are confirmed on these runs
"""

from dataclasses import dataclass

from editorial_config import DATASET_ORDER

# The six non-scalability datasets of the main experiment and the two
# highest-dimensional scalability streams.
DATASETS = [*DATASET_ORDER, "SCL-32", "SCL-64"]

DEVELOPMENT_RUNS = range(94, 126)
VALIDATION_RUNS = range(126, 220)


@dataclass(frozen=True)
class Experiment:
    key: str
    title: str
    question: str
    comparisons: tuple  # of (variant, control)
    runs: range = DEVELOPMENT_RUNS

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
        "Which combination of corrections can replace the vectorized model?",
        _against(
            "vectorized",
            "corrected",
            "cc_corrected_mature",
            "cc_nearest_nopen",
            "cc_corrected_nopen",
            "cc_corrected_mature_nopen",
            "cc_nearest_nopen_wide",
        ),
    ),
    Experiment(
        "improvements_single",
        "Improvements, one at a time",
        "Which single change improves the corrected model?",
        _against(
            "corrected",
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
    ),
    Experiment(
        "m_function",
        "Reparameterizations of the m function",
        "Does a tunable m function improve the corrected model?",
        _against("corrected", "m_reparam", "m_constant", "m_asymptote"),
    ),
    Experiment(
        "improvements_combined",
        "Combined improvements",
        "Which combination of the helpful changes improves the corrected model without regression?",
        _against(
            "corrected",
            "improved",
            "ic_improved_single",
            "ic_improved_recency",
            "ic_improved_recency_single",
            "ic_improved_cap",
            "ic_noise_single",
            "ic_noise_recency",
        ),
    ),
    Experiment(
        "geometry",
        "Geometry of the microclusters",
        "Can the dimensionality penalty be removed, and does a full covariance help?",
        (
            ("ic_improved_nopen", "improved"),
            ("cov_directional", "ic_improved_nopen"),
            ("cov_bhattacharyya", "ic_improved_nopen"),
        ),
    ),
    Experiment(
        "improvements_refined",
        "Refinements of the improved model",
        "Which further change improves the improved model?",
        _against(
            "improved",
            "ic_improved_labels",
            "ic_improved_cap",
            "ic_improved_cap_recency",
            "ic_improved_cap_forget",
            "ic_improved_cap_labels",
        ),
    ),
    Experiment(
        "final_candidates",
        "Final candidates",
        "Which combination of the refinements becomes the final model?",
        _against("improved", "final_nocap", "final", "final_forget"),
    ),
    Experiment(
        "validation",
        "Validation on held-out runs",
        "Do the accepted steps hold on runs that did not guide any choice?",
        (
            ("corrected", "vectorized"),
            ("improved", "corrected"),
            ("final", "improved"),
            ("final", "vectorized"),
        ),
        runs=VALIDATION_RUNS,
    ),
]

EXPERIMENT_BY_KEY = {e.key: e for e in EXPERIMENTS}
