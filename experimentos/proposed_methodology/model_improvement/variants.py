"""Model variants studied in the Model Improvement chapter.

A variant is a fixed set of engine options plus the hyperparameters that the
tuner searches. Every variant is built from the options of its base model, so
the registry reads as the sequence of changes that were tried:

    original -> vectorized -> corrected -> improved -> final
"""

from dataclasses import dataclass, field
from functools import partial
from itertools import combinations

BASE_SPACE = {
    "variance_limit": ("float", 0.0001, 100.0, True),
    "decay": ("int", 10, 10000, True),
}
RELATIVE_SPACE = {"variance_limit": ("float", 1e-6, 1.0, True)}
NOISE_SPACE = {"noise_m": ("float", 1.0, 10.0, False)}
FORGET_SPACE = {"s_max": ("float", 10.0, 10000.0, True)}


@dataclass(frozen=True)
class Variant:
    key: str
    label: str  # name used in the tables of the thesis
    summary: str  # what changes relative to the base model
    options: dict = field(default_factory=dict)
    space: dict = field(default_factory=dict)
    implementation: str = "engine"  # engine | covariance | original


VARIANTS: dict[str, Variant] = {}


def _add(key, label, summary, options=None, space=None, implementation="engine"):
    VARIANTS[key] = Variant(
        key, label, summary, dict(options or {}), dict(space or {}), implementation
    )
    return VARIANTS[key]


# ----------------------------------------------------------------------------
# Implementations of the original algorithm
# ----------------------------------------------------------------------------
_add(
    "original",
    "original",
    "reference implementation of MicroTEDAClus",
    implementation="original",
)
_add("vectorized", "vectorized", "the same algorithm on the vectorized engine")

# ----------------------------------------------------------------------------
# Corrections, one at a time
# ----------------------------------------------------------------------------
_add(
    "c_born",
    "born-inactive",
    "new microclusters are created inactive",
    {"born_active": False},
)
_add(
    "c_peak",
    "peak-density",
    r"density $2S$, the density at the microcluster's own mean",
    {"density": "peak"},
)
_add("c_volume", "volume-density", r"density $S/\sigma^2$", {"density": "volume"})
_add(
    "c_mature",
    "mature-mean",
    "activation mean over mature microclusters only",
    {"activation_scope": "mature"},
)
_add("c_nopen", "no-penalty", "no dimensionality penalty", {"dim_penalty": False})
_add(
    "c_normtyp",
    "norm-typicality",
    "membership by normalized typicality",
    {"membership": "norm_typicality"},
)
_add(
    "c_local",
    "local-weights",
    "membership with densities evaluated at the sample",
    {"membership": "local_weights"},
)
_add(
    "c_dsum",
    "density-sum",
    "membership by the sum of densities at the sample",
    {"membership": "density_sum"},
)
_add(
    "c_dmax",
    "density-max",
    "membership by the largest density at the sample",
    {"membership": "density_max"},
)
_add(
    "c_nearest",
    "nearest",
    r"membership by the smallest $d^2/\sigma^2$",
    {"membership": "nearest"},
)

# ----------------------------------------------------------------------------
# Combined corrections, on top of the single correction that improved the model
# ----------------------------------------------------------------------------
NEAREST = {"membership": "nearest"}
NEAREST_PEAK = {**NEAREST, "density": "peak"}

_add("cc_nearest_peak", "nearest + peak-density", "", NEAREST_PEAK)
_add(
    "cc_nearest_peak_mature",
    "nearest + peak-density + mature-mean",
    "",
    {**NEAREST_PEAK, "activation_scope": "mature"},
)
_add("cc_nearest_nopen", "nearest + no-penalty", "", {**NEAREST, "dim_penalty": False})
_add(
    "cc_nearest_peak_nopen",
    "nearest + peak-density + no-penalty",
    "",
    {**NEAREST_PEAK, "dim_penalty": False},
)
_add(
    "cc_nearest_peak_mature_nopen",
    "nearest + peak-density + mature-mean + no-penalty",
    "",
    {**NEAREST_PEAK, "activation_scope": "mature", "dim_penalty": False},
)
_add(
    "c_nearest_wide",
    "wide-range",
    r"variance limit searched in $[10^{-4}, 10^{4}]$",
    NEAREST,
    {"variance_limit": ("float", 0.0001, 10000.0, True)},
)

# The corrected model is the vectorized model with the nearest membership (c_nearest).
CORRECTED = NEAREST

# ----------------------------------------------------------------------------
# Improvements on the corrected model, one at a time
# ----------------------------------------------------------------------------
_add(
    "i_cap",
    "variance-cap",
    "variance limit applied to microclusters of every size",
    {**CORRECTED, "cap_variance": True},
)
_add(
    "i_soft",
    "soft-activation",
    r"activation threshold $f \cdot \mathrm{mean}(D)$, $f$ tuned in $[0, 1.5]$",
    CORRECTED,
    {"activation_factor": ("float", 0.0, 1.5, False)},
)
_add(
    "i_noact",
    "no-activation",
    "every mature microcluster is active",
    {**CORRECTED, "activation": "none"},
)
_add(
    "i_g0",
    "single-graph",
    r"macroclusters taken from $\mathbb{G}_0$",
    {**CORRECTED, "topology": "g0"},
)
_add(
    "i_link",
    "link-factor",
    r"intersection $\mathrm{dist} \le \ell (\sigma_i + \sigma_j)$, $\ell$ tuned in $[0.5, 4]$",
    CORRECTED,
    {"link": ("float", 0.5, 4.0, False)},
)
_add(
    "i_recency",
    "recency",
    "every microcluster not updated by a sample decays",
    {**CORRECTED, "life_mode": "recency"},
)
_add(
    "i_recency_inactive",
    "recency-inactive",
    "inactive microclusters not updated by a sample decay",
    {**CORRECTED, "life_mode": "recency_inactive"},
)
_add(
    "i_forget",
    "forgetting",
    r"effective count capped at $S_{\max}$, tuned",
    CORRECTED,
    FORGET_SPACE,
)
_add(
    "i_noise",
    "noise-label",
    r"samples beyond $n_\sigma$ of every active microcluster are noise",
    CORRECTED,
    NOISE_SPACE,
)
_add(
    "i_single",
    "single-assign",
    "a sample updates only the nearest accepting microcluster",
    {**CORRECTED, "single_assign": True},
)
_add(
    "i_chi",
    "chi-threshold",
    "dimension-aware outlier threshold (without the penalty)",
    {**CORRECTED, "dim_penalty": False, "chi_dim": True},
)
_add(
    "i_relative",
    "relative-limit",
    "variance limit relative to the stream's variance",
    {**CORRECTED, "relative_limit": True},
    RELATIVE_SPACE,
)

# ----------------------------------------------------------------------------
# m function, on the corrected model
# ----------------------------------------------------------------------------
_add(
    "m_reparam",
    "m-reparam",
    r"$m(3) = 1$, $m(S_{95}) = 0.95\,m_{\max}$; $m_{\max}$ and $S_{95}$ tuned",
    {**CORRECTED, "m_mode": "reparam"},
    {"m_up": ("float", 1.05, 10.0, False), "m_a": ("float", 5.0, 5000.0, True)},
)
_add(
    "m_constant",
    "m-constant",
    r"constant $m$, tuned in $[1, 5]$",
    {**CORRECTED, "m_mode": "const"},
    {"m_up": ("float", 1.0, 5.0, False)},
)
_add(
    "m_asymptote",
    "m-asymptote",
    r"original sigmoid with its asymptote tuned in $[1.05, 10]$",
    CORRECTED,
    {"m_up": ("float", 1.05, 10.0, False)},
)


# ----------------------------------------------------------------------------
# Combined improvements: every combination of the four changes that helped
# ----------------------------------------------------------------------------
COMPONENTS = {
    "single": ("single", {"single_assign": True}, {}),
    "cap": ("cap", {"cap_variance": True}, {}),
    "noise": ("noise", {}, NOISE_SPACE),
    "relative": ("relative", {"relative_limit": True}, RELATIVE_SPACE),
}


def combination(*names) -> tuple[str, dict, dict]:
    """Label, options and search space of the corrected model plus the named changes."""
    options, space = dict(CORRECTED), {}
    for name in names:
        options.update(COMPONENTS[name][1])
        space.update(COMPONENTS[name][2])
    return " + ".join(COMPONENTS[name][0] for name in names), options, space


for size in (2, 3, 4):
    for names in combinations(COMPONENTS, size):
        _add(
            "ic_" + "_".join(names),
            *combination(*names)[:1],
            "",
            *combination(*names)[1:]
        )


# ----------------------------------------------------------------------------
# Refinements of the improved model (single-assign + noise-label + relative-limit)
# ----------------------------------------------------------------------------
_, IMPROVED, IMPROVED_SPACE = combination("single", "noise", "relative")

_add(
    "ir_nopen",
    "improved + no-penalty",
    "",
    {**IMPROVED, "dim_penalty": False},
    IMPROVED_SPACE,
)
_add(
    "ir_labels",
    "improved + persistent-labels",
    "macroclusters keep their label across updates",
    {**IMPROVED, "stable_labels": True},
    IMPROVED_SPACE,
)
_add(
    "ir_forget",
    "improved + forgetting",
    "",
    IMPROVED,
    {**IMPROVED_SPACE, **FORGET_SPACE},
)
_add(
    "ir_recency",
    "improved + recency",
    "",
    {**IMPROVED, "life_mode": "recency"},
    IMPROVED_SPACE,
)

# Geometry: full covariance against its isotropic counterpart (improved + no-penalty)
_add(
    "cov_directional",
    "covariance, directional link",
    "full covariance and Mahalanobis distance; intersection along the line that joins the means",
    {**IMPROVED, "link_mode": "directional"},
    IMPROVED_SPACE,
    implementation="covariance",
)
_add(
    "cov_bhattacharyya",
    "covariance, Bhattacharyya link",
    "full covariance and Mahalanobis distance; intersection by the Bhattacharyya distance",
    {**IMPROVED, "link_mode": "bhattacharyya"},
    IMPROVED_SPACE,
    implementation="covariance",
)


# ----------------------------------------------------------------------------
# The final model (improved + no-penalty) and the densities revisited on it
# ----------------------------------------------------------------------------
FINAL = {**IMPROVED, "dim_penalty": False}

_add(
    "fd_peak", "final + peak-density", "", {**FINAL, "density": "peak"}, IMPROVED_SPACE
)
_add(
    "fd_volume",
    "final + volume-density",
    "",
    {**FINAL, "density": "volume"},
    IMPROVED_SPACE,
)


def make_space(spec):
    class Space:
        def sample_parameters(self, trial):
            params = {}
            for name, (kind, low, high, log) in spec.items():
                if kind == "float":
                    params[name] = trial.suggest_float(name, low, high, log=log)
                else:
                    params[name] = trial.suggest_int(name, low, high, log=log)
            return params

    return Space


def resolve(key: str):
    """Return (model_class, search_space_class) of a variant, as the protocol expects them."""
    variant = VARIANTS[key]
    if variant.implementation == "original":
        from models.MicroTEDAClus.MicroTEDAClus import (
            MicroTEDAClus,
            MicroTEDAClusParameterSearchSpace,
        )

        return MicroTEDAClus, MicroTEDAClusParameterSearchSpace
    if variant.implementation == "covariance":
        from model_improvement.engine_cov import MicroTEDAClusCovarianceEngine as Engine
    else:
        from model_improvement.engine import MicroTEDAClusEngine as Engine
    return partial(Engine, **variant.options), make_space(
        {**BASE_SPACE, **variant.space}
    )
