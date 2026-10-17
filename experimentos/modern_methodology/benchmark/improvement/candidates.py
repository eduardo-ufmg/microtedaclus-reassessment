"""Candidates of the improvement experiments on the real datasets.

A candidate is an engine (scalar, diagonal or covariance), a set of engine
options and the search space of its hyperparameters, each of them a category of
the widened rules, a fixed range, or a range derived from those of the
categories. The registry grows round by round: each round compares its
candidates with a control, and a candidate that is kept becomes the control of
the next round, so the entries read as the sequence of what was tried.

The first control is the final model of the Model Improvement chapter,
MicroTEDAClus+, on the copied engine: nearest membership, single assignment,
noise label, variance limit relative to the stream, no dimensionality penalty.
"""

from dataclasses import dataclass, field, replace

from benchmark.hyperparameters import Category, Range
from benchmark.improvement.engine import MicroTEDAClusEngine
from benchmark.improvement.engine_cov import MicroTEDAClusCovarianceEngine
from benchmark.improvement.engine_diag import MicroTEDAClusDiagonalEngine

ENGINES = {
    "scalar": MicroTEDAClusEngine,
    "diagonal": MicroTEDAClusDiagonalEngine,
    "covariance": MicroTEDAClusCovarianceEngine,
}

FINAL = {
    "membership": "nearest",
    "single_assign": True,
    "relative_limit": True,
    "dim_penalty": False,
}
FINAL_SPACE = {
    "variance_limit": Category.RELATIVE_VARIANCE,
    "decay": Category.HORIZON,
    "noise_m": Category.MULTIPLIER,
}


@dataclass(frozen=True)
class Candidate:
    name: str
    round: int
    control: str | None  # the candidate it is compared with
    summary: str
    options: dict = field(default_factory=dict)
    space: dict = field(default_factory=dict)
    engine: str = "scalar"


CANDIDATES: dict[str, Candidate] = {}


def add(name, round, control, summary, options, space, engine="scalar"):
    CANDIDATES[name] = Candidate(
        name, round, control, summary, dict(options), dict(space), engine
    )
    return CANDIDATES[name]


def model_class(name: str):
    """The model of a candidate, with the interface of the benchmark models."""
    candidate = CANDIDATES[name]
    engine = ENGINES[candidate.engine]

    class Model:
        HYPERPARAMETERS = candidate.space

        def __init__(self, **hyperparameters):
            self.core = engine(**{**candidate.options, **hyperparameters})

        def learn(self, X):
            self.core.fit(X)

        def predict(self, X):
            return self.core.predict(X)

        def summary_size(self):
            return self.core.microcluster_count()

    Model.__name__ = Model.__qualname__ = f"Candidate_{name}"
    return Model


def lower_end_tenfold(category):
    """The range of a category with its lower end one decade lower."""

    def derive(of_category):
        of = of_category[category]
        return replace(of, low=of.low / 10)

    return derive


# ----------------------------------------------------------------------------
# Round 0: the control
# ----------------------------------------------------------------------------
add("control", 0, None, "MicroTEDAClus+ on the copied engine", FINAL, FINAL_SPACE)

# ----------------------------------------------------------------------------
# Round 1: the search range of the variance limit
# ----------------------------------------------------------------------------
add(
    "low_variance",
    1,
    "control",
    "variance limit searched from a tenth of the lower end of the relative-variance range",
    FINAL,
    {
        **FINAL_SPACE,
        "variance_limit": lower_end_tenfold(Category.RELATIVE_VARIANCE),
    },
)


def threshold_round(round, control):
    """The formation of microclusters: the m function and the dimension-aware threshold.

    The five variants of m that Eduardo proposed before the Model Improvement chapter
    (commit f36ba01), and Chapter 3's chi-threshold, each on the given control.
    """
    base = CANDIDATES[control]
    k95 = Range(5.0, 5000.0, log=True, integer=False)
    variants = {
        "m_const": (
            "constant m = 3, so that q <= 9 from the third sample on",
            {"m_mode": "const", "m_up": 3.0},
            {},
        ),
        "m_scalar": (
            "constant m, tuned in [1, 5]",
            {"m_mode": "const"},
            {"m_up": Range(1.0, 5.0, log=False, integer=False)},
        ),
        "m_param": (
            "original sigmoid 3 / (1 + exp(-a (S - b))) with a and b tuned",
            {"m_mode": "sigmoid", "m_up": 3.0},
            {
                "m_a": Range(0.0007, 0.07, log=True, integer=False),
                "m_b": Range(3.0, 1000.0, log=True, integer=False),
            },
        ),
        "m_reparam": (
            "m(3) = 1, m(k95) = 0.95 up; up and k95 tuned",
            {"m_mode": "reparam"},
            {"m_up": Range(1.05, 10.0, log=False, integer=False), "m_a": k95},
        ),
        "m_reparam_up3": (
            "m(3) = 1, m(k95) = 0.95 * 3; k95 tuned",
            {"m_mode": "reparam", "m_up": 3.0},
            {"m_a": k95},
        ),
        "chi_threshold": (
            "dimension-aware threshold: q <= 1 + m sqrt(2 / d)",
            {"chi_dim": True},
            {},
        ),
    }
    for name, (summary, options, space) in variants.items():
        add(
            name,
            round,
            control,
            summary,
            {**base.options, **options},
            {**base.space, **space},
            base.engine,
        )


def geometry_round(round, control):
    """The shape of the microclusters and the density that decides their activation.

    The scalar variance assumes the same dispersion in every direction; the real
    classes are strongly anisotropic. The diagonal engine keeps one variance per
    feature, the covariance engine a full covariance (directional link, defined in
    every dimension). The densities: S / sigma^2 (volume, Eduardo's order-independent
    density) and the logarithm of samples per volume of the geometry of the engine.
    """
    base = CANDIDATES[control]
    if base.engine != "scalar":
        raise ValueError("the geometry round starts from the scalar engine")
    variants = {
        "diagonal": ("one variance per feature", "diagonal", {}),
        "covariance": (
            "full covariance, directional link",
            "covariance",
            {"link_mode": "directional"},
        ),
        "volume_density": (
            "density S / sigma^2 (order-independent)",
            "scalar",
            {"density": "volume"},
        ),
        "log_density": (
            "log of samples per volume of an isotropic microcluster",
            "scalar",
            {"density": "log_volume"},
        ),
        "diagonal_log_density": (
            "one variance per feature, log of samples per axis-aligned volume",
            "diagonal",
            {"density": "log_volume"},
        ),
        "covariance_log_density": (
            "full covariance, log of samples per volume of the covariance",
            "covariance",
            {"link_mode": "directional", "density": "log_volume"},
        ),
    }
    for name, (summary, engine, options) in variants.items():
        add(
            name,
            round,
            control,
            summary,
            {**base.options, **options},
            base.space,
            engine,
        )


def remaining_round(round, control):
    """Chapter 3's candidates outside its final model, and two suggested by the diagnosis.

    Of the diagnosis: the noise label is off (the real classes have no noise), and every
    mature microcluster is active (the activation discards most of the few mature ones).
    """
    base = CANDIDATES[control]
    without_noise = {k: v for k, v in base.space.items() if k != "noise_m"}
    variants = {
        "recency": (
            "every microcluster not updated by a sample decays",
            {"life_mode": "recency"},
            base.space,
        ),
        "link_factor": (
            "intersection dist <= l (sigma_i + sigma_j), l tuned in [0.5, 4]",
            {},
            {**base.space, "link": Range(0.5, 4.0, log=False, integer=False)},
        ),
        "norm_typicality": (
            "membership by normalized typicality",
            {"membership": "norm_typicality"},
            base.space,
        ),
        "local_weights": (
            "membership weighted by the density at the sample",
            {"membership": "local_weights"},
            base.space,
        ),
        "mature_mean": (
            "activation mean over mature microclusters only",
            {"activation_scope": "mature"},
            base.space,
        ),
        "no_noise": ("no noise label", {"noise_m": 0.0}, without_noise),
        "no_activation": (
            "every mature microcluster is active",
            {"activation": "none"},
            base.space,
        ),
    }
    if base.engine == "scalar":
        variants["dim_penalty"] = (
            "dimensionality penalty 2/d restored",
            {"dim_penalty": True},
            base.space,
        )
    for name, (summary, options, space) in variants.items():
        add(
            name,
            round,
            control,
            summary,
            {**base.options, **options},
            space,
            base.engine,
        )


def combine(name, round, control, change, summary=""):
    """The change that `change` made to its own control, applied to another control."""
    base, candidate = CANDIDATES[control], CANDIDATES[change]
    origin = CANDIDATES[candidate.control]
    options = {
        k: v for k, v in candidate.options.items() if origin.options.get(k, None) != v
    }
    space = {
        k: v for k, v in candidate.space.items() if origin.space.get(k, None) is not v
    }
    removed = [k for k in origin.space if k not in candidate.space]
    merged_space = {
        k: v for k, v in {**base.space, **space}.items() if k not in removed
    }
    if candidate.engine != origin.engine:
        raise ValueError("combine applies changes of options and space only")
    add(
        name,
        round,
        control,
        summary or f"{control} with the change of {change}",
        {**base.options, **options},
        merged_space,
        base.engine,
    )


# ----------------------------------------------------------------------------
# Round 2: the formation of microclusters, on low_variance (kept in round 1)
# ----------------------------------------------------------------------------
threshold_round(2, "low_variance")


# ----------------------------------------------------------------------------
# Round 3: geometry and density, on low_variance (round 2 kept nothing)
# ----------------------------------------------------------------------------
geometry_round(3, "low_variance")


# ----------------------------------------------------------------------------
# Round 4: the remaining candidates, on low_variance (round 3 kept nothing)
# ----------------------------------------------------------------------------
remaining_round(4, "low_variance")


# ----------------------------------------------------------------------------
# Round 5: on no_noise (kept in round 4)
# ----------------------------------------------------------------------------
combine(
    "no_noise_link",
    5,
    "no_noise",
    "link_factor",
    "no noise label, intersection factor l tuned in [0.5, 4] (link_factor, kept provisionally)",
)
add(
    "no_noise_link_wide",
    5,
    "no_noise",
    "no noise label, intersection factor l tuned in [0.05, 4] on a log scale; the 2/d penalty "
    "acted as l of about 4/d, which helped INS-I and lies below the range of link_factor",
    CANDIDATES["no_noise"].options,
    {**CANDIDATES["no_noise"].space, "link": Range(0.05, 4.0, log=True, integer=False)},
)
