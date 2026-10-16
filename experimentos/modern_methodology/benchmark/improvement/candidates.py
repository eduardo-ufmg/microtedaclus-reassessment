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

from benchmark.hyperparameters import Category
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
