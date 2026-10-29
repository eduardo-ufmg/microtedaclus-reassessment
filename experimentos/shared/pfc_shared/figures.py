"""Colours and line styles of the thesis figures, the same in every chapter.

Every model keeps its colour, from the XKCD color survey, and its line pattern in
every figure. The model in focus of a comparison is drawn thicker, with its
interquartile band; every other model is drawn as a thin median line. The stages
between MicroTEDAClus and MicroTEDAClus+ that are not models of a comparison are
drawn in grey.
"""

MODEL_COLORS = {
    "MicroTEDAClusPlus": "xkcd:red",
    "MicroTEDAClusPlusNoNoise": "xkcd:red",
    "MicroTEDAClus": "xkcd:dusk blue",
    "DenStream": "xkcd:off green",
    "CluStream": "xkcd:blue violet",
    "StreamKM": "xkcd:teal blue",
    "AutoCloud": "xkcd:pumpkin",
    "CEDAS": "xkcd:pink purple",
    "TEDACloud": "xkcd:brown red",
    # The models of the two additional experiments of v1.0 keep their earlier colours.
    "DBSTREAM": "xkcd:goldenrod",
    "BIRCH": "xkcd:magenta",
    "MiniBatchKMeans": "xkcd:cyan",
    "BICO": "xkcd:green",
}
MODEL_STYLES = {
    "MicroTEDAClusPlus": "-",
    "MicroTEDAClusPlusNoNoise": "-",
    "MicroTEDAClus": "-",
    "DenStream": (0, (6, 2)),
    "CluStream": (0, (2, 1.5)),
    "StreamKM": (0, (6, 2, 1.5, 2)),
    "AutoCloud": (0, (1, 1.2)),
    "CEDAS": (0, (8, 2, 1.5, 2, 1.5, 2)),
    "TEDACloud": (0, (4, 1.5)),
    "DBSTREAM": "-",
    "BIRCH": "-",
    "MiniBatchKMeans": "-",
    "BICO": "-",
}

# The chain from MicroTEDAClus to MicroTEDAClus+: its ends keep the colours of the two
# models, the vectorized implementation that of MicroTEDAClus, and the stages between
# them are grey.
STAGE_COLORS = {
    "original": MODEL_COLORS["MicroTEDAClus"],
    "vectorized": MODEL_COLORS["MicroTEDAClus"],
    "corrected": "xkcd:dark grey",
    "improved": "xkcd:grey",
    "final": MODEL_COLORS["MicroTEDAClusPlus"],
}
STAGE_STYLES = {
    "original": "-",
    "vectorized": (0, (6, 2)),
    "corrected": (0, (2, 1.5)),
    "improved": (0, (6, 2, 1.5, 2)),
    "final": "-",
}


def band(ax, x, median, q1, q3, model: str, label: str, focus: bool):
    """Median line of a model, thicker and with its interquartile band if it is the model in focus."""
    color = MODEL_COLORS[model]
    ax.plot(
        x,
        median,
        color=color,
        linestyle=MODEL_STYLES[model],
        linewidth=2.4 if focus else 1.4,
        zorder=3 if focus else 2,
        label=label,
    )
    if focus:
        ax.fill_between(x, q1, q3, color=color, alpha=0.2, zorder=1, linewidth=0)
