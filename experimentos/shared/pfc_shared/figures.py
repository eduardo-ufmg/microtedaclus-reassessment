"""Colours and line styles of the thesis figures, the same in every chapter.

The model in focus of a comparison is drawn in one colour, prominently, with
its interquartile band; every other model keeps its own colour in every
chapter and is drawn as a thin median line.
"""

FOCUS_COLOR = "xkcd:red"
MODEL_COLORS = {
    "MicroTEDAClus": "xkcd:dark grey",
    "CluStream": "xkcd:purple",
    "DenStream": "xkcd:blue",
    "StreamKM": "xkcd:teal",
    "AutoCloud": "xkcd:orange",
    "CEDAS": "xkcd:pink",
    "TEDACloud": "xkcd:brown",
    "DBSTREAM": "xkcd:goldenrod",
    "BIRCH": "xkcd:magenta",
    "MiniBatchKMeans": "xkcd:cyan",
    "BICO": "xkcd:green",
}


def band(ax, x, median, q1, q3, model: str, label: str, focus: bool):
    """Median line of a model, and its interquartile band if it is the model in focus."""
    color = FOCUS_COLOR if focus else MODEL_COLORS[model]
    ax.plot(
        x,
        median,
        color=color,
        linewidth=2.2 if focus else 1.0,
        alpha=1.0 if focus else 0.75,
        zorder=3 if focus else 2,
        label=label,
    )
    if focus:
        ax.fill_between(x, q1, q3, color=color, alpha=0.25, zorder=1, linewidth=0)
