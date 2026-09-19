"""
Illustrates why a single (mean, std) summary is not descriptive enough
for a score that is expected to evolve over a stream, by constructing
three synthetic score curves with similar overall (mean, std) but
qualitatively very different temporal behavior:

    1. smooth   : monotonic smoothstep growth, 0 -> 1
    2. spike    : very fast rise to 1, immediately followed by decay
                  back to 0 (mimics prequential evaluation: a model
                  cannot start at peak quality, and its quality may
                  degrade as the stream progresses)
    3. plateau  : fast rise to a plateau just above the mean of the
                  means of (1) and (2), with noise added to raise its
                  dispersion

Sequence (1) is generated first and used as the statistical reference
(mean1, std1). The free shape parameters of (2) and (3) are then solved
for via 1-D bracketed root finding (scipy.optimize.brentq) so that:
    - their means match mean1 closely  (tight tolerance, MEAN_TOL)
    - their stds approach std1 loosely (loose tolerance, STD_TOL)
Exactly matching a single (mean, std) pair across three visually
distinct curves is the point of the demonstration; the fit is validated
numerically below rather than assumed.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, fsolve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
PLOT_DIR = os.path.join(
    REPOSITORY_ROOT, "monografia", "MethodologyReassessment", "figures"
)
TABLE_DIR = os.path.join(
    REPOSITORY_ROOT,
    "monografia",
    "MethodologyReassessment",
    "artifacts",
    "standalone",
)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

N_BATCHES = 100
RNG_SEED = 0

MEAN_TOL = 0.05  # tight: means must match the reference closely
STD_TOL = 0.1  # loose: dispersions only need to be roughly similar


def smoothstep(n):
    """Sequence 1: smooth monotonic growth 0 -> 1 (Hermite smoothstep)."""
    x = np.linspace(0.0, 1.0, n)
    return 3 * x**2 - 2 * x**3


def spike(n, peak):
    """Sequence 2 shape: very fast rise to 1 at batch `peak`, then
    exponential decay back towards 0. y(0) = 0, y(peak) = 1 by
    construction (classic single-parameter impulse-response pulse)."""
    t = np.arange(n, dtype=float)
    return (t / peak) * np.exp(1.0 - t / peak)


def fit_spike_mean(n, target_mean):
    """Solve for `peak` so that mean(spike(n, peak)) == target_mean.
    mean(spike) is monotonically increasing in `peak` over the fixed
    window [0, n), so a bracketed root find is well-posed."""

    def f(peak):
        return spike(n, peak).mean() - target_mean

    lo, hi = 0.2, n - 1
    if f(lo) > 0 or f(hi) < 0:
        raise RuntimeError(
            f"target_mean={target_mean:.4f} not bracketed by peak in [{lo}, {hi}]"
        )
    return brentq(f, lo, hi)


def plateau_base(n, level, tau):
    """Sequence 3 shape (pre-noise): exponential saturation to `level`
    with rise time-constant `tau`."""
    t = np.arange(n, dtype=float)
    return level * (1.0 - np.exp(-t / tau))


def fit_plateau_tau(n, level, target_mean):
    """Solve for `tau` so that mean(plateau_base) == target_mean. The
    mean is monotonically decreasing in tau (a slower rise spends more
    time below `level`, lowering the average)."""

    def f(tau):
        return plateau_base(n, level, tau).mean() - target_mean

    lo, hi = 1e-3, n
    if f(lo) < 0 or f(hi) > 0:
        raise RuntimeError(
            f"target_mean={target_mean:.4f} not bracketed by tau in [{lo}, {hi}] for level={level:.4f}"
        )
    return brentq(f, lo, hi)


def fit_plateau_tau_and_noise(n, level, target_mean, target_std, rng):
    """Jointly solve for (tau, sigma) so that the *post-clip* plateau
    sequence y = clip(plateau_base(n, level, tau) + sigma * z, 0, 1)
    matches (target_mean, target_std). A single fixed standard-normal
    draw `z` is used throughout so the 2-equation/2-unknown system is
    deterministic; clipping couples mean and std together (it biases
    the mean too), so tau and sigma cannot be fit independently in
    separate passes -- hence the joint solve here, seeded from the
    no-noise tau estimate."""
    z = rng.standard_normal(n)
    tau0 = fit_plateau_tau(n, level, target_mean)

    def residuals(params):
        tau, sigma = params
        tau = max(tau, 1e-3)
        sigma = max(sigma, 0.0)
        y = np.clip(plateau_base(n, level, tau) + sigma * z, 0.0, 1.0)
        return [y.mean() - target_mean, y.std() - target_std]

    (tau, sigma), info, ier, msg = fsolve(residuals, x0=[tau0, 0.3], full_output=True)
    if ier != 1:
        raise RuntimeError(f"joint (tau, sigma) fit did not converge: {msg}")
    tau, sigma = max(tau, 1e-3), max(sigma, 0.0)
    y = np.clip(plateau_base(n, level, tau) + sigma * z, 0.0, 1.0)
    return y, tau, sigma


def main():
    rng = np.random.default_rng(RNG_SEED)

    # --- Sequence 1: reference ------------------------------------------------
    y1 = smoothstep(N_BATCHES)
    mean1, std1 = y1.mean(), y1.std()

    # --- Sequence 2: fast rise, then decay back to 0 ---------------------------
    peak = fit_spike_mean(N_BATCHES, mean1)
    y2 = spike(N_BATCHES, peak)
    mean2, std2 = y2.mean(), y2.std()

    # --- Sequence 3: fast rise to a plateau just above mean(mean1, mean2), + noise
    level = 1.05 * 0.5 * (mean1 + mean2)  # "just above" the mean of means
    y3, tau, sigma3 = fit_plateau_tau_and_noise(N_BATCHES, level, mean1, std1, rng)
    mean3, std3 = y3.mean(), y3.std()

    # --- Validation --------------------------------------------------------
    results = [
        ("smooth", y1, mean1, std1),
        ("spike", y2, mean2, std2),
        ("plateau", y3, mean3, std3),
    ]

    print(f"reference (seq 1): mean={mean1:.4f}  std={std1:.4f}")
    print(f"{'seq':<10}{'mean':>10}{'std':>10}{'|d_mean|':>12}{'|d_std|':>10}  status")
    ok = True
    for name, _, m, s in results:
        d_mean, d_std = abs(m - mean1), abs(s - std1)
        pass_mean, pass_std = d_mean <= MEAN_TOL, d_std <= STD_TOL
        ok &= pass_mean and pass_std
        status = "OK" if (pass_mean and pass_std) else "FAIL"
        print(f"{name:<10}{m:>10.4f}{s:>10.4f}{d_mean:>12.4f}{d_std:>10.4f}  {status}")
    print(
        f"fitted params: spike peak={peak:.3f}, plateau tau={tau:.3f}, "
        f"plateau level={level:.4f}, noise sigma={sigma3:.4f}"
    )

    assert (
        ok
    ), "one or more sequences failed to match the reference statistics within tolerance"

    # --- Plot ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    for ax, (_, y, m, s) in zip(axes, results):
        ax.plot(np.arange(N_BATCHES), y, linewidth=1.5)
        ax.set_title(f"mean: {m:.3f}, std: {s:.3f}")
        ax.set_xlabel("Batch Index")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Score")
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, "similar_stats_different_shapes.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved to {out_path}")
    scores = {
        "Smooth": complementary_scores(y1),
        "Spike": complementary_scores(y2),
        "Plateau": complementary_scores(y3),
    }
    table = latex_table(scores)

    out_path = os.path.join(TABLE_DIR, "complementary_scores_on_the_hypotheticals.tex")
    with open(out_path, "w") as f:
        f.write(table)
    print(f"Table saved to {out_path}")


def complementary_scores(y):
    return {
        "penultimate": y[-2],
        "fifth": y[4],
        "peak_drop": y.max() - y[-1],
    }


def latex_table(scores):
    """
    Produce a LaTeX table with the complementary metrics.

    Best values:
      - higher is better:
            penultimate batch ARI
            fifth batch ARI
      - lower is better:
            peak drop
    """

    best = {
        "penultimate": max(v["penultimate"] for v in scores.values()),
        "fifth": max(v["fifth"] for v in scores.values()),
        "peak_drop": min(v["peak_drop"] for v in scores.values()),
    }

    def fmt(val, best_val):
        if np.isclose(val, best_val):
            return rf"\textbf{{{val:.3f}}}"
        return f"{val:.3f}"

    lines = [
        r"\begin{table}",
        r"    \centering",
        r"    \caption{Complementary scores for the hypothetical score curves. The bold values indicate the best performance for each metric.}",
        r"    \begin{tabular}{|c|c|c|c|}",
        r"        \hline",
        r"        Curve & Penultimate batch & Fifth batch & Highest $-$ Last \\",
        r"        \hline",
    ]

    for name, s in scores.items():
        lines.append(
            "        "
            + f"{name}"
            + " & "
            + fmt(s["penultimate"], best["penultimate"])
            + " & "
            + fmt(s["fifth"], best["fifth"])
            + " & "
            + fmt(s["peak_drop"], best["peak_drop"])
            + r" \\"
        )

    lines.extend(
        [
            r"        \hline",
            r"    \end{tabular}",
            r"    \label{tab:complementary_scores_on_the_hypotheticals}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
