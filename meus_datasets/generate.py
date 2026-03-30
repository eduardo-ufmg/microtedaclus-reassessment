import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Configuration
N_SAMPLES = 3000
NOISE_STD = 0.1


def generate_static_moons(n_samples, noise):
    X, y = make_moons(n_samples=n_samples, noise=noise)
    return X, y


def generate_nonoverlap_movingblobs(n_samples, noise):
    # Three moving blobs, parallel linear trajectories, no overlap
    t = np.random.uniform(0, 1, n_samples)
    y = np.random.randint(0, 3, n_samples)
    X = np.zeros((n_samples, 2))

    for i in range(3):
        mask = y == i
        # x(t) = -2 + 4t; y(t) = 2 - 2i
        X[mask, 0] = (-2 + 4 * t[mask] + np.random.normal(0, noise, mask.sum())) * (
            -1
        ) ** i
        X[mask, 1] = 2 - (i * 2) + np.random.normal(0, noise, mask.sum())

    return X, y, t


def generate_crossing_movingblobs(n_samples, noise):
    # Three blobs crossing in the center from a circumference
    t = np.random.uniform(0, 1, n_samples)
    y = np.random.randint(0, 3, n_samples)
    X = np.zeros((n_samples, 2))
    angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
    R = 2.0

    for i in range(3):
        mask = y == i
        mu_0 = np.array([R * np.cos(angles[i]), R * np.sin(angles[i])])
        t_masked = t[mask][:, None]
        # mu(t) = (1 - 2t) * mu_0
        X[mask] = (1 - 2 * t_masked) * mu_0 + np.random.normal(
            0, noise, (mask.sum(), 2)
        )

    return X, y, t


def generate_birthdeath(n_samples, noise):
    # Two static blobs initially, a third appears, then the first disappears
    t = np.random.uniform(0, 1, n_samples)
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples, dtype=int)
    centers = np.array([[-2, 2], [2, 2], [0, -2]])  # Classes 0, 1, 2

    for i in range(n_samples):
        ti = t[i]
        if ti < 1 / 3:
            c = np.random.choice([0, 1])
        elif ti < 2 / 3:
            c = np.random.choice([0, 1, 2])
        else:
            c = np.random.choice([1, 2])

        y[i] = c
        X[i] = centers[c] + np.random.normal(0, noise, 2)

    return X, y, t


def generate_rotating_blobs(n_samples, noise):
    # Three blobs rotating around the origin over the same circular trajectory
    t = np.random.uniform(0, 1, n_samples)
    y = np.random.randint(0, 3, n_samples)
    X = np.zeros((n_samples, 2))
    R = 2.0

    for i in range(3):
        mask = y == i
        angles = 2 * np.pi * t[mask] + (i * 2 * np.pi / 3)
        X[mask, 0] = R * np.cos(angles) + np.random.normal(0, noise, mask.sum())
        X[mask, 1] = R * np.sin(angles) + np.random.normal(0, noise, mask.sum())

    return X, y, t


def save_dataset(X, y, filename):
    df = pd.DataFrame(
        {
            "x1": X[:, 0],
            "x2": X[:, 1],
            "y": y,
        }
    )
    df.to_csv(filename, index=False)


def plot_static(X, y, title, filename):
    plt.figure(figsize=(6, 6))
    for i in np.unique(y):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1])
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def plot_dynamic(X, y, t, title, filename):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
    windows = [0.0, 0.25, 0.50, 0.75, 1.0]

    for i in range(4):
        mask = (t >= windows[i]) & (t <= windows[i + 1])
        X_w, y_w = X[mask], y[mask]

        for c in np.unique(y):
            c_mask = y_w == c
            axes[i].scatter(X_w[c_mask, 0], X_w[c_mask, 1])

        axes[i].set_title(f"$t \\in [{windows[i]}, {windows[i+1]}]$")

        # Fixed limits to ensure spatial consistency across time windows
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-3, 3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "static_moons",
            "nonoverlap_movingblobs",
            "crossing_movingblobs",
            "birthdeath",
            "rotating_blobs",
            "all",
        ],
        default="all",
        help="Which dataset to generate",
    )
    args = parser.parse_args()

    if args.dataset in ("static_moons", "all"):
        print("Generating and plotting static dataset...")
        X_stat, y_stat = generate_static_moons(N_SAMPLES, NOISE_STD)
        save_dataset(X_stat, y_stat, "static_moons.csv")
        plot_static(X_stat, y_stat, "Static Moons Dataset", "static_moons.png")

    if args.dataset in ("nonoverlap_movingblobs", "all"):
        print("Generating and plotting dynamic dataset 1 (Linear/Parallel)...")
        X_dyn1, y_dyn1, t_dyn1 = generate_nonoverlap_movingblobs(N_SAMPLES, NOISE_STD)
        save_dataset(X_dyn1, y_dyn1, "dynamic_nonoverlap_movingblobs.csv")
        plot_dynamic(
            X_dyn1,
            y_dyn1,
            t_dyn1,
            "Dynamic 1: Linear Parallel Trajectories",
            "dynamic_nonoverlap_movingblobs.png",
        )

    if args.dataset in ("crossing_movingblobs", "all"):
        print("Generating and plotting dynamic dataset 2 (Radial Crossing)...")
        X_dyn2, y_dyn2, t_dyn2 = generate_crossing_movingblobs(N_SAMPLES, NOISE_STD)
        save_dataset(X_dyn2, y_dyn2, "dynamic_crossing_movingblobs.csv")
        plot_dynamic(
            X_dyn2,
            y_dyn2,
            t_dyn2,
            "Dynamic 2: Radial Crossing",
            "dynamic_crossing_movingblobs.png",
        )

    if args.dataset in ("birthdeath", "all"):
        print("Generating and plotting dynamic dataset 3 (Appearance/Disappearance)...")
        X_dyn3, y_dyn3, t_dyn3 = generate_birthdeath(N_SAMPLES, NOISE_STD)
        save_dataset(X_dyn3, y_dyn3, "dynamic_birthdeath.csv")
        plot_dynamic(
            X_dyn3,
            y_dyn3,
            t_dyn3,
            "Dynamic 3: Concept Drift via Addition/Deletion",
            "dynamic_birthdeath.png",
        )

    if args.dataset in ("rotating_blobs", "all"):
        print("Generating and plotting rotating blobs dataset...")
        X_dyn4, y_dyn4, t_dyn4 = generate_rotating_blobs(N_SAMPLES, NOISE_STD)
        save_dataset(X_dyn4, y_dyn4, "dynamic_rotating_blobs.csv")
        plot_dynamic(
            X_dyn4,
            y_dyn4,
            t_dyn4,
            "Dynamic 4: Rotating Blobs",
            "dynamic_rotating_blobs.png",
        )


if __name__ == "__main__":
    main()
