import optuna
import numpy as np
import collections


class TEDACloud:
    def __init__(self, m: float = 3.0):
        """
        Initializes the TEDA-Cloud algorithm.
        m: Number of standard deviations for the Chebyshev threshold.
        """
        self.m = m

        self.epsilon = 1e-8  # Small constant to avoid division by zero

        self.k = 0
        self.next_cloud_id = 1
        self.clouds = {}  # Format: id: {'s': int, 'mu': np.ndarray, 'var': float}
        self.intersections = collections.defaultdict(int)

    def _to_dict(self, x) -> np.ndarray:
        """
        Ensures the input is a 1D numpy array for mathematical operations.
        Handles dicts (like River/scikit-multiflow), lists, or existing arrays.
        """
        if isinstance(x, dict):
            return np.array(list(x.values()), dtype=float)
        return np.array(x, dtype=float).flatten()

    def fit(self, X, y=None):
        for x in X:
            self.learn_one(self._to_dict(x))
        return self

    def predict(self, X):
        return [self.predict_one(self._to_dict(x)) for x in X]

    def microcluster_count(self) -> int:
        return len(self.clouds)

    def _calculate_temporary_params(self, cloud, x_k):
        """Calculates temporary parameters as per Equations 9, 10, 11."""
        s_prime = cloud["s"] + 1

        # Equation 10
        mu_prime = ((s_prime - 1) / s_prime) * cloud["mu"] + (1 / s_prime) * x_k

        # Equation 11
        diff = np.linalg.norm(x_k - mu_prime) ** 2
        var_prime = ((s_prime - 1) / s_prime) * cloud["var"] + (
            1 / (s_prime - 1)
        ) * diff

        return s_prime, mu_prime, var_prime

    def _calculate_eccentricity(self, s_prime, mu_prime, var_prime, x_k):
        """Calculates normalized eccentricity as per Equations 7 and 8."""
        diff_sq = np.linalg.norm(mu_prime - x_k) ** 2
        # Equation 7 (with epsilon for stability)
        xi = (1 / s_prime) + (diff_sq / (s_prime * (var_prime + self.epsilon)))

        # Equation 8
        zeta = xi / 2
        return zeta

    def learn_one(self, x_k: np.ndarray):
        self.k += 1

        if self.k == 1:
            self.clouds[self.next_cloud_id] = {"s": 1, "mu": x_k.copy(), "var": 0.0}
            self.next_cloud_id += 1
            return

        if self.k == 2:
            cloud = self.clouds[1]
            s_prime = 2
            mu_prime = 0.5 * cloud["mu"] + 0.5 * x_k
            var_prime = np.linalg.norm(x_k - mu_prime) ** 2

            self.clouds[1] = {"s": s_prime, "mu": mu_prime, "var": var_prime}
            return

        # k >= 3 logic
        updated_clouds = []

        for c_id, cloud in list(self.clouds.items()):
            s_prime, mu_prime, var_prime = self._calculate_temporary_params(cloud, x_k)
            zeta = self._calculate_eccentricity(s_prime, mu_prime, var_prime, x_k)

            # Equation 12
            threshold = (self.m**2 + 1) / (2 * s_prime)

            if zeta <= threshold:
                self.clouds[c_id]["s"] = s_prime
                self.clouds[c_id]["mu"] = mu_prime
                self.clouds[c_id]["var"] = var_prime
                updated_clouds.append(c_id)

        # Update intersection matrix for concurrently updated clouds
        for i in range(len(updated_clouds)):
            for j in range(i + 1, len(updated_clouds)):
                c1, c2 = min(updated_clouds[i], updated_clouds[j]), max(
                    updated_clouds[i], updated_clouds[j]
                )
                self.intersections[(c1, c2)] += 1

        # Create new cloud if x_k does not belong to any existing cloud
        if not updated_clouds:
            self.clouds[self.next_cloud_id] = {"s": 1, "mu": x_k.copy(), "var": 0.0}
            self.next_cloud_id += 1

        self._evaluate_merges()

    def _evaluate_merges(self):
        """Evaluates and executes merges based on Equations 13, 14, 15, 16, 17."""
        merge_occurred = True
        while merge_occurred:
            merge_occurred = False
            cloud_ids = list(self.clouds.keys())

            for i in range(len(cloud_ids)):
                for j in range(i + 1, len(cloud_ids)):
                    c1_id, c2_id = min(cloud_ids[i], cloud_ids[j]), max(
                        cloud_ids[i], cloud_ids[j]
                    )
                    c1, c2 = self.clouds[c1_id], self.clouds[c2_id]
                    inter = self.intersections.get((c1_id, c2_id), 0)

                    # Equations 13 & 14
                    if inter > c1["s"] - inter or inter > c2["s"] - inter:
                        # Perform merge
                        new_id = self.next_cloud_id
                        self.next_cloud_id += 1

                        # Equation 15
                        s_l = c1["s"] + c2["s"] - inter

                        # Equation 16
                        mu_l = (c1["s"] * c1["mu"] + c2["s"] * c2["mu"]) / (
                            c1["s"] + c2["s"]
                        )

                        # Equation 17
                        denom = c1["s"] + c2["s"] - 2
                        var_l = (
                            (c1["s"] - 1) * c1["var"] + (c2["s"] - 1) * c2["var"]
                        ) / (denom if denom > 0 else 1)

                        self.clouds[new_id] = {"s": s_l, "mu": mu_l, "var": var_l}

                        # Transfer and sum intersection histories
                        for other_id in self.clouds.keys():
                            if other_id not in (c1_id, c2_id, new_id):
                                inter_c1 = self.intersections.pop(
                                    (min(c1_id, other_id), max(c1_id, other_id)), 0
                                )
                                inter_c2 = self.intersections.pop(
                                    (min(c2_id, other_id), max(c2_id, other_id)), 0
                                )
                                if inter_c1 + inter_c2 > 0:
                                    self.intersections[
                                        (min(new_id, other_id), max(new_id, other_id))
                                    ] = (inter_c1 + inter_c2)

                        # Cleanup
                        self.intersections.pop((c1_id, c2_id), None)
                        del self.clouds[c1_id]
                        del self.clouds[c2_id]

                        merge_occurred = True
                        break
                if merge_occurred:
                    break

    def predict_one(self, x_k: np.ndarray) -> int:
        if not self.clouds:
            return -1

        best_cloud_id = -1
        min_zeta = float("inf")

        for c_id, cloud in self.clouds.items():
            # For prediction, we evaluate against the current state without mutating
            s_prime = cloud["s"]
            mu_prime = cloud["mu"]
            var_prime = cloud["var"]

            if s_prime == 0:
                continue

            zeta = self._calculate_eccentricity(s_prime, mu_prime, var_prime, x_k)
            threshold = (self.m**2 + 1) / (2 * s_prime)

            if zeta <= threshold and zeta < min_zeta:
                min_zeta = zeta
                best_cloud_id = c_id

        return best_cloud_id


class TEDACloudParameterSearchSpace:
    def sample_parameters(self, trial: optuna.Trial) -> dict:
        return {
            "m": trial.suggest_float("m", 1.0, 5.0),
        }
