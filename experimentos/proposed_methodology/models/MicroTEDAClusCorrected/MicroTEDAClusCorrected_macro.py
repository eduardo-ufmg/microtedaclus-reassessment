import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import optuna
from matplotlib.patches import Circle
from numba import jit


class MicroTEDAClusCorrectedMacro:
    def __init__(
        self,
        macro_cluster_update=1,
        verbose=0,
        variance_limit=0.001,
        decay=100,
        debug=False,
        plot_graph=False,
        epsilon=1e-8,
    ):
        self.verbose = verbose
        self.total_num_samples = 0
        self.micro_clusters = []
        self.macro_clusters = []
        self.graph = nx.Graph()
        self.variance_limit = variance_limit
        self.macro_cluster_update = macro_cluster_update
        self.debug = debug
        self.plot_graph = plot_graph
        self.changed_micro_clusters = []
        self.fading_factor = 1 / decay
        self.epsilon = epsilon

    @staticmethod
    def get_micro_cluster(id, num_samples, mean, variance, density, life):
        return {
            "id": id,
            "num_samples": num_samples,
            "mean": mean,
            "variance": variance,
            "density": density,
            "changed": True,
            "life": life,
        }

    def create_new_micro_cluster(self, x):

        if len(self.micro_clusters) == 0:
            id = 1
        else:
            id = max([m["id"] for m in self.micro_clusters]) + 1

        num_samples = 1
        mean = x
        variance = 0
        density = num_samples / (variance + self.epsilon)
        life = 1.0
        new_mc = MicroTEDAClusCorrectedMacro.get_micro_cluster(
            id, num_samples, mean, variance, density, life
        )
        self.micro_clusters.append(new_mc)
        self.changed_micro_clusters.append(new_mc)
        self.graph.add_node(id)

    def is_outlier(self, s_ik, var_ik, norm_ecc):

        if s_ik < 3:
            outlier = var_ik > self.variance_limit
        else:
            mik_sik = 3 / (1 + math.exp(-0.007 * (s_ik - 100)))
            outlier_limit = ((mik_sik**2) + 1) / (2 * s_ik)
            outlier = norm_ecc > outlier_limit

        return outlier

    def update_micro_cluster(self, xk, micro_cluster, num_samples, mean, variance):
        micro_cluster["num_samples"] = num_samples
        micro_cluster["mean"] = mean
        micro_cluster["variance"] = variance
        micro_cluster["density"] = num_samples / (variance + self.epsilon)
        micro_cluster["life"] = 1.0
        micro_cluster["changed"] = True

        if micro_cluster not in self.changed_micro_clusters:
            self.changed_micro_clusters.append(micro_cluster)

    @staticmethod
    def get_updated_micro_cluster_values(x, s_ik, mu_ik, var_ik):

        s_ik += 1
        mean = ((s_ik - 1) / s_ik) * mu_ik + (x / s_ik)

        # Codigo dissertacao
        delta = x - mean
        variance = MicroTEDAClusCorrectedMacro.update_variance(delta, s_ik, var_ik)

        norm_ecc = MicroTEDAClusCorrectedMacro.get_normalized_eccentricity(
            x, s_ik, mean, variance
        )
        return (s_ik, mean, variance, norm_ecc)

    @staticmethod
    @jit(nopython=True)
    def update_variance(delta, s_ik, var_ik):
        variance = ((s_ik - 1) / s_ik) * var_ik + (
            (np.linalg.norm(delta) ** 2) / (s_ik - 1)
        )
        return variance

    @staticmethod
    def get_normalized_eccentricity(x, num_samples, mean, var):
        ecc = MicroTEDAClusCorrectedMacro.get_eccentricity(x, num_samples, mean, var)
        return ecc / 2

    @staticmethod
    @jit(nopython=True)
    def get_eccentricity(x, num_samples, mean, var):
        if var == 0 and num_samples > 1:
            result = 1 / num_samples
        else:
            a = mean - x
            result = (1 / num_samples) + (
                (np.linalg.norm(a) ** 2) / (num_samples * var)
            )

        return result

    @staticmethod
    def get_typicality(x, num_samples, mean, var):
        ecc = MicroTEDAClusCorrectedMacro.get_eccentricity(x, num_samples, mean, var)
        return 1.0 - ecc

    @staticmethod
    def get_normalized_typicality(x, num_samples, mean, var):
        tau = MicroTEDAClusCorrectedMacro.get_typicality(x, num_samples, mean, var)
        return tau / (num_samples - 2.0)

    def fit(self, X, update_macro_clusters=True, prune_micro_clusters=True):

        lenx = len(X)

        if self.debug:
            print("Training...")

        for xk in X:
            self.update_micro_clusters(xk)

            if prune_micro_clusters:
                self.prune_micro_clusters()

            self.total_num_samples += 1

            if self.debug:
                print("Training %d of %d" % (self.total_num_samples, lenx))

        if update_macro_clusters:
            if self.debug:
                print("Updating Macro_clusters")
            self.update_macro_clusters()

        if self.plot_graph:
            self.plot_micro_clusters(X)

    def update_micro_clusters(self, xk):
        # First sample
        if self.total_num_samples == 0:
            self.create_new_micro_cluster(xk)
        else:
            new_micro_cluster = True

            for mi in self.micro_clusters:
                mi["changed"] = False
                s_ik = mi["num_samples"]
                mu_ik = mi["mean"]
                var_ik = mi["variance"]

                num_samples, mean, variance, norm_ecc = (
                    MicroTEDAClusCorrectedMacro.get_updated_micro_cluster_values(
                        xk, s_ik, mu_ik, var_ik
                    )
                )

                if not self.is_outlier(num_samples, variance, norm_ecc):
                    self.update_micro_cluster(xk, mi, num_samples, mean, variance)
                    new_micro_cluster = False
                else:
                    mi["life"] -= self.fading_factor / mi["num_samples"]

            if new_micro_cluster:
                self.create_new_micro_cluster(xk)

    def update_macro_clusters(self):
        self.define_macro_clusters()

    def get_micro_clusters_by_ids(self, mg):
        mcs = []
        for mi_ind in mg:
            mi = next(
                (item for item in self.micro_clusters if item["id"] == mi_ind), None
            )
            if mi:
                mcs.append(mi)
        return mcs

    def predict(self, X):
        self.labels_ = np.zeros(len(X), dtype=int)
        index = 0
        lenx = len(X)

        if self.debug:
            print("Predicting...")

        for xk in X:
            memberships = []
            for mg in self.macro_clusters:
                mcs = self.get_micro_clusters_by_ids(mg)
                if not mcs:
                    continue

                total_density = sum(m["density"] for m in mcs)
                mean_density = total_density / len(mcs)

                active_micro_clusters = [
                    m
                    for m in mcs
                    if m["num_samples"] > 2 and m["density"] >= mean_density
                ]

                memberships.append(
                    MicroTEDAClusCorrectedMacro.calculate_membership(
                        xk, active_micro_clusters
                    )
                )

            if memberships is not None and len(memberships) > 0:
                self.labels_[index] = np.argmax(memberships)
            else:
                self.labels_[index] = -1
            index += 1

            if self.debug:
                print("Predicting %d of %d" % (index, lenx))

        return self.labels_

    @staticmethod
    def calculate_membership(x, active_micro_clusters):
        total_density = 0
        for m in active_micro_clusters:
            total_density += m["density"]

        mb = 0
        if total_density == 0:
            return mb

        for m in active_micro_clusters:
            d = m["density"]

            t = MicroTEDAClusCorrectedMacro.get_normalized_typicality(
                x, m["num_samples"], m["mean"], m["variance"]
            )
            mb += (d / total_density) * t
        return mb

    @staticmethod
    def calculate_micro_membership(x, params):

        micro_cluster = params[0]
        total_density = params[1]

        d = micro_cluster["density"]

        t = MicroTEDAClusCorrectedMacro.get_normalized_typicality(
            x,
            micro_cluster["num_samples"],
            micro_cluster["mean"],
            micro_cluster["variance"],
        )
        return (d / total_density) * t

    def get_total_density(self):
        return sum(m["density"] for m in self.micro_clusters)

    def get_changed_micro_clusters(self):
        changed_micro_clusters = []

        for m in self.micro_clusters:
            if m["changed"]:
                changed_micro_clusters.append(m)
        return changed_micro_clusters

    def define_macro_clusters(self):

        for mi in self.changed_micro_clusters:
            for mj in self.micro_clusters:
                if mi["id"] != mj["id"]:
                    edge = (mi["id"], mj["id"])
                    if MicroTEDAClusCorrectedMacro.has_intersection(mi, mj):
                        self.graph.add_edge(*edge)
                    elif MicroTEDAClusCorrectedMacro.nodes_connected(
                        mi["id"], mj["id"], self.graph
                    ):
                        self.graph.remove_edge(*edge)

        self.macro_clusters = list(nx.connected_components(self.graph))
        self.changed_micro_clusters.clear()

    @staticmethod
    def nodes_connected(u, v, G):
        return u in G.neighbors(v)

    @staticmethod
    def has_intersection(mi, mj):
        mu_i = mi["mean"]
        mu_j = mj["mean"]
        var_i = mi["variance"]
        var_j = mj["variance"]

        d = MicroTEDAClusCorrectedMacro.get_euclidean_distance(mu_i, mu_j)
        dist = np.sqrt(np.sum(d))

        deviation = MicroTEDAClusCorrectedMacro.get_deviation(var_i, var_j)

        return dist <= deviation

    @staticmethod
    @jit(nopython=True)
    def get_deviation(var_i, var_j):
        deviation = 2 * (np.sqrt(var_i) + np.sqrt(var_j))
        return deviation

    @staticmethod
    @jit(nopython=True)
    def get_euclidean_distance(mu_i, mu_j):
        dist = [(a - b) ** 2 for a, b in zip(mu_i, mu_j)]
        return dist

    def plot_micro_clusters(self, X):

        ax = plt.gca()

        ax.scatter(X[:, 0], X[:, 1], s=1, color="b")

        for m in self.micro_clusters:
            mean = m["mean"]
            std = math.sqrt(m["variance"])

            circle = Circle(mean, std, color="r", fill=False)

            ax.add_artist(circle)
        plt.draw()

    def prune_micro_clusters(self):
        for mc in self.micro_clusters[:]:
            if mc["life"] <= 0:
                self.micro_clusters.remove(mc)

                if mc in self.changed_micro_clusters:
                    self.changed_micro_clusters.remove(mc)

                if self.graph.has_node(mc["id"]):
                    self.graph.remove_node(mc["id"])

    def microcluster_count(self):
        return len(self.micro_clusters)


class MicroTEDAClusCorrectedMacroParameterSearchSpace:
    def sample_parameters(self, trial: optuna.Trial) -> dict:
        return {
            "variance_limit": trial.suggest_float(
                "variance_limit", 0.0001, 100.0, log=True
            ),
            "decay": trial.suggest_int("decay", 10, 10000, log=True),
        }
