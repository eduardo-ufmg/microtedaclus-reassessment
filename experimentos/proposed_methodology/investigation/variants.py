"""Registry of the model variants evaluated by the investigation harness.

A variant is a set of fixed TedaClus options plus the hyperparameters Optuna
tunes. Every variant tunes variance_limit and decay over the reference ranges
unless its entry overrides the search space.
"""

BASE_SPACE = {
    "variance_limit": ("float", 0.0001, 100.0, True),
    "decay": ("int", 10, 10000, True),
}

# name -> (fixed options, search-space additions/overrides)
VARIANTS = {
    "engine_ref": ({}, {}),
    # --- single corrections applied to the reference behaviour ---
    "c_prune": ({"prune_skip": False}, {}),
    "c_born": ({"born_active": False}, {}),
    "c_peak": ({"density": "peak"}, {}),
    "c_snap": ({"density": "snapshot"}, {}),
    "c_scope": ({"activation_scope": "mature"}, {}),
    "c_nodim": ({"dim_penalty": False}, {}),
    "c_mem_normtyp": ({"membership": "norm_typ"}, {}),
    "c_mem_paper": ({"membership": "paper"}, {}),
    "c_mem_dsum": ({"membership": "density_sum"}, {}),
    "c_mem_dmax": ({"membership": "density_max"}, {}),
    "c_mem_near": ({"membership": "nearest"}, {}),
    # --- combined corrections ---
    "k1": ({"membership": "nearest", "dim_penalty": False, "prune_skip": False}, {}),
    "k2": (
        {
            "membership": "nearest",
            "dim_penalty": False,
            "prune_skip": False,
            "density": "peak",
        },
        {},
    ),
    "k3": (
        {
            "membership": "nearest",
            "dim_penalty": False,
            "prune_skip": False,
            "density": "peak",
            "activation_scope": "mature",
        },
        {},
    ),
    "a1": ({"membership": "nearest", "prune_skip": False}, {}),
    "a2": ({"membership": "nearest", "prune_skip": False, "density": "peak"}, {}),
    "a3": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "activation_scope": "mature",
        },
        {},
    ),
    # --- stage 3: single improvements on top of the corrected model (a2) ---
    "s_cap": (
        {
            **{"membership": "nearest", "prune_skip": False, "density": "peak"},
            "cap_variance": True,
        },
        {},
    ),
    "s_soft": (
        {"membership": "nearest", "prune_skip": False, "density": "peak"},
        {"activation_factor": ("float", 0.0, 1.5, False)},
    ),
    "s_none": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "activation": "none",
        },
        {},
    ),
    "s_g0": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "topology": "g0",
        },
        {},
    ),
    "s_link": (
        {"membership": "nearest", "prune_skip": False, "density": "peak"},
        {"link": ("float", 0.5, 4.0, False)},
    ),
    "s_m": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "m_mode": "reparam",
        },
        {"m_up": ("float", 1.05, 10.0, False), "m_a": ("float", 5.0, 5000.0, True)},
    ),
    "s_rec": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "life_mode": "recency",
        },
        {},
    ),
    "s_reci": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "life_mode": "recency_inactive",
        },
        {},
    ),
    "s_smax": (
        {"membership": "nearest", "prune_skip": False, "density": "peak"},
        {"s_max": ("float", 10.0, 10000.0, True)},
    ),
    "s_noise": (
        {"membership": "nearest", "prune_skip": False, "density": "peak"},
        {"noise_m": ("float", 1.0, 10.0, False)},
    ),
    "s_single": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "single_assign": True,
        },
        {},
    ),
    "s_chi": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "dim_penalty": False,
            "chi_dim": True,
        },
        {},
    ),
    "s_rel": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "relative_limit": True,
        },
        {"variance_limit": ("float", 1e-6, 1.0, True)},
    ),
    # --- stage 3: combinations ---
    "c_n_rel": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "relative_limit": True,
        },
        {
            "variance_limit": ("float", 1e-6, 1.0, True),
            "noise_m": ("float", 1.0, 10.0, False),
        },
    ),
    "c_n_rel_single": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "relative_limit": True,
            "single_assign": True,
        },
        {
            "variance_limit": ("float", 1e-6, 1.0, True),
            "noise_m": ("float", 1.0, 10.0, False),
        },
    ),
    "c_n_rel_rec": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "relative_limit": True,
            "life_mode": "recency",
        },
        {
            "variance_limit": ("float", 1e-6, 1.0, True),
            "noise_m": ("float", 1.0, 10.0, False),
        },
    ),
    "c_n_rel_rec_single": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "relative_limit": True,
            "life_mode": "recency",
            "single_assign": True,
        },
        {
            "variance_limit": ("float", 1e-6, 1.0, True),
            "noise_m": ("float", 1.0, 10.0, False),
        },
    ),
    "c_n_rel_cap": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "relative_limit": True,
            "cap_variance": True,
        },
        {
            "variance_limit": ("float", 1e-6, 1.0, True),
            "noise_m": ("float", 1.0, 10.0, False),
        },
    ),
    "c_n_single": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "single_assign": True,
        },
        {"noise_m": ("float", 1.0, 10.0, False)},
    ),
    "c_n_rec": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "life_mode": "recency",
        },
        {"noise_m": ("float", 1.0, 10.0, False)},
    ),
    # --- m-function variants on the corrected model ---
    "s_mscalar": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "m_mode": "const",
        },
        {"m_up": ("float", 1.0, 5.0, False)},
    ),
    "s_mup": (
        {"membership": "nearest", "prune_skip": False, "density": "peak"},
        {"m_up": ("float", 1.05, 10.0, False)},
    ),
    # --- geometry: isotropic without the penalty vs full covariance, on the improved model (c_n_rel) ---
    "c_n_rel_nodim": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "relative_limit": True,
            "dim_penalty": False,
        },
        {
            "variance_limit": ("float", 1e-6, 1.0, True),
            "noise_m": ("float", 1.0, 10.0, False),
        },
    ),
    "v_cov_dir": (
        {
            "_cov": True,
            "membership": "nearest",
            "density": "peak",
            "relative_limit": True,
            "link_mode": "directional",
        },
        {
            "variance_limit": ("float", 1e-6, 1.0, True),
            "noise_m": ("float", 1.0, 10.0, False),
        },
    ),
    "v_cov_bhat": (
        {
            "_cov": True,
            "membership": "nearest",
            "density": "peak",
            "relative_limit": True,
            "link_mode": "bhattacharyya",
        },
        {
            "variance_limit": ("float", 1e-6, 1.0, True),
            "noise_m": ("float", 1.0, 10.0, False),
        },
    ),
    # --- further combinations on the improved model (c_n_rel) ---
    "c_n_rel_stable": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "relative_limit": True,
            "stable_labels": True,
        },
        {
            "variance_limit": ("float", 1e-6, 1.0, True),
            "noise_m": ("float", 1.0, 10.0, False),
        },
    ),
    "c_n_rel_cap_rec": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "relative_limit": True,
            "cap_variance": True,
            "life_mode": "recency",
        },
        {
            "variance_limit": ("float", 1e-6, 1.0, True),
            "noise_m": ("float", 1.0, 10.0, False),
        },
    ),
    "c_n_rel_cap_smax": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "relative_limit": True,
            "cap_variance": True,
        },
        {
            "variance_limit": ("float", 1e-6, 1.0, True),
            "noise_m": ("float", 1.0, 10.0, False),
            "s_max": ("float", 10.0, 10000.0, True),
        },
    ),
    "c_n_rel_cap_stable": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "relative_limit": True,
            "cap_variance": True,
            "stable_labels": True,
        },
        {
            "variance_limit": ("float", 1e-6, 1.0, True),
            "noise_m": ("float", 1.0, 10.0, False),
        },
    ),
    # --- final candidates ---
    "f1": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "relative_limit": True,
            "dim_penalty": False,
            "stable_labels": True,
        },
        {
            "variance_limit": ("float", 1e-6, 1.0, True),
            "noise_m": ("float", 1.0, 10.0, False),
        },
    ),
    "f2": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "relative_limit": True,
            "dim_penalty": False,
            "stable_labels": True,
            "cap_variance": True,
        },
        {
            "variance_limit": ("float", 1e-6, 1.0, True),
            "noise_m": ("float", 1.0, 10.0, False),
        },
    ),
    "f3": (
        {
            "membership": "nearest",
            "prune_skip": False,
            "density": "peak",
            "relative_limit": True,
            "dim_penalty": False,
            "stable_labels": True,
            "cap_variance": True,
        },
        {
            "variance_limit": ("float", 1e-6, 1.0, True),
            "noise_m": ("float", 1.0, 10.0, False),
            "s_max": ("float", 10.0, 10000.0, True),
        },
    ),
    "k1_wide": (
        {"membership": "nearest", "dim_penalty": False, "prune_skip": False},
        {"variance_limit": ("float", 0.0001, 10000.0, True)},
    ),
}


def make_space(spec):
    class Space:
        def sample_parameters(self, trial):
            params = {}
            for name, (kind, lo, hi, log) in spec.items():
                if kind == "float":
                    params[name] = trial.suggest_float(name, lo, hi, log=log)
                elif kind == "int":
                    params[name] = trial.suggest_int(name, lo, hi, log=log)
                else:
                    params[name] = trial.suggest_categorical(name, lo)
            return params

    return Space


def resolve(name: str):
    """Return (model_class, search_space_class, fixed_kwargs) for a variant name."""
    if name == "reference":
        from models.MicroTEDAClus.MicroTEDAClus import (
            MicroTEDAClus,
            MicroTEDAClusParameterSearchSpace,
        )

        return MicroTEDAClus, MicroTEDAClusParameterSearchSpace, {}
    if name == "vectorized":
        from models.MicroTEDAClusVectorized.MicroTEDAClusVectorized import (
            MicroTEDAClusVectorized,
            MicroTEDAClusVectorizedParameterSearchSpace,
        )

        return MicroTEDAClusVectorized, MicroTEDAClusVectorizedParameterSearchSpace, {}

    from investigation.engine import TedaClus

    fixed, extra = VARIANTS[name]
    fixed = dict(fixed)
    space = {k: v for k, v in {**BASE_SPACE, **extra}.items() if v is not None}
    if fixed.pop("_cov", False):
        from investigation.engine_cov import TedaCov

        return TedaCov, make_space(space), fixed
    return TedaClus, make_space(space), fixed
