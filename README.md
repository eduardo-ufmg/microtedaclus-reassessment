# MicroTEDAClus reassessment and improvement

LaTeX sources and experiment code of the undergraduate thesis *MicroTEDAClus Formulation,
Implementation, and Experimental Validation Reassessment and Improvement*, by Eduardo Henrique
Basilio de Carvalho, advised by Prof. Janier Arias Garcia, PhD. Bachelor's Degree in Control and
Automation Engineering, Federal University of Minas Gerais, November 2026.

The thesis reassesses the formulation, the implementation and the experimental methodology of
MicroTEDAClus, an evolving clustering algorithm for data streams, and defines an improved model,
MicroTEDAClus+, by controlled experiments.

## Versions

| Tag | Document |
|---|---|
| `thesis-defense` | The thesis as submitted for the defense. Its PDF is `monografia/UndergradThesis.pdf` at that tag. |
| `v1.0` | An extended version of the work, with two more chapters: the experiment repeated in a Python-only environment, and an evaluation on real datasets. Its PDF is attached to the [v1.0 release](https://github.com/eduardo-ufmg/microtedaclus-reassessment/releases/tag/v1.0). |

The branch `defense-edition` holds the thesis, and `main` holds `v1.0`. The other branches keep
earlier stages of the work.

## Layout

| Path | Content |
|---|---|
| `monografia/` | LaTeX sources of the thesis. Each chapter's `artifacts/` folder holds the tables and figures that the scripts below generate. |
| `experimentos/proposed_methodology/` | The methodology of the chapter *Methodology Reassessment*: streams, models, protocol, main experiment, statistics, tables and figures. |
| `experimentos/proposed_methodology/model_improvement/` | The experiments of the chapter *Model Improvement*: runner, variants, comparisons, timing, probe and their tables and figures. |
| `experimentos/preprocessing/` | The effect of the preprocessing of the original work on MicroTEDAClus. |
| `experimentos/multiple_runs/` | The choice of the number of runs. |
| `experimentos/shared/` | The package `pfc_shared`, code shared by the experiments. |
| `experimentos/modern_methodology/` | The two additional experiments of `v1.0`. |
| `experimentos/datasets/`, `experimentos/metrics/` | Figures of the datasets and of the metrics. |
| `testes_preeliminares/` | Preliminary tests, among them the measurement of the interface to the benchmark models in Java. |
| `referencia/` | Earlier work, superseded; see its README. |

## Reproducing the results

The result folders of the experiments (`results*`) are not versioned. Every table and figure of
the thesis is generated from them by a script, so regenerating one means running its experiment
first.

### Environment

- Python 3.11 or later (the experiments ran on 3.12).
- R with the packages `stream`, `streamMOA` and `mlbench`, and Java for `streamMOA`, which generate
  the synthetic streams. If `rpy2` cannot load Java, put the JVM's `lib/server` directory on
  `LD_LIBRARY_PATH`.

```bash
cd experimentos/proposed_methodology
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python verify_environment.py
```

Unless noted, the commands below run from `experimentos/proposed_methodology` with this
environment.

### Methodology Reassessment

```bash
python -m streams 0 93          # generate and cache the streams of the 94 runs (needs R)
python main_experiment.py       # every model on every dataset and run; resumes where it stopped
python check_for_completion.py
python compute_metrics.py
python statistic_analysis.py
python tables_and_plots.py
python parameter_investigation.py
```

The preprocessing study and the choice of the number of runs have their own folders:

```bash
cd experimentos/preprocessing
../proposed_methodology/.venv/bin/python run_preprocessing_experiment.py
../proposed_methodology/.venv/bin/python analyze_preprocessing_results.py
```

```bash
cd experimentos/multiple_runs
../proposed_methodology/.venv/bin/python run_multiple_runs.py
../proposed_methodology/.venv/bin/python statistical_power_investigation.py
```

### Model Improvement

The comparisons and their runs are registered in `model_improvement/experiments.py`: development
runs 94 to 125, validation runs 126 to 219. The runner refuses to start while the model code has
uncommitted changes, and `--experiment KEY` runs a single comparison.

```bash
python -m streams 94 219                          # the streams of these runs (needs R)
python -m model_improvement.harness --all         # every comparison, on its registered runs
python -m model_improvement.timing                # serial timing pass, on an otherwise idle machine
python -m model_improvement.probe SPACE DATASET   # reachability probe
python -m model_improvement.artifacts --summary   # tables, figures and decisions
```

### The thesis

```bash
cd monografia
latexmk -pdf UndergradThesis.tex
```

## Citing

```bibtex
@misc{carvalho2026microtedaclus,
  author       = {Eduardo Henrique Basilio de Carvalho},
  title        = {MicroTEDAClus Formulation, Implementation, and Experimental Validation Reassessment and Improvement},
  year         = {2026},
  howpublished = {\url{https://github.com/eduardo-ufmg/microtedaclus-reassessment}},
  note         = {Undergraduate thesis, Federal University of Minas Gerais}
}
```

## License

The code is under the MIT License (`LICENSE`). The text of the thesis and the tables and figures
generated for it, in `monografia/`, are under CC BY 4.0 (`monografia/LICENSE.md`). Material of
other authors keeps its own terms; see `THIRD_PARTY_NOTICES.md`.
