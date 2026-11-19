# Micromagnetic GNN

Code accompanying the paper

> H. Moustafa, S. Schaffer, C. O. W. Trost, A. Kovacs, L. Exl, H. Oezelt,
> *Graph Neural Networks for Explainable and Uncertainty-Aware Design of
> Permanent-Magnet Microstructures*, journal TBD, 2026.

An edge-centric graph neural network with global state updates is trained as a
surrogate for micromagnetic simulation of Nd₂Fe₁₄B polycrystalline
microstructures. It predicts the coercive field, the maximum energy product
(BH)max and the per-grain nucleation field from grain-graph representations, with
Monte Carlo dropout and deep-ensemble uncertainty quantification and
gradient-based explanations.

| | |
|---|---|
| Paper | journal TBD; the DOI will be added once the paper is published |
| Code (this repository) | GitHub release, archived on Zenodo: [10.5281/zenodo.24367671](https://doi.org/10.5281/zenodo.24367671) (all versions) |
| Data and figures | Zenodo: [10.5281/zenodo.22703510](https://doi.org/10.5281/zenodo.22703510) |

The trained checkpoints used in the paper are part of this repository (`results/`,
about 7 MB). The training data (1612 simulated microstructures), the raw
simulation output and the figures as published are in the data record, because
they are large and carry a different licence (CC BY 4.0).

## Installation

```bash
micromamba env create -f environment.yml    # or: conda env create -f environment.yml
micromamba activate micromagnetic_gnn
pip install -e ".[dev]"
```

`environment.yml` pins the versions that produced the paper (Python 3.11,
PyTorch 2.5.1 with CUDA 12.4, PyTorch Geometric 2.6.1). On a machine without a GPU,
replace the `pytorch-cuda=12.4` line by `pytorch-mutex=*=cpu`.

Scripts are run from the repository root and import the installed package.

## Data

The scripts read `data_cache.pkl`, the parsed grain structures with their
simulation labels, from the working directory. It is not in this repository;
download it from the data record and check it:

```bash
curl -L -o data_cache.pkl "https://zenodo.org/records/22703510/files/data_cache.pkl?download=1"
curl -L -o data_cache.pkl.sha256 "https://zenodo.org/records/22703510/files/data_cache.pkl.sha256?download=1"
sha256sum -c data_cache.pkl.sha256
```

`data_cache.pkl` is a Python pickle of a list of `micromagnetic_gnn.grain_structure.GrainStructure`
objects, so loading it requires this package to be installed. Only unpickle a file
whose checksum matches. The data record describes every field, its units and the
normalisation.

The raw simulation output (`ROM_Simulations.zip`, also in the data record) is only
needed to rebuild the cache (`default_datasets(..., rebuild=True)`) and for
`analysis/plot_hys.py`.

## Reproducing the figures of the paper

```bash
bash reproduce_figures.sh           # everything, figures go to figures/
QUICK=1 bash reproduce_figures.sh   # without the slow easy-axis optimisation in the saliency figures
```

The script needs `data_cache.pkl` in the repository root and runs these scripts:

| Figure (file in `figures/`) | Script |
|---|---|
| `bhmax_prediction`, `bhmax_gen` | `analysis/edge_model_figures.py --target bhmax` (MC dropout mean, 20 passes) |
| `coercivity_prediction`, `coercivity_gen` | `analysis/edge_model_figures.py --target coercivity --member 15` |
| `bhmax_analytic` | `analysis/bhmax_analytic.py` |
| `parity_comparison` | `analysis/parity_comparison_coercivity.py` |
| `calibration_test`, `calibration_ext` | `analysis/calibration_coercivity.py` |
| `calibration_ens_test`, `calibration_ens_ext`, `uncertainty_ens_ext` | `analysis/ensemble_coercivity.py` |
| `uncertainty_ext` | `analysis/uncertainty_decomposition_coercivity.py` |
| `switching_parity` | `analysis/parity_comparison_switching.py` |
| `feature_corr_node_spearman`, `feature_corr_edge_spearman`, `feature_corr_graph_spearman` | `analysis/feature_correlation.py` |
| `bhmax_saliency_score`, `bhmax_saliency_score_opt` | `analysis/saliency_figures.py --target bhmax` |
| `coercivity_saliency_score`, `coercivity_saliency_score_opt` | `analysis/saliency_figures.py --target coercivity --member 15` |
| `switching_saliency_score` | `analysis/saliency_figures.py --target switching` |
| `bhmax_saliency_map`, `bhmax_saliency_map_opt` | `bhmax_prediction/feature_importance.ipynb` (needs a CUDA GPU and `pip install jupyterlab ipympl`, which `environment.yml` does not include; writes next to the notebook) |
| `graph_fig_a_tessellation`, `graph_fig_b_exploded_graph`, `graph_fig_c_graph_only`, `graph_fig_all` | static illustrations, in the data record |
| `parity_comparison_paper` | not reproducible from this release: needs macro-grain simulations that are not included |

The outputs stored inside `bhmax_prediction/feature_importance.ipynb` come from an
earlier run and do not match the published saliency maps; re-running the notebook
writes new versions of the two figures next to it.

Some numbers of the paper are printed rather than plotted:
`analysis/bhmax_analytic_vs_gnn.py` (analytical formula against the GNN),
`analysis/intra_spearman_switching.py` (nucleation-field Spearman, pooled and per
structure) and `analysis/switching_field_drivers.py`.

`analysis/` also holds further scripts (`infer_coercivity.py`,
`coercivity_baseline.py`, `node_level_analysis.py`, `shape_features.py`,
`graph_structure.py`, `nucleation_site.py`, `avalanche_percolation.py`,
`bhmax_loops.py`, `plot_hys.py`, `plot_volume_vs_edge_length.py`) that are not
needed for the figures above.

**Randomness.** The data split is fixed (`SPLIT_SEED = 42`). The Monte Carlo
dropout passes are not seeded, so regenerated figures and metrics agree with the
paper up to Monte Carlo noise, not digit for digit.

## Evaluation protocol

* **Generalization set.** Structures with more than 1500 grains or a domain edge
  longer than 30 µm are withheld from training (88 structures; 57 of them have
  more than 1500 grains). The other 1524 structures are split 75 / 10 / 15 %
  into 1143 training, 152 validation and 229 test structures. The definition is
  `extrapolation_cond` and `SPLIT_SEED` in `src/micromagnetic_gnn/dataset.py`;
  every script uses it.
* **Uncertainty.** The coercivity model has a second output head for the
  aleatoric variance (Gaussian negative log-likelihood); the epistemic part comes
  from Monte Carlo dropout or from the deep ensemble. The number of dropout
  passes is a per-script default, changeable with `--mc-samples`: 20 in
  `edge_model_figures.py` and `calibration_coercivity.py`, 30 in
  `parity_comparison_coercivity.py` and `uncertainty_decomposition_coercivity.py`,
  and a single deterministic pass per member in `ensemble_coercivity.py`.
* **Scaling.** Each checkpoint stores its fitted scaler and the exact feature
  lists it was trained on, so evaluation needs no hyper-parameters.

## Trained models

All checkpoints are in `results/`. Each file is self-describing: encoder and model
configuration, fitted scaler, feature lists and weights.

| file | model | trained by |
|---|---|---|
| `results/coercivity.tar` | coercive field, with Gaussian-NLL variance head | `coercivity_prediction/main.py` |
| `results/coercivity_m15.tar` … `m19.tar` | the five deep-ensemble members of the paper; member 15 is the MC-dropout model | `coercivity_prediction/main.py --member K` |
| `results/bhmax.tar` | maximum energy product | `bhmax_prediction/main.py` |
| `results/switching_field.tar` | per-grain nucleation field | `switching_field_prediction/main.py` |

```python
from micromagnetic_gnn.checkpoints import load_coercivity_model, load_coercivity_ensemble

trained = load_coercivity_model()             # or load_bhmax_model(), load_switching_model()
train, val, test, gen = trained.datasets()    # its own features and fitted scaler
loader = trained.loader(test)

ensemble = load_coercivity_ensemble(members=[15, 16, 17, 18, 19])
```

The training scripts write their checkpoint into `results/`, which would overwrite
the shipped ones. Set `RESULTS_DIR=/some/other/dir` first when retraining.

## Repository layout

```
src/micromagnetic_gnn/       the package: geometry, data pipeline, models, training, inference, explanations
analysis/                    evaluation and figure scripts (see above)
coercivity_prediction/       training pipeline: coercive field
bhmax_prediction/            training pipeline: maximum energy product
switching_field_prediction/  training pipeline: per-grain nucleation field
results/                     trained checkpoints
tests/                       pytest suite
ROM_Simulations_test/        one small simulated structure used by the tests (a sample of the data record, CC BY 4.0)
docs/                        Sphinx documentation
reproduce_figures.sh         regenerates the figures of the paper
environment.yml              environment that produced the results
```

## Tests

```bash
pytest
```

## Citation

Please cite the paper and the archived release; see `CITATION.cff`.

## License

Code: MIT (`LICENSE`). The data record is licensed separately (CC BY 4.0).

Funded by the Austrian Science Fund (FWF), grants 10.55776/PAT7615923 and
10.55776/P35413.
