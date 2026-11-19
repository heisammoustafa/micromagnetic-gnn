#!/usr/bin/env bash
# Regenerate the figures and numbers of the paper from the shipped checkpoints
# (results/) and the data record (data_cache.pkl).
#
#   bash reproduce_figures.sh            # everything
#   QUICK=1 bash reproduce_figures.sh    # skip the slow easy-axis optimisation in the saliency figures
#
# Run it from the repository root, inside the environment of environment.yml, with
# the package installed (pip install -e .). Figures go to figures/ (or $FIGURE_DIR).
# Which script produces which figure: see the table in README.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PYTHON:-python}"

if [ ! -f data_cache.pkl ]; then
    cat >&2 <<'EOF'
data_cache.pkl not found in the repository root.
Download it from the data record (see "Data" in README.md) and verify it:
    sha256sum -c data_cache.pkl.sha256
Several scripts read data_cache.pkl from the working directory, so it (or a symlink to
it) has to sit next to this script.
EOF
    exit 1
fi

run() { echo; echo ">>> $PY $*"; "$PY" "$@"; }

SALIENCY_FLAGS=()
if [ "${QUICK:-0}" = "1" ]; then SALIENCY_FLAGS+=(--no-opt); fi

# --- parity plots and generalization error (BHmax and coercivity, MC dropout, 20 passes) --
# The coercivity results of the paper are ensemble member 15; the script's default,
# results/coercivity.tar, is a different model and gives other numbers.
run analysis/edge_model_figures.py --member 15

# --- BHmax analytical formula ------------------------------------------------------------
run analysis/bhmax_analytic.py
run analysis/bhmax_analytic_vs_gnn.py

# --- coercivity: uncertainty quantification ---------------------------------------------
run analysis/parity_comparison_coercivity.py
run analysis/calibration_coercivity.py
run analysis/ensemble_coercivity.py
run analysis/uncertainty_decomposition_coercivity.py

# --- nucleation field --------------------------------------------------------------------
run analysis/parity_comparison_switching.py
run analysis/intra_spearman_switching.py
run analysis/switching_field_drivers.py

# --- feature correlations and saliency --------------------------------------------------
run analysis/feature_correlation.py
run analysis/saliency_figures.py --target bhmax ${SALIENCY_FLAGS[@]+"${SALIENCY_FLAGS[@]}"}
# coercivity saliency of the paper: ensemble member 15 (the default, results/coercivity.tar, differs)
run analysis/saliency_figures.py --target coercivity --member 15 ${SALIENCY_FLAGS[@]+"${SALIENCY_FLAGS[@]}"}
run analysis/saliency_figures.py --target switching ${SALIENCY_FLAGS[@]+"${SALIENCY_FLAGS[@]}"}

cat <<'EOF'

Done. Not produced by this script:
  * graph_fig_a/b/c (+ graph_fig_all): static illustrations shipped in the data record.
  * bhmax_saliency_map.pdf, bhmax_saliency_map_opt.pdf: run
    bhmax_prediction/feature_importance.ipynb (needs Jupyter with ipympl and a CUDA GPU).
  * parity_comparison_paper.pdf: needs the macro-grain data, which is not part of this release.
EOF
