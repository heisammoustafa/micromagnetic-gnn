# Feature Importance for Micromagnetic GNNs

```{contents} Contents
:depth: 3
```

## 1. Motivation

A trained GNN gives you a number — a coercivity, a BHmax, a switching field —
but **why** that number?  Which grains, which edges, which material properties
drove the prediction?  Answering those questions is the goal of **feature
importance analysis**.

This document explains the two methods implemented in
{mod}`micromagnetic_gnn.explain`:

1. **Mask learning** (primary) — learns real-valued masks alongside the
   frozen model by gradient descent; based on the
   GNNExplainer framework [^ying2019].
2. **Ablation** (fallback) — perturbs each feature column in turn and
   measures the change in prediction error.

---

## 2. What is a "feature" in a GNN?

A single graph in our dataset carries four kinds of input:

| Kind | Symbol | Shape | Example |
|------|--------|-------|---------|
| Node features | $\mathbf{X}$ | $N \times F_n$ | `easy_axis`, `volume`, `sphericity` … |
| Edge connectivity | $\mathbf{A}$ | $E$ (indices) | which grain touches which |
| Edge attributes | $\mathbf{E}$ | $E \times F_e$ | `face_area`, `dipole_coupling` … |
| Global features | $\mathbf{u}$ | $1 \times F_g$ | `gb_t`, `applied_field` |

where

- $N$ = number of nodes (grains) in the structure,
- $E$ = number of edges (grain-boundary contacts),
- $F_n$ = number of node-feature columns,
- $F_e$ = number of edge-attribute columns,
- $F_g$ = number of global-feature columns.

All four are inputs to the GNN.  Feature importance tells us which columns
matter most for the prediction.

---

## 3. Mask Learning (ExtendedGNNExplainer)

### 3.1 Intuition

Imagine placing a **dimmer switch** on every input channel:

- For node features, one dimmer per node-per-feature (an $N \times F_n$ grid).
- For each edge, one dimmer.
- For each edge-attribute column, one dimmer (shared across all edges).
- For each global-feature column, one dimmer.

Each dimmer is a number between $0$ and $1$ (after the sigmoid, §3.4).
We *multiply* the original value by the dimmer setting:

- **Dimmer at $1$**: the value passes through unchanged.
- **Dimmer at $0$**: the value is **zeroed out** — the GNN cannot see it.
- **Dimmer at $0.3$**: the value is scaled down to 30 % of its original
  magnitude (it is weakened but not completely removed).

This is **soft masking**: every input is scaled continuously toward zero,
never replaced with a baseline or a learned constant.  The ablation method
(§4) uses replacement, but the mask learner does not.

Initially all dimmers are near $0.5$ (small random initialisation).  We
*freeze the model* and then ask: can we turn down some dimmers while keeping
the prediction almost unchanged?  If we can, those channels were unimportant.
The dimmer values after training are the **feature importance scores**.

### 3.2 Formal objective

The original GNNExplainer [^ying2019] maximises the
mutual information between the model's prediction and a masked subgraph:

```{math}
\max_{\mathbf{M}_s,\, \mathbf{M}_f}
  \;\; I\bigl( \Phi(G),\; \Phi(G_s) \bigr)
  \qquad\text{s.t.}\quad |G_s| \le K
```

where

- $\Phi$ = the trained (frozen) GNN,
- $G$ = the original input graph (all nodes, edges, features),
- $G_s$ = the **explanation subgraph** — what remains after the masks
  $\mathbf{M}_s$ and $\mathbf{M}_f$ are applied,
- $\mathbf{M}_s$ = a structural (edge) mask,
- $\mathbf{M}_f$ = a feature mask,
- $I(\cdot, \cdot)$ = mutual information,
- $|G_s|$ = the number of edges in the explanation subgraph,
- $K$ = a **budget**: the maximum number of edges the explanation may
  contain.  $K$ enforces conciseness — we want the smallest subgraph that
  still gives the same prediction.

The constraint $|G_s| \le K$ is never enforced as a hard
inequality in our implementation.  Instead, it is *relaxed* into the
sparsity regularisation terms described below (§3.5): the size penalties
($\lambda_{\text{edge}}$, $\lambda_{\text{node}}$, etc.) push the masks
toward zero, and the hyperparameters act as a soft budget.  This is
standard practice [^ying2019] and makes the
optimisation differentiable.

Our implementation {class}`~micromagnetic_gnn.explain.ExtendedGNNExplainer`
uses the equivalent **loss minimisation** form:

```{math}
\mathcal{L} \;=\;
\underbrace{\mathcal{L}_{\text{pred}}}_{\text{prediction loss}} \;+\;
\underbrace{\lambda_{\text{edge}} \,
  \bigl\| \sigma(\mathbf{m}_{\text{edge}}) \bigr\|}_{\text{edge sparsity}} \;+\;
\underbrace{\lambda_{\text{node}} \,
  \bigl\| \sigma(\mathbf{M}_{\text{node}}) \bigr\|}_{\text{node-feature sparsity}} \;+\;
\underbrace{\lambda_{\text{ea}} \,
  \bigl\| \sigma(\mathbf{m}_{\text{ea}}) \bigr\|}_{\text{edge-attr sparsity}} \;+\;
\underbrace{\lambda_{\text{u}} \,
  \bigl\| \sigma(\mathbf{m}_{\text{u}}) \bigr\|}_{\text{global-feature sparsity}} \;+\;
\mathcal{L}_{\text{entropy}}
```

Let us unpack each term.

### 3.3 Prediction loss $\mathcal{L}_{\text{pred}}$

For our regression tasks:

```{math}
\mathcal{L}_{\text{pred}} \;=\; \frac{1}{B} \sum_{i=1}^{B}
  \bigl( \Phi(\tilde{G}_i) - y_i \bigr)^2
```

where

- $\tilde{G}_i$ = the $i$-th graph with all masks applied,
- $\Phi(\tilde{G}_i)$ = the frozen model's prediction on the masked graph,
- $y_i$ = ground-truth target,
- $B$ = batch size (1 for per-graph explanations).

This is simply the **mean squared error** between the prediction on the
masked input and the original target.  If masking an unimportant feature
doesn't change the prediction, this loss stays low.

### 3.4 Masks

#### Node-feature mask $\mathbf{M}_{\text{node}}$

Shape: $N \times F_n$.  Every node $v$ and every feature $f$ gets its own
mask parameter $M_{\text{node}}^{(v,f)}$.  The masked input is:

```{math}
\tilde{\mathbf{X}} \;=\; \mathbf{X} \odot \sigma(\mathbf{M}_{\text{node}})
```

where

- $\odot$ = element-wise multiplication,
- $\sigma(z) = 1 / (1 + e^{-z})$ = the **logistic sigmoid** function,
  squeezing the raw parameter into $(0, 1)$.

Initialisation: $\mathbf{M}_{\text{node}} \sim \mathcal{N}(0,\; 0.1^2)$
(small random values so masks start near $\sigma(0) = 0.5$).

#### Edge mask $\mathbf{m}_{\text{edge}}$

Shape: $E$ (one scalar per edge).  The masked adjacency is implemented by
intercepting the message-passing inside the GNN via
{func}`~torch_geometric.explain.algorithm.utils.set_masks`:

```{math}
\tilde{\mathbf{A}} \;:\; \text{edge } e \text{ is scaled by } \sigma(m_{\text{edge}}^{(e)})
```

Initialisation:  

```{math}
m_{\text{edge}}^{(e)} \sim \mathcal{N}\!\left(0,\;
  \sqrt{\frac{2}{2N}} \cdot \sqrt{\frac{2}{1 + 0^2}}\;\right)
```

(stddev derived from {func}`torch.nn.init.calculate_gain` for ReLU, times
Kaiming uniform bound for fan-in $2N$; this matches PyG's default).

#### Edge-attribute mask $\mathbf{m}_{\text{ea}}$

Shape: $1 \times F_e$ — a **common-attributes** mask applied identically to
all edges:

```{math}
\tilde{\mathbf{E}} \;=\; \mathbf{E} \odot \sigma(\mathbf{m}_{\text{ea}})
```

Initialisation: $\mathbf{m}_{\text{ea}} \sim \mathcal{N}(0,\; 0.1^2)$.

#### Global-feature mask $\mathbf{m}_{\text{u}}$

Shape: $1 \times F_g$ — same common-attributes pattern:

```{math}
\tilde{\mathbf{u}} \;=\; \mathbf{u} \odot \sigma(\mathbf{m}_{\text{u}})
```

Initialisation: $\mathbf{m}_{\text{u}} \sim \mathcal{N}(0,\; 0.1^2)$.

### 3.5 Sparsity regularisation

Without constraints, the optimisation would leave all masks at $\approx 1$
(the prediction is easiest when nothing is masked).  Sparsity terms push
masks toward zero.

For a mask $\mathbf{m}$ (or $\mathbf{M}$) with sigmoid $\mathbf{s} = \sigma(\mathbf{m})$,
the size penalty is:

```{math}
R_{\text{size}} \;=\; \lambda \cdot \operatorname{reduce}(\mathbf{s})
```

where $\operatorname{reduce}$ is either `mean` or `sum`:

| `reduction` | Formula | When to use |
|-------------|---------|-------------|
| `mean` | $\frac{1}{K}\sum_{k=1}^{K} s_k$ | Graph-size normalised; default for all masks |
| `sum` | $\sum_{k=1}^{K} s_k$ | Scales with graph size; was used for edges in original impl. but caused bias on large graphs (see §5) |

The entropy bonus further encourages masks to be **binary** (close to 0 or 1):

```{math}
R_{\text{ent}} \;=\; \lambda_{\text{ent}} \cdot \frac{1}{K}\sum_{k=1}^{K}
\Bigl[ -s_k \log(s_k + \varepsilon) - (1 - s_k) \log(1 - s_k + \varepsilon) \Bigr]
```

where $\varepsilon = 10^{-15}$ prevents $\log(0)$.  This is the mean
**binary cross-entropy** of each mask entry.  It is minimised when
$s_k \in \{0, 1\}$.

### 3.6 Default coefficients

```{list-table} Regularisation coefficients (defaults)
:header-rows: 1
:name: tab-default-coeffs

* - Term
  - Symbol
  - Default $\lambda$
  - Reduction
  - Entropy $\lambda_{\text{ent}}$
* - Edge (connectivity)
  - $\lambda_{\text{edge}}$
  - `1.0`
  - `mean`
  - `1.0`
* - Node features
  - $\lambda_{\text{node}}$
  - `1.0`
  - `mean`
  - `0.1`
* - Edge attributes
  - $\lambda_{\text{ea}}$
  - `1.0`
  - `mean`
  - `0.1`
* - Global features
  - $\lambda_{\text{u}}$
  - `1.0`
  - `mean`
  - `0.1`
```

```{note}
The entropy coefficient for edges is deliberately higher (`1.0` vs `0.1`)
because the edge mask has $E$ independent parameters (one per edge) and we
want a crisper selection of important edges.  Node, edge-attribute and
global-feature masks use softer entropy regularisation.
```

### 3.7 Optimisation loop

1. **Initialise** all mask parameters with small random noise.
2. **Set masks** on the model via PyG's `set_masks` (intercepts message
   passing to scale edges by $\sigma(\mathbf{m}_{\text{edge}})$).
3. **Forward pass** with masked inputs.
4. **Compute** $\mathcal{L} = \mathcal{L}_{\text{pred}} + R_{\text{size}} + R_{\text{ent}}$
   (all terms combined).
5. **Back-propagate** through the *masks only* — the model weights are frozen.
6. **Update** mask parameters with Adam [^kingma2017],
   learning rate `lr = 0.01`.
7. **After epoch 0**, compute a one-shot hard mask via gradient thresholding
   ({meth}`~micromagnetic_gnn.explain.ExtendedGNNExplainer._collect_grad_masks`):
   any mask element whose gradient was non-zero in the first step is
   considered "active" and allowed to remain learnable.  This step
   prunes irrelevant edges/nodes early.
8. Repeat for `epochs = 200` steps.

### 3.8 From masks to importance scores

After training, {func}`~micromagnetic_gnn.explain.rank_importance` converts
the raw masks into a {class}`pandas.DataFrame`:

- **Node features**: $\mathbf{s}_{\text{node}} = \sigma(\mathbf{M}_{\text{node}})$,
  then `mean(dim=0)` (average over all nodes) → one score per feature.
- **Edge mask**: $\mathrm{mean}(\sigma(\mathbf{m}_{\text{edge}}))$ → single
  score labelled `"edge (connectivity)"`.
- **Edge attribute mask**: $\sigma(\mathbf{m}_{\text{ea}})$ → one score per
  edge-feature column.
- **Global mask**: $\sigma(\mathbf{m}_{\text{u}})$ → one score per global-feature
  column.

All scores lie in $(0, 1)$ and can be compared directly.

---

## 4. Ablation (fallback)

{class}`~micromagnetic_gnn.explain.feature_ablation` implements a
perturbation-based alternative that does *not* require mask training.

### 4.1 Procedure

1. Compute the **baseline MSE** on the dataset using the original,
   unperturbed inputs.
2. For each feature column $j$ (in $\mathbf{X}$, $\mathbf{E}$, or $\mathbf{u}$):
   a. Replace column $j$ with the dataset-wide mean of that column.
   b. Compute the MSE on the perturbed dataset.
   c. **Importance** $=$ MSE$_{\text{perturbed}} -$ MSE$_{\text{baseline}}$.
3. Return columns ranked by importance (largest MSE increase = most important).

### 4.2 When to use

Ablation is a simple, deterministic baseline.  Use it when:

- You want a quick sanity check on the mask-learning results.
- The mask-learning optimisation does not converge well.
- You need a model-agnostic method (works with any model, not just GNNs
  that support `set_masks`).

Limitation: ablation perturbs one column at a time and cannot capture
feature interactions.

---

## 5. A note on scaling

Earlier versions of the code used `edge_reduction="sum"` (which summed the
edge mask across all $E$ edges) while node, edge-attribute and global
masks used `mean`.  This made the edge sparsity penalty proportional to
graph size.  For small structures ($E \approx 10$), edge masks could be
relatively unconstrained; for large structures ($E > 5000$), the penalty
was huge and forced all edge masks to near zero, making them appear
unimportant.

**Fix**: all masks now use `reduction="mean"` uniformly, and all size
coefficients are `1.0`.  This puts every feature type on the same
scale regardless of graph size.

---

## 6. References

[^ying2019]: Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik,
    Jure Leskovec. *GNNExplainer: Generating Explanations for Graph Neural
    Networks*. NeurIPS 2019.
    [https://arxiv.org/abs/1903.03894](https://arxiv.org/abs/1903.03894)

[^kingma2017]: Diederik P. Kingma, Jimmy Ba. *Adam: A Method for
    Stochastic Optimization*. ICLR 2015.
    [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

### Further reading

- Mauparna Nandan, Soumita Mitra, Debashis De. *GraphXAI: a survey of
  graph neural networks (GNNs) for explainable AI (XAI)*. Neural Computing
  and Applications 37, 10949–11000 (2025).
  [https://doi.org/10.1007/s00521-025-11054-3](https://doi.org/10.1007/s00521-025-11054-3)

- Hao Yuan, Haiyang Yu, Shurui Gui, Shuiwang Ji. *Explainability in Graph
  Neural Networks: A Taxonomic Survey*. IEEE Transactions on Pattern
  Analysis and Machine Intelligence 45(5), 5782–5799 (2023).
  [https://doi.org/10.1109/TPAMI.2022.3204236](https://doi.org/10.1109/TPAMI.2022.3204236)

- Kaitlyn Ma, et al. *A Survey on Explainability of Graph Neural Networks*.
  arXiv:2306.01958 (2023).
  [https://arxiv.org/abs/2306.01958](https://arxiv.org/abs/2306.01958)

## Appendix: Notation reference

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $N$ | Number of nodes (grains) | scalar |
| $E$ | Number of edges (grain boundaries) | scalar |
| $F_n$ | Node-feature dimension | scalar |
| $F_e$ | Edge-attribute dimension | scalar |
| $F_g$ | Global-feature dimension | scalar |
| $\mathbf{X}$ | Node-feature matrix | $N \times F_n$ |
| $\mathbf{A}$ | Adjacency (edge indices) | $2 \times E$ |
| $\mathbf{E}$ | Edge-attribute matrix | $E \times F_e$ |
| $\mathbf{u}$ | Global-feature vector | $1 \times F_g$ |
| $\Phi$ | Trained GNN model | — |
| $G$ | Original input graph | — |
| $G_s$ | Explanation subgraph (after masking) | — |
| $K$ | Explanation budget (max edges in $G_s$) | scalar |
| $I$ | Mutual information | scalar |
| $\odot$ | Element-wise multiplication | — |
| $\sigma(z)$ | Logistic sigmoid $1/(1+e^{-z})$ | — |
| $\mathbf{M}_{\text{node}}$ | Node-feature mask (learnable) | $N \times F_n$ |
| $\mathbf{m}_{\text{edge}}$ | Edge mask (learnable) | $E$ |
| $\mathbf{m}_{\text{ea}}$ | Edge-attribute mask (learnable) | $1 \times F_e$ |
| $\mathbf{m}_{\text{u}}$ | Global-feature mask (learnable) | $1 \times F_g$ |
| $\tilde{\mathbf{X}}$ | Masked node features | $N \times F_n$ |
| $\tilde{\mathbf{E}}$ | Masked edge attributes | $E \times F_e$ |
| $\tilde{\mathbf{u}}$ | Masked global features | $1 \times F_g$ |
| $\mathcal{L}_{\text{pred}}$ | Prediction loss (MSE) | scalar |
| $R_{\text{size}}$ | Size regularisation | scalar |
| $R_{\text{ent}}$ | Entropy regularisation | scalar |
| $\lambda$ | Regularisation coefficient | scalar |
| $\varepsilon$ | Small constant ($10^{-15}$) for numerical stability | scalar |
