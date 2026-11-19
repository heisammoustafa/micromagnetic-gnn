from micromagnetic_gnn.grain_structure import load_one_grain_structure, GrainStructure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_grain(grain, ax, alpha, color):
    # draw the polygons of the convex hull
    polygons = []
    for idx in grain.faces.values():
        vertices = grain.vertices[idx]
        origin = np.mean(vertices, axis=0)
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]

        n = np.cross(v1, v2)
        n /= np.linalg.norm(n)

        e1 = v1 / np.linalg.norm(v1)
        e2 = np.cross(n, e1)
        # project to 2D
        p = vertices - origin

        x = p @ e1
        y = p @ e2

        # cyclic ordering
        angles = np.arctan2(y, x)
        sort_idx = np.argsort(angles)

        polygons.append(vertices[sort_idx])

    tri = Poly3DCollection(
        polygons,
        shade=True,
        facecolors=color,
        edgecolors="k",
    )
    tri.set_alpha(alpha)
    tri.set_linewidth(0.0)
    ax.add_collection3d(tri)


def draw_grain_structure(
    gs: GrainStructure,
    alpha: np.ndarray,
    color: np.ndarray,
    alpha_min=0.01,
    alpha_max=0.6,
    cmap=cm.plasma_r,
    figsize=(10, 10),
    cbar_label: str | None = None
) -> plt.Figure:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    alpha = MinMaxScaler((alpha_min, alpha_max)).fit_transform(alpha[:, None])[:, 0]

    norm = mcolors.Normalize(vmin=min(color), vmax=max(color))
    rgba = cmap(norm(np.asarray(color)))

    for a, c, grain in zip(alpha, rgba, gs.grains.values(), strict=True):
        plot_grain(grain, ax, alpha=a, color=c)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    if cbar_label is not None:
        cbar.set_label(cbar_label)
    ax.set_axis_off()
    ax.set_axis_off()
    return fig
