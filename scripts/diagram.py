#%%

import numpy as np
import matplotlib.pyplot as plt
from pkg.plot import set_theme

from pkg.io import OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
import networkx as nx

from matplotlib.offsetbox import AnnotationBbox, OffsetImage

FILENAME = "diagram"

DISPLAY_FIGS = True

OUT_PATH = OUT_PATH / FILENAME


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


set_theme()
rng = np.random.default_rng(8888)


#%%


# REF: tree https://thenounproject.com/icon/biological-tree-4037951/
# REF: dna https://scidraw.io/drawing/383
# REF: adult drosophila https://scidraw.io/drawing/30
# REF: larva drosophila https://scidraw.io/drawing/405
# REF: fly behavior https://scidraw.io/drawing/150
# REF: activity https://scidraw.io/drawing/517
# REF: network https://scidraw.io/drawing/431

#%%

g = nx.DiGraph()
g.add_edge("evolution", "genetics")
g.add_edge("evolution", "connectome")
g.add_edge("genetics", "development")
g.add_edge("development", "connectome")
g.add_edge("connectome", "activity")
g.add_edge("activity", "behavior")
g.add_edge("activity", "connectome")
g.add_edge("behavior", "connectome")


nx.draw_networkx(g)

#%%


def imscatter(x, y, image, ax=None, zoom=1):
    # REF: https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    # x, y = np.atleast_1d(x, y)
    # artists = []
    # for x0, y0 in zip(x, y):
    ab = AnnotationBbox(im, (x, y), xycoords="data", frameon=False)
    ab = ax.add_artist(ab)
    ax.update_datalim(np.column_stack([x, y]))
    return ab


fig, ax = plt.subplots(1, 1, figsize=(15, 8))

positions = [
    (0.05, 0.85),
    (0.05, 0.25),
    (0.25, 0.55),
    (0.5, 0.55),
    (0.7, 0.55),
    (0.9, 0.85),
    (0.9, 0.25),
]
file_names = [
    "tree",
    "dna",
    "development",
    "network",
    "spike-trains",
    "behavior",
    "world",
]
zooms = [0.15, 0.1, 0.065, 0.1, 0.3, 0.3, 0.15]
labels = [
    "Evolution",
    "Genome",
    "Development",
    "Connectome",
    "Activity",
    "Behavior",
    "World",
]
artists = []
for i in range(len(positions)):
    pos = positions[i]
    image_path = "pcc/data/icons/" + file_names[i] + ".png"
    zoom = zooms[i]
    artist = imscatter(*pos, image_path, ax=ax, zoom=zoom)
    artists.append(artist)
fig.canvas.draw()
renderer = ax.get_figure().canvas.get_renderer()
for i, artist in enumerate(artists):
    disp = artist.get_window_extent(renderer)
    bbox = disp.transformed(ax.transAxes.inverted())
    ymin = bbox.ymin
    xcenter = (bbox.x1 + bbox.x0) / 2
    ax.text(xcenter, ymin, labels[i], va="top", ha="center", fontsize="x-large")

ax.axis("off")

pos_map = dict(zip(labels, positions))

connections = [
    ("World", "Connectome", "Activity"),
    ("Connectome", "Behavior", "Activity"),
    # (
    #     "Connectome",
    #     "Evolution",
    # ),
]

from scipy.interpolate import splrep


def draw_curve(p1, p2, p3):
    # REF: https://stackoverflow.com/questions/71960071/is-there-a-way-to-achieve-a-smooth-curve-between-two-points-for-larger-x-y-value
    f = np.poly1d(np.polyfit((p1[0], p2[0], p3[0]), (p1[1], p2[1], p3[1]), 2))
    x = np.linspace(p1[0], p2[0], 100)
    return x, f(x)


plt.autoscale(False)
# for connection in connections:

connection_positions = [(0.57, 0.55), (0.82, 0.77), (0.75, 0.6)]
# connection_positions = np.array(list(map(pos_map.get, connection)))
x, y = draw_curve(*connection_positions)
plt.plot(x, y, linewidth=6, label="Curved line", color="black", zorder=5, alpha=0.2)

# ax.plot(connection_positions[:, 0], connection_positions[:, 1], linewidth=3)

# splrep(connection_positions[:, 0], connection_positions[:, 1], k=2, s=None)
# inds = np.argsort(connection_positions[:, 0])
# connection_positions = connection_positions[inds]

# uv = UnivariateSpline(
#     x=connection_positions[:, 0],
#     y=connection_positions[:, 1],
#     k=3,
#     s=None,
# )
# for start, end in nx.utils.pairwise(connection):
#     ax.plot()

gluefig("link_connectome", fig)

# #%%
# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# image = plt.imread(image_path)
# im = OffsetImage(image)
# x0 = 0.5
# y0 = 0.5
# ab = AnnotationBbox(im, (x0, y0), xycoords="data", frameon=True)
# dir(ab)

# %%
