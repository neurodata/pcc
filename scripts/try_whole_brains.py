# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import seaborn as sns

from pkg.data import DATA_PATH

data_dir = DATA_PATH / "hackathon"


# %%

currtime = time.time()
nblast = pd.read_feather(
    data_dir / "nblast" / "nblast" / "nblast_flywire_mcns_comp.feather",
)
nblast = nblast.astype(float, copy=False)
print(
    f"{time.time() - currtime:.3f} seconds elapsed to load FlyWire vs. MaleCNS NBLASTs."
)
print()

print(nblast.head())
print()

# %%

n_left = nblast.shape[0]
n_right = nblast.shape[1]

a = np.ones(n_left)
b = np.ones(n_right) / n_right * n_left
S = nblast.values

currtime = time.time()
sinkhorn_sol, log = ot.sinkhorn(
    a, b, -S, reg=0.03, numItermax=200, verbose=True, log=True
)
print(f"{time.time() - currtime:.3f} seconds elapsed to sinkhorn.")
print()

currtime = time.time()
threshold = 1e-3
sinkhorn_sol[sinkhorn_sol < threshold] = 0
print(f"{time.time() - currtime:.3f} seconds elapsed to threshold.")
print()

currtime = time.time()
sinkhorn_sol = pd.DataFrame(sinkhorn_sol, index=nblast.index, columns=nblast.columns)
sinkhorn_sol.to_feather(data_dir / "nblast" / "nblast" / "sinkhorn_sol.feather")
print(f"{time.time() - currtime:.3f} seconds elapsed to save.")
print()

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(y=log["err"], x=np.arange(len(log["err"])) * 10, ax=ax)
ax.set_yscale("log")
