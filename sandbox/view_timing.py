#%%
import pstats

p = pstats.Stats("timing2.file")
p.sort_stats("cumulative").print_stats(50)

# %%
