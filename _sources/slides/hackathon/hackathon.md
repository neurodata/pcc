---
marp: true
theme: slides
size: 16:9
paginate: true
---

<!-- _paginate: false -->

# Cambridge Hackathon

<br>

## Benjamin D. Pedigo
_(he/him)_
_[NeuroData lab](https://neurodata.io/)_
_Johns Hopkins University - Biomedical Engineering_

![icon](../../images/icons/email.png) [_bpedigo@jhu.edu_](mailto:bpedigo@jhu.edu)
![icon](../../images/icons/github.png) [_@bdpedigo (Github)_](https://github.com/bdpedigo)
![icon](../../images/icons/twitter.png) [_@bpedigod (Twitter)_](https://twitter.com/bpedigod)
![icon](../../images/icons/web.png) [_bdpedigo.github.io_](https://bdpedigo.github.io/)


![bg center blur:3px opacity:15%](../../images/background.svg)

---

# Cell types

![center h:500](../../images/cell-types-1.png)

---

# Cell types and "distribution shift"

![center h:500](../../images/cell-types-2.png)

---
# Matters for both "inductive" and "de novo" cell typing

- E.g. mapping labels from one dataset onto another via a matching, or
- Creating new labels by clustering on a unified representation of datasets

---
# Antennal lobe as a test case
- Using FlyWire data
- Trying to match left/right
- ~1750 neurons per side
- Has good labels to use for evaluation

---

# NBLAST clustering suggests this could be happening

<div class="columns">
<div>

![](../../images/side-legend.png)

</div>
<div>

![](../../images/cell-type-legend.png)


</div>
</div>

![bg right h:625](./../../../results/figs/nblast_embed/no_embed_clustermap.png)

---

# NBLAST embedding

![bg right h:625](./../../../results/figs/nblast_embed/raw_side_pairplot.png)

---

# How to line things up, based on NBLAST?


Say we have sets of objects $I = \{i : i=1,...,n\}$ and $J = \{j : j=1,...,m\}$

For example, $I$ is the set of neurons on the left, $J$ those on the right.

Say $S$ is a $n \times m$ matrix such that $S_{ij}$ has the NBLAST score between neuron $i$ (left) and neuron $j$ (right)

---

# Linear assignment problem

$$\max_{P} trace(S P^T) = \max_{P} \sum_{ij} S_{ij} P_{ij} $$

- $P$ is a permutation matrix.
- $P_{ij}$ is 1 if $i \leftrightarrow j$, 0 otherwise.

## Intuition:
Maximize the total NBLAST scores of neurons which are matched, over the set of all matchings.

$P^T$ reshuffles the columns of $S$ to make the diagonal big

---

<div class="columns">
<div>

![](./../../../results/figs/transport/nblast.png)

</div>
<div>

![](./../../../results/figs/transport/lap_solution.png)

</div>
</div>

---

# Issue with a "hard" matching?

![center h:500](../../images/cell-types-3.png)

---

# Issue with a "hard" matching?

![center h:500](../../images/cell-types-4.png)

---

# Smoother matchings

Or, how stable are these matchings?

- Add i.i.d. Gaussian noise to NBLAST matrix, $\tilde{S}_{ij} = S_{ij} + Normal(0, 0.05)$
- Run linear assignment problem, get $P^*$
- Take the average of 100 runs of the above.

---

<div class="columns">
<div>

![](./../../../results/figs/transport/nblast.png)

</div>
<div>

![](./../../../results/figs/transport/noisy_emd_solution.png)

</div>
</div>

---

# Regularized optimal transport

$$\max_{D} trace(S D^T) + \lambda \Omega(D)$$

- $D$ is a transportation matrix, i.e. rows/columns sum to 1, and $D_{ij}$ represents the amount of "flow" or matching weight from $i$ to $j$.
- $\Omega(D)$ is a regularizer which promotes "smooth" solutions (i.e. not 0-1),
- and $\lambda$ is a weight on the regularization.


## Intuition
Maximize the NBLAST scores of soft-matched neurons, weighted by how strongly those neurons are matched, over the set of (somewhat smooth) soft matchings

---

<div class="columns">
<div>

![](./../../../results/figs/transport/nblast.png)

</div>
<div>

![](./../../../results/figs/transport/sinkhorn_solution.png)

</div>
</div>

---

# Benefit (hopefully) of a smoother matching

![center h:500](../../images/cell-types-5.png)

---

# Benefit (hopefully) of a smoother matching

![center h:500](../../images/cell-types-6.png)


<!-- ---


# Optimal transport

## Earth mover's distance


The "cost" of moving a unit of mass from $i$ to $j$ is given by $d_{ij}$.

We seek a set of "flows" $f_{ij}$ which minimizes the overall cost 

$$\min_F \sum_{ij} f_{ij} d_{ij} = trace(F^TD)$$ -->


<!-- If $D_{ij}$ is the NBLAST cost of matching $i$ and $j$... -->

---

![center h:600](./../../../results/figs/transport/composite_view_2.png)

---

![center h:600](./../../../results/figs/transport/composite_view_3.png)

---

![center h:600](./../../../results/figs/transport/composite_view_4.png)


---

# What if we want to include connectivity?

## Cosine similarity:

for two vectors, $a$ and $b$...

$$cos(a,b) = \frac{<a, b>}{\|a\| \|b\|} = \frac{a^T b}{\|a\| \|b\|}$$

if we have a bunch of vectors stored in the matrices $A$ and $B$, then

$$C = A^T B$$

since $C_{ij} = \sum_k a_{ki} b_{kj}$

---
# Graph matching

$$ C = A^T I B $$

where $I$ is the identity matrix - this represents a belief about the permutation of rows of $B$ with respect to $A$... so more generally, could write:

$$C = A^T P B$$

Much like before, if we want to measure the "matchedness"

$$\max_P trace(CP^T) = trace(A^TPBP^T)$$

Can show this is equivalent to:

$$\min_P \|A - PBP^T\|_F$$


---

# Accuracy on known labels

![center](./../../../results/figs/al_explore/al_accuracy.svg)

---

# Connectivity score (low is good)

![center](./../../../results/figs/al_explore/conn_score.svg)

---

# NBLAST score (high is good)

![center](./../../../results/figs/al_explore/nblast_between_score.svg)

---

# Can we tell when we're wrong?

<div class="columns">
<div>

![center h:500](./../../../results/figs/al_explore_redo/post-hoc-metrics-nblast-between-only.svg)

</div>
<div>

![center h:500](./../../../results/figs/al_explore_redo/post-hoc-metrics-nblast-between-connectivity.svg)

</div>
</div>

---

# Can we tell when we're wrong?

For optimal transport on NBLAST, and grouping by label on the left side:

<div class="columns">
<div>

![center h:500](./../../../results/figs/al_explore_redo/sinkhorn_prop_vs_conf.svg)

</div>
<div>

![center h:500](./../../../results/figs/al_explore_redo/sinkhorn_acc_vs_conf.svg)

</div>
</div>


---

<div class="columns">
<div>

![center](./../../../results/figs/nblast_embed/emd_side_pairplot.png)

</div>
<div>

![center](./../../../results/figs/nblast_embed/emd_class_pairplot.png)

</div>
</div>

---

![center h:600](./../../../results/figs/nblast_embed/emd_side_scatter.png)

---

![center h:600](./../../../results/figs/nblast_embed/emd_clustermap.png)

---

# TODOs

- Generalizing to more than 2 datasets at a time
- Scaling experiments
  - Optimal transport runs on central brain in ~minutes (on laptop)
  - Need to see whether graph matching can scale to that size
- Seeds/soft seeds?
  - Using pre-known matchings in the optimization

---
# Appendix 


---

# Using this rough assignment to induce a matching for connectivity

$$A^T F B$$

where $F$ is the transportation solution we found above, which roughly maps neurons together based on their NBLAST similarity

---

![](./../../../results/figs/al_explore/1-step-cost-comparison-v2.png)


---

<div class="columns">
<div>

![center](./../../../results/figs/nblast_embed/raw_side_pairplot.png)

</div>
<div>

![center](./../../../results/figs/nblast_embed/raw_class_pairplot.png)

</div>
</div>

---

![center h:600](./../../../results/figs/nblast_embed/raw_side_scatter.png)

---

![center h:600](./../../../results/figs/nblast_embed/raw_clustermap.png)
