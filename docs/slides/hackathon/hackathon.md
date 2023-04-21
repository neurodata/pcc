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
# Antennal lobe as a test case

---
# Graph matching

$$\min_P \|A - P B P^T\|_F^2$$

equivalent to 

$$\min_P -trace(A^TPBP^T)$$

what is this thing?

---
# Cosine similarity 

for two vectors, $a$ and $b$...

$$cos(a,b) = \frac{<a, b>}{\|a\| \|b\|} = \frac{a^T b}{\|a\| \|b\|}$$

if we have a bunch of vectors stored in the matrices $A$ and $B$, then

$$D = A^T B$$

since $D_{ij} = \sum_k a_{ki} b_{kj}$

---
# But implicitly...

$$ C = A^T I B $$

where $I$ is the identity matrix



---

![center](./../../../results/figs/al_explore/al_accuracy.svg)

---

![center](./../../../results/figs/al_explore/conn_score.svg)

---

![center](./../../../results/figs/al_explore/nblast_between_score.svg)

---

<div class="columns">
<div>

![center h:600](./../../../results/figs/al_explore/post-hoc-metrics-nblast-between-only.svg)

</div>
<div>

![center h:600](./../../../results/figs/al_explore/post-hoc-metrics-nblast-between-connectivity.svg)

</div>
</div>

---

# Optimal transport

## Earth mover's distance

Say we have sets of objects $I = \{i : i=1,...,n\}$ and $J = \{j : j=1,...,m\}$

The "cost" of moving a unit of mass from $i$ to $j$ is given by $d_{ij}$.

We seek a set of "flows" $f_{ij}$ which minimizes the overall cost 

$$\min_F \sum_{ij} f_{ij} d_{ij} = trace(F^TD)$$

Turns out to be a sub-problem of graph matching

If $D_{ij}$ is the NBLAST cost of matching $i$ and $j$...

---

# Optimal transport on NBLAST alone

![](./../../../results/figs/al_explore/nblast-noisy-transport-compare.png)

Right is sum of 100 solutions using $D + Normal(0, 0.05)$ (iid) as the cost

---
# Using this rough assignment to induce a matching for connectivity

$$A^T F B$$

where $F$ is the transportation solution we found above, which roughly maps neurons together based on their NBLAST similarity

---

![](./../../../results/figs/al_explore/1-step-cost-comparison-v2.png)