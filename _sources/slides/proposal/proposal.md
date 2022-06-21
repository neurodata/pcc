---
marp: true
theme: slides
size: 16:9
paginate: true
---

<!-- _paginate: false -->

# On the power of comparative connectomics

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
# Table of contents

---

<!-- _class: center -->

# Motivation

---
# Connectomics as a field
- Maps are great! Provide some examples of how they have been useful for neuroscientists
- Yet, many of the claims of connectomics come from linking the connectome to other properties...

---

# Connectome $\leftrightarrow$ individuality

> Understanding statistical regularities and learning which variations are stochastic and which are secondary to an animal’s life history will help define the **substrate upon which individuality rests** and require comparisons between circuit maps within and between animals.

*Emphasis added*

<!-- _footer: Mind of a mouse, Abbott et al. 2020 -->

---

# Connectome $\leftrightarrow$ memory

> Understanding this variability likely holds a key to deciphering **how experiences are stored in the brain**, a profoundly interesting and important aspect of our own makeup.

> ...the acquisition of wiring diagrams across multiple individuals will yield insights into **how experiences shape neural connections.**

*Emphasis added*

<!-- _footer: Mind of a mouse, Abbott et al. 2020 -->

---

# Connectome $\leftrightarrow$ evolution

> Comparative connectomics of suitable, relatively small, representative **species across the phylogenetic tree** can infer the archetypal neural architecture of each bauplan and identify any circuits that possibly converged onto a shared and potentially optimal, structure.

*Emphasis added*

<!-- _footer: Neural architectures in the light of comparative connectomics, Barsotti + Correia et al. 2021-->

--- 

# Connectome $\leftrightarrow$ disease

--- 

# Connectome $\leftrightarrow$ development

---

# Comparative connectomics

> An approach to analyze neural architectures is **comparative connectomics**, consisting of first mapping the synaptic wiring diagram or connectome, of the whole or a part of a nervous system, and then analyzing its structure relative to another connectome, either of another species or of a different genotype or life stage.

<!-- _footer: Neural architectures in the light of comparative connectomics, Barsotti + Correia et al. 2021-->

---

<style scoped> 
ul{ list-style: none}
ul li:before {
    content: '';
}
</style>

# Sounds great! :smiley:

But...
- :anguished: the data still takes a long time to acquire for most systems. 
- :anguished: Connectome collection is normally not hypothesis driven, e.g. we don't really know what we're looking for in the circuit

Since these experiments take so much money and time to do, it's worth thinking about...
- :thinking: what we are trying to detect in a comparative connectomics study
- :thinking: whether we'll be able to detect it with the current sample size/statistical tools
- :thinking: whether this will lead to a mechanistic understanding (i.e. does it answer *how*)

---
# What do we ultimately want from connectomics?

## *Testable* predictions of *how* ...
- ## $X$ changes in circuit cause $Y$
- ## $U$ changes in {learning, disease, ...} cause $V$ changes in circuit

--- 

diagram of how all the variables are related

--- 

# The dream connectome experiment
1. measure connectomes 
2. compare and detect differences (predictions)
3. perturb in an experiment
4. evaluate whether mechanism in 2. is correct, if not, go back to 1.

---

<!-- _class: center -->

# Issues

---

# We don't know what we're after ahead of time

--- 

# Network statistics are typically related

---

# Causal inference is hard

---

# Acknowledgements
<style scoped> 
p {
    font-size: 26px;
}
</style>

<!-- Start people panels -->
<div class='minipanels'>

<div>

![person](./../../images/people/mike-powell.jpg)
Mike Powell

</div>

<div>

![person](./../../images/people/bridgeford.jpg)
Eric Bridgeford

</div>

<div>

![person](./../../images/people/michael_winding.png)
Michael Winding

</div>

<div>

![person](./../../images/people/marta_zlatic.jpeg)
Marta Zlatic

</div>

<div>

![person](./../../images/people/albert_cardona.jpeg)
Albert Cardona

</div>

<div>

![person](./../../images/people/priebe_carey.jpg)
Carey Priebe

</div>

<div>

![person](./../../images/people/vogelstein_joshua.jpg)
Joshua Vogelstein

</div>

<!-- End people panels -->
</div>

![bg center blur:3px opacity:15%](../../images/background.svg)
