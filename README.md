
# PDRS: Peak-Driven Region Segmentation

PDRS is an **O(N)** algorithm designed to efficiently isolate transient high-activity regions in irregularly sampled time series data. 

Originally developed to detect flaring activity in Active Galactic Nuclei (AGN) and quasar light curves, PDRS serves as a highly scalable, linear-time alternative to dynamic programming methods like Bayesian Blocks. By seeding candidate regions at statistically significant local maxima and expanding them via a gradient-aware multi-source breadth-first search, PDRS extracts bursty events while aggressively suppressing background stochastic noise.

While built for astronomy, its domain-agnostic architecture makes it applicable to any discipline analyzing stochastic, bursty signals.

## Features
* **Linear Time Complexity:** Segment massive datasets in strict $\mathcal{O}(N)$ time.
* **Gradient-Aware Expansion:** Pushes through localized noise and plateaus using a pre-computed smoothed gradient.
* **Saddle-Point Merging:** Prevents over-segmentation of single physical events that display minor substructure.
* **Interpretable Parameters:** Tunable morphological constraints (minimum cluster size, region support, maximum temporal gaps) provide strict control over false-positive rates.
