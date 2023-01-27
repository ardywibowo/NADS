# NADS
["Neural Architecture Distribution Search for Uncertainty Awareness"](https://proceedings.mlr.press/v119/ardywibowo20a)

Randy Ardywibowo, Shahin Boluki, Xinyu Gong, Zhangyang Wang, Xiaoning Qian

## Overview

Machine learning (ML) systems often encounter Out-of-Distribution (OoD) errors when dealing with testing data coming from a distribution different from training data. It becomes important for ML systems in critical applications to accurately quantify its predictive uncertainty and screen out these anomalous inputs. However, existing OoD detection approaches are prone to errors and even sometimes assign higher likelihoods to OoD samples. Unlike standard learning tasks, there is currently no well established guiding principle for designing OoD detection architectures that can accurately quantify uncertainty. To address these problems, we first seek to identify guiding principles for designing uncertainty-aware architectures, by proposing Neural Architecture Distribution Search (NADS). NADS searches for a distribution of architectures that perform well on a given task, allowing us to identify common building blocks among all uncertainty-aware architectures. With this formulation, we are able to optimize a stochastic OoD detection objective and construct an ensemble of models to perform OoD detection. We perform multiple OoD detection experiments and observe that our NADS performs favorably, with up to 57% improvement in accuracy compared to state-of-the-art methods among 15 different testing configurations.

## Citation

Please consider citing our paper if you find the software useful for your work.

```
@InProceedings{pmlr-v119-ardywibowo20a,
  title = 	 {{NADS}: Neural Architecture Distribution Search for Uncertainty Awareness},
  author =       {Ardywibowo, Randy and Boluki, Shahin and Gong, Xinyu and Wang, Zhangyang and Qian, Xiaoning},
  booktitle = 	 {Proceedings of the 37th International Conference on Machine Learning},
  pages = 	 {356--366},
  year = 	 {2020},
  editor = 	 {III, Hal Daumé and Singh, Aarti},
  volume = 	 {119},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {13--18 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v119/ardywibowo20a/ardywibowo20a.pdf},
  url = 	 {https://proceedings.mlr.press/v119/ardywibowo20a.html},
}
```
