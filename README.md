# Learning Fast and Slow: Towards Inclusive Federated Learning

This GitHub repository contains the code and resources required to reproduce the experiments presented in the paper titled "Learning Fast and Slow: Towards Inclusive Federated Learning".

**Abstract:**
Today’s deep learning systems rely on large amounts of useful data to make accurate predictions. Often such data is private and thus not readily available due to rising privacy concerns. Federated learning (FL) tackles this problem by training a shared model locally on devices to aid learning in a privacy-preserving manner Unfortunately, FL’s effectiveness degrades when model training involves clients with heterogeneous devices; a common case especially in developing countries. Slow clients are dropped in FL, which not only limits learning but also systematically excludes slow clients thereby potentially biasing results. We propose Hasaas; a system that tackles this challenge by adapting the model size for slow clients based on their hardware resources. By doing so, Hasaas obviates the need to drop slow clients which improves model accuracy and fairness. To improve robustness in the presence of statistical heterogeneity, Hasaas uses insights from the Central Limit Theorem to estimate model parameters in every round. Experimental evaluation involving large-scale simulations and a small-scale real testbed shows that Hasaas provides robust performance in terms of test accuracy, fairness, and convergence times compared to state-of-the-art schemes.

## Notes

- Install the libraries listed in ```requirements.txt```
    - I.e. with pip: run ```pip3 install -r requirements.txt```
- Go to directory of respective dataset for instructions on generating data
- Go to directory of respective approach for instructions to run experiments
- Information regarding mobile experiments can be found within the `pygrid-federated` directory.
