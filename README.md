# Code of causalKANS: interpretable treatment effect estimation with Kolmogorov-Arnorld networks

## Get started

Install the dependecies with 
`pip install -r requirements.txt`. In an evironment of python 3.10

Note that tueplots is a package only used for plot styling. You can remove it, taking care of the plot styles in the code.

## Usage

To replicate the experiments in the paper, follow these steps:

1. The comparison with causalNN is made by
    2.Running the scripts of causalKAN and causalNNs hyperparameters search `python experiment_search_kan.py` and `python experiment_search_mlp.py`
    3. Running the script of metric extraction, which will produce the results for constructing Table 1, Table 3 and Table 4

2. The visualization and interpretability results can be run from the notebooks
    - `visualization_acic_tkaam.ipynb` for ACIC-7 data and T-KAAM model
    - `visualization_ihdp_skaam.ipynb` for IHDP A and S-KAAM model
    - `visualization_other_datasets.ipynb` for IHDP B and ACIC-2 with DragonKAAM
Table 2 was also extracted from these notebooks.

Apart of that, representation tools are stored in `representation.py`. The alternative symbolic substitution that we used is stored in `experiment_symbolic.py`
Other utils can be found in `utils.py`. causalKANs models and causalNNs models can be found in `kan_model`and `mlp_model` respectively.