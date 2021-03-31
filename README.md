# variational_bayes_models
This is a Python3 library implementing the linear and logistic variational Bayes models as formulated in PRML and other sources. The mathematical derivation with step-by-step explanations are provided for both models in PDF files. Additionally, Jupyter notebook files showcase the flow of code for easy understanding.

Figures showcasing output from the linear (left) and logistic (right) models:

![VB model output figures](https://user-images.githubusercontent.com/34254153/113080331-b729c780-9211-11eb-8429-5b6c6ba0961e.png "VB model output figures")

### How to use

1. Add the library as a submodule within your repository
```python:
git submodule add https://github.com/rufuzzz0/variational_bayes_models.git
```

2. Import module in your file
```python:
# from variational_bayes_models.variational_logistic_regression.var_log_reg import VarLogRegModel
```

3. Initialize the model
```python:
model = VarLogRegModel()
```

4. Train the model using a dataset
```python:
dataset = ('data_matrix X', 'target labels Y')
model.train(dataset)
```

5. Do inference on new input feature vectors x
```python:
p(Y=True) = model.predictive_posterior_distr(x)
```

### Dependencies

Python3

- Version: 3.6
- Packages:
 - matplotlib
 - numpy
 - scikit-learn

### Dataset structure

A dataset consists of a tuple (X, Y).

- X denotes a data matrix with dimensions (#samples N, #input features M).
- Y denotes a target column vector with dimensions (#samples N, 1).

Note that the dataset do not have to be normalized as this is done within the module.

### Inference sample structure

An input feature vector x is a column vector with dimensions (#input features M, 1).

Note that the feature vector do not have to be normalized as this is done within the module (applying the same normalization as used on the dataset the model was trained on).

### Repository file structure

```
variational_bayes_models
└───variational_linear_regression
│   │   variational_bayes_regression_notes.pdf
│   │   var_lin_reg.ipynb  # Jupter notebook demonstration
│   │   var_lin_reg.py     # Importable module
│   
└───variational_logistic_regression
|   |   local_variational_methods_notes.pdf
|   |   variational_bayes_logistic_regression_notes.pdf
|   |   var_log_reg.ipynb  # Jupter notebook demonstration
|   |   var_log_reg.py     # Importable module
|
|   LICENSE
|   README.md
```