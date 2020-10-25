# Linear_Regression_Iris_Dataset


This repository contains an implementation of a linear regression algorithm using gradient descent to update the weights and biases while calculating the loss/cost using mean squared loss, $MSE (X,\theta)= \frac{1}{2N}\sum_{i = 1}^{N}\left (\hat{y_{i}} - y_{i} \right)^{2}$. 

All the costs and weights after each iteration are logged inside a dataframe `outputdf` which can be used to visualize the variation of cost and weights against training iterations. An early stopping criterion ($\epsilon$) has been incorporated which stops the training process at the iteration `i` when $cost_{i} - cost_{i-1} < \epsilon$ becomes true. 

Function `gradient_descent` is tested by training for the given attribute `sepal length` with label `petal width` for two classes of flowers `Iris Versicolor` and `Iris Virginica` collected from [Iris Dataset](http://archive.ics.uci.edu/ml/datasets/Iris/). 

You will need to install [python](https://www.python.org), [numpy](https://numpy.org), [pandas](https://pandas.pydata.org), [matplotlib](https://matplotlib.org), and [scikit-learn](https://scikit-learn.org/stable/). [Scikit-learn](https://scikit-learn.org/stable/) is used for importing the [Iris Dataset](http://archive.ics.uci.edu/ml/datasets/Iris/). If you have anaconda installed, run the following:
```bash
conda create -n envName python numpy pandas matplotlib scikit-learn
```
This will create a conda environment with python, numpy, pandas, matplotlib, and scikit-learn installed in it. Run `conda activate envName` to activate or `conda deactivate` to deactivate the environment.

If you are not seeing the equations above, please install the [MathJax](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima?hl=en) plugin for your chrome browser.
