from optimizers import  gradient_descent
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris



# Testing the custom gradient descent for iris dataset

max_iter = 500000 # number of iterations
tol = 0.00000000001 # stopping criteria
lrs = [0.00001, 0.0000001] # two learning rates
# set seed
np.random.seed(23)


# load iris dataset
iris = load_iris()

x_column = 'sepal length (cm)'
y_column = 'petal width (cm)'
targets = ['versicolor', 'virginica']

for target in targets:
    # get the index value for the x and y column from dataset
    idx = iris.feature_names.index(x_column)
    idy = iris.feature_names.index(y_column)
    # get the index value for the target_names from dataset
    idtarget = np.where(iris.target_names == target)[0][0] # this slicing was needed for find the value of array in a tuple
    # print(idtarget)
    x = iris.data[idtarget * 50 : (idtarget + 1) * 50,  idx] # get x values for the target
    y = iris.data[idtarget * 50 : (idtarget + 1) * 50,  idy] # get x values for the target
    results = []
    for lr in lrs:
        # fit and get a DataFrame with columns ['theta', 'cost']
        print('Regression has started running for {} with learning rate: {}'.format(target, lr))
        linreg = gradient_descent(x, y, max_iter, lr, tol)
        results.append(linreg)

    plt.plot(results[0]['cost'], c = 'r' , label = 'learning rate = 1e-5')
    plt.plot(results[1]['cost'], c = 'b' , label = 'learning rate = 1e-7')
    plt.xlabel('Iterations')
    plt.ylabel('Costs')
    plt.legend(loc = 'upper right')
    name = target + x_column.split(' ')[0]+'.png'
    plt.savefig(name, figsize = (8, 8), dpi = 400) #save the plots as png
    plt.clf() # to clear out the previous iteration plot otherwise it was overlaying with the previous one.