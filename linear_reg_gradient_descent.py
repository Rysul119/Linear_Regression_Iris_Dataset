import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def gradient_descent(input, label, tot_epochs, learning_rate, tol):

    #prepare the input of size (Nx,2) by inserting a new column of value 1
    #getting the sample size
    n = input.shape[0]
    new_column = np.ones(n,) # a column vector of ones.
    modinput = np.column_stack((input.T, new_column)) # creates a (n,2) array
    #initialize the weights and biases(theta1, theta0) with values from normal distribution
    #initialize with a seed for repetition of result
    thetas = []
    costs = []
    # set seed
    np.random.seed(23)
    theta = np.random.normal(size=(1, modinput.shape[1]))
    #thetas.append(theta)

    for epoch in range(tot_epochs):
        #get the prediction
        y_hat = np.dot(modinput, theta.T)
        #calculate the cost
        diff = y_hat - label.reshape(n, 1)
        #print(diff.shape)
        cost = 0.5 * (np.square(diff)).mean(axis = 0)

        #print the cost
        #print('Epoch: {} Cost: {}'.format(epoch+1, cost[0]))

        #get the gradient
        dtheta0 =  np.dot(input.reshape(n,1).T, diff)/n   #slope
        dtheta1 =  np.mean(diff)   #intersection
        #print(dtheta0.shape)
        #update theta
        theta[0, 0] = (theta[0, 0] - learning_rate * dtheta0)[0]
        theta[0, 1] = theta[0, 1] - learning_rate * dtheta1
        # log the outputs
        thetas.append(theta)
        costs.append(cost[0])
        # Checking with tolerance
        if epoch > 0 and (costs[epoch - 1] - costs[epoch]) < tol:
            break

    output = {
        'theta': thetas,
        'cost': costs
    }
    #convert the dictionary to a dataframe.
    outputdf = pd.DataFrame(output)
    return outputdf

#main

max_iter = 500000 #number of iterations
tol = 0.00000000001 #stopping criteria
lrs = [0.00001, 0.0000001] #two learning rates

#load iris dataset
iris = load_iris()

x_column = 'sepal length (cm)'
y_column = 'petal width (cm)'
targets = ['versicolor', 'virginica']

for target in targets:
    # get the index value for the x and y column from dataset
    idx = iris.feature_names.index(x_column)
    idy = iris.feature_names.index(y_column)
    # get the index value for the target_names from dataset
    idtarget = np.where(iris.target_names == target)[0][0] #this slicing was needed for find the value of array in a tuple
    print(idtarget)
    x = iris.data[idtarget * 50 : (idtarget + 1) * 50,  idx] #get x values for the target
    y = iris.data[idtarget * 50 : (idtarget + 1) * 50,  idy] #get x values for the target
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
    plt.savefig(name, figsize = (8, 8), dpi = 400)
    plt.clf() # to clear out the previous iteration plot otherwise it was overlaying with the previous one.