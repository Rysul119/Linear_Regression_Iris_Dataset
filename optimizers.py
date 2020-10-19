import numpy as np
import pandas as pd

# gradient descent algorithm for linear regression
def gradient_descent(input, label, tot_epochs, learning_rate, tol):

    # prepare the input of size (Nx,2) by inserting a new column of value 1
    # getting the sample size
    n = input.shape[0]
    new_column = np.ones(n,) # a column vector of ones.
    modinput = np.column_stack((input.T, new_column)) # creates a (n,2) array
    # initialize the weights and biases(theta1, theta0) with values from normal distribution
    # initialize with a seed for repetition of result
    output = {
        'theta': [],
        'cost': []
    }
    theta = np.random.normal(size=(1, modinput.shape[1]))
    # thetas.append(theta)

    for epoch in range(tot_epochs):
        # get the prediction
        y_hat = np.dot(modinput, theta.T)
        # calculate the cost
        diff = y_hat - label.reshape(n, 1)
        # print(diff.shape)
        cost = 0.5 * (np.square(diff)).mean(axis = 0)

        # print the cost
        # print('Epoch: {} Cost: {}'.format(epoch+1, cost[0]))

        # get the gradient
        dtheta0 =  np.dot(input.reshape(n,1).T, diff)/n   #slope
        dtheta1 =  np.mean(diff)   #intersection
        # print(dtheta0.shape)
        # update theta
        theta[0, 0] = (theta[0, 0] - learning_rate * dtheta0)[0]
        theta[0, 1] = theta[0, 1] - learning_rate * dtheta1
        # log the outputs
        output["theta"].append(theta)
        output["cost"].append(cost[0])
        # Checking with tolerance
        if epoch > 0 and (output["cost"][epoch - 1] - output["cost"][epoch]) < tol:
            break

    # convert the dictionary to a dataframe.
    outputdf = pd.DataFrame(output)
    return outputdf