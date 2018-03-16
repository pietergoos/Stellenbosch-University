import math
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
df = pd.read_csv("data.csv", header=0)
df.columns = ["grade1","grade2","label"]
x = df["label"].map(lambda x: float(x.rstrip(';')))
X = df[["grade1","grade2"]]
X = np.array(X)
X = min_max_scaler.fit_transform(X)
Y = df["label"].map(lambda x: float(x.rstrip(';')))
Y = np.array(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)




def sigmoid(z):
    s= 1/(1 + np.exp(-z))
    return s

def propagate(w, b, X, Y):

    m = X.shape[1]

    A = sigmoid(np.dot(w.T,X)+b)                                     # compute activation

    cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * (np.log(1-A)))

    dz= (1/m)*(A - Y)
    dw = np.dot(X, dz.T)
    db = np.sum(dz)


    cost = np.squeeze(cost)
    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):

    costs = []

    for i in range(num_iterations):
        m = X.shape[1]
        grads,cost = propagate(w, b, X, Y)
        b = b - learning_rate*grads["db"]
        w = w - learning_rate*grads["dw"]
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}
    return params, grads, costs



def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T,X)+ b)

    for i in range(A.shape[1]):
        x_exp = np.exp(A)
        x_sum = np.sum(x_exp,axis=1,keepdims=True)
        s = np.divide(x_exp,x_sum)

    Y_prediction = 1. * (A > 0.5)

    return Y_prediction


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros((dim,1))
    b = 0
    ### END CODE HERE ###

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])

    print("learning rate:",learning_rate)

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = False)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_train = predict(w,b,X_train)
    Y_prediction_test = predict(w,b,X_test)
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d

d = model(X_train, Y_train, X_test, Y_test, num_iterations = 1000, learning_rate = 0.0001, print_cost = True)
