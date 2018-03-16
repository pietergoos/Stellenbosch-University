from numpy import loadtxt, where, zeros, e, array, log, ones, append, linspace
from pylab import scatter, show, legend, xlabel, ylabel, contour, title
from scipy.optimize import fmin_bfgs


def sigmoid(X):
    '''Compute the sigmoid function '''
    #d = zeros(shape=(X.shape))

    den = 1.0 + e ** (-1.0 * X)

    d = 1.0 / den

    return d


def cost_function_reg(theta, X, y, l):
    '''Compute the cost and partial derivatives as grads
    '''

    h = sigmoid(X.dot(theta))

    thetaR = theta[1:, 0]

    J = (1.0 / m) * ((-y.T.dot(log(h))) - ((1 - y.T).dot(log(1.0 - h)))) \
            + (l / (2.0 * m)) * (thetaR.T.dot(thetaR))

    delta = h - y
    sumdelta = delta.T.dot(X[:, 1])
    grad1 = (1.0 / m) * sumdelta

    XR = X[:, 1:X.shape[1]]
    sumdelta = delta.T.dot(XR)

    grad = (1.0 / m) * (sumdelta + l * thetaR)

    out = zeros(shape=(grad.shape[0], grad.shape[1] + 1))

    out[:, 0] = grad1
    out[:, 1:] = grad

    return J.flatten(), out.T.flatten()




m, n = X.shape

y.shape = (m, 1)

it = map_feature(X[:, 0], X[:, 1])

#Initialize theta parameters
initial_theta = zeros(shape=(it.shape[1], 1))

#Set regularization parameter lambda to 1
l = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = cost_function_reg(initial_theta, it, y, l)

def decorated_cost(theta):
    return cost_function_reg(theta, it, y, l)

print fmin_bfgs(decorated_cost, initial_theta, maxfun=400)
