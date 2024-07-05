
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rn
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.figure_factory as ff

NUMPARAMS = 2
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)

def f1(params):
    x = params[0]
    y = params[1]
    return x**2 + 100 * (y - x**2)**2

def df1(params):
    x = params[0]
    y = params[1]
    return np.array([2*x - 400*x*(y - x**2), 200*(y - x**2)])

def f2(params):
    x = params[0]
    y = params[1]
    a = x**2 + y**2
    return (50/9)*(a**3) - (209/18)*(a**2) + (59/9)*(a)

def df2(params):
    x = params[0]
    y = params[1]
    a = x**2 + y**2
    return np.array([100*x*(a - (209/18)), 100*y*(a - (209/18))])

def f3(params):
    x = params[0]
    y = params[1]
    return (x*y)**2/1000

def df3(params):
    x = params[0]
    y = params[1]
    return np.array([2*x*y*y/1000, 2*x*y*x/1000])

def f4(params):
    x = params[0]
    y = params[1]
    return (np.sin(x) + np.sin(y))**2

def df4(params):
    x = params[0]
    y = params[1]
    return np.array([2*(np.sin(x)+np.sin(y))*np.cos(x), 2*(np.sin(x)+np.sin(y))*np.cos(y)])

def f5(params):
    x = params[0]
    y = params[1]
    return (x**4-2*x**3-10*x**2+8*x+40)/10

def df5(params):
    x = params[0]
    y = params[1]
    return np.array([(4*x**3-6*x**2-20*x+8)/10, 0])

def f6(params):
    x = params[0]
    y = params[1]
    return np.sin(x)**2 + np.cos(x)**2 + np.sin(y)**2 + np.cos(y)**2

def df6(params):
    x = params[0]
    y = params[1]
    return np.array([0, 0])

def adam_optimizer(df, numparams, lr, tol, beta1, beta2, lambdaa):
    curr_params = init_params
    params = init_params
    m = np.zeros(numparams)
    v = np.zeros(numparams)
    t = 0
    while True:
        flag = False
        t = t + 1
        grad = df(curr_params)
        m = beta1*m + (1 - beta1)*grad
        v = beta2*v + (1 - beta2)*(grad**2)
        mhat = m/(1 - beta1**t)
        vhat = v/(1 - beta2**t)
        curr_params = curr_params - lr*(mhat/(np.sqrt(vhat) + lambdaa * curr_params))
        params = np.vstack((params, curr_params))
        if np.linalg.norm(grad) > tol:
            flag = True
        if flag == False:
            break
    return params

def lion_optimizer(df, numparams, lr, tol, beta1, beta2, lambdaa):
    max_it = 20000
    curr_params = init_params
    params = init_params
    m = np.zeros(numparams)
    # v = np.zeros(numparams)
    # t = 0
    h = 0
    while True:
        h += 1
        if h == max_it:
            break
        flag = False
        # t = t + 1
        grad = df(curr_params)
        # m = beta1*m + (1 - beta1)*grad
        # v = beta2*v + (1 - beta2)*(grad**2)
        # mhat = m/(1 - beta1**t)
        # vhat = v/(1 - beta2**t)
        curr_params = curr_params - lr * (np.sign(beta1 * m + (1 - beta1) * grad) + lambdaa * curr_params)
        params = np.vstack((params, curr_params))
        m = beta2 * m + (1 - beta2) * grad
        if np.linalg.norm(grad) > tol:
            flag = True
        if flag == False:
            break
    return params

def plot_subplots(f, params):
    # Plot the gradient descent path for each optimizer
    fig = plt.figure(figsize=(15, 7))
    fig.suptitle('Gradient Descent Paths')

    # Create 3D subplot
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    # Plot the contour in 3D
    x = np.linspace(-5, 7, 1000)
    y = np.linspace(-5, 7, 1000)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

    # Calculate function evaluations for each point
    Z_points = f(params[0].T)

    # Plot the arrows
    ax.quiver(params[0][:-1, 0], params[0][:-1, 1], Z_points[:-1],
              params[0][1:, 0] - params[0][:-1, 0], params[0][1:, 1] - params[0][:-1, 1], Z_points[1:] - Z_points[:-1],
              color='k', length=0.1, normalize=True)

    # Plot the final point
    ax.scatter(params[0][-1, 0], params[0][-1, 1], Z_points[-1], c='r', marker='o')

    # Set title
    ax.set_title('Lion')

    # Create 3D subplot
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    # Plot the contour in 3D
    x = np.linspace(-5, 7, 1000)
    y = np.linspace(-5, 7, 1000)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

    # Calculate function evaluations for each point
    Z_points = f(params[1].T)

    # Plot the arrows
    ax.quiver(params[1][:-1, 0], params[1][:-1, 1], Z_points[:-1],
              params[1][1:, 0] - params[1][:-1, 0], params[1][1:, 1] - params[1][:-1, 1], Z_points[1:] - Z_points[:-1],
              color='k', length=0.1, normalize=True)

    # Plot the final point
    ax.scatter(params[1][-1, 0], params[1][-1, 1], Z_points[-1], c='r', marker='o')

    # Set title
    ax.set_title('Adam W')

    plt.show()

# LR = 0.02
# LIMIT = 0.5
# GAMMA = 0.9
# BETA1 = 0.9
# BETA2 = 0.95
# LAMBDAA = 1e-8
# init_params = np.array([4, 3.5])

# LR = 0.001
# LIMIT = 0.1
# GAMMA = 0.9
# BETA1 = 0.9
# BETA2 = 0.95
# LAMBDAA = 1e-8
# init_params = np.array([4, 3.5])

# params = [lion_optimizer(df1, NUMPARAMS, LR, LIMIT, BETA1, BETA2, LAMBDAA), adam_optimizer(df1, NUMPARAMS, LR, LIMIT, BETA1, BETA2, LAMBDAA)]
# print("Lion Optimizer : ", len(params[0]), "Adam Optimizer : ", len(params[1]))
# plot_subplots(f1, params)

# LR = 0.001
# LIMIT = 0.01
# GAMMA = 0.9
# BETA1 = 0.9
# BETA2 = 0.95
# LAMBDAA = 1e-8
# init_params = np.array([6, 5.5])

# LR = 0.01
# LIMIT = 0.01
# GAMMA = 0.9
# BETA1 = 0.9
# BETA2 = 0.95
# LAMBDAA = 1e-8
# init_params = np.array([6, 5.5])

# LR = 0.1
# LIMIT = 0.01
# GAMMA = 0.9
# BETA1 = 0.9
# BETA2 = 0.95
# LAMBDAA = 1e-8
# init_params = np.array([6, 5.5])

# LR = 1
# LIMIT = 0.01
# GAMMA = 0.9
# BETA1 = 0.9
# BETA2 = 0.95
# LAMBDAA = 1e-8
# init_params = np.array([6, 5.5])

# params = [lion_optimizer(df2, NUMPARAMS, LR, LIMIT, BETA1, BETA2, LAMBDAA), adam_optimizer(df2, NUMPARAMS, LR, LIMIT, BETA1, BETA2, LAMBDAA)]
# print("Lion Optimizer : ", len(params[0]), "Adam Optimizer : ", len(params[1]))
# plot_subplots(f2, params)

# LR = 1
# LIMIT = 0.01
# GAMMA = 0.9
# BETA1 = 0.9
# BETA2 = 0.95
# LAMBDAA = 1e-8
# init_params = np.array([6, 5.5])

# params = [lion_optimizer(df3, NUMPARAMS, LR, LIMIT, BETA1, BETA2, LAMBDAA), adam_optimizer(df3, NUMPARAMS, LR, LIMIT, BETA1, BETA2, LAMBDAA)]
# print("Lion Optimizer : ", len(params[0]), "Adam Optimizer : ", len(params[1]))
# plot_subplots(f3, params)

# LR = 0.001
# LIMIT = 0.0001
# GAMMA = 0.9
# BETA1 = 0.9
# BETA2 = 0.95
# LAMBDAA = 1e-8
# init_params = np.array([1.57, 2])

# LR = 0.001
# LIMIT = 0.01
# GAMMA = 0.9
# BETA1 = 0.9
# BETA2 = 0.95
# LAMBDAA = 1e-8
# init_params = np.array([1.57, 2])

# LR = 0.0001
# LIMIT = 0.0001
# GAMMA = 0.9
# BETA1 = 0.9
# BETA2 = 0.95
# LAMBDAA = 1e-8
# init_params = np.array([1.57, 2])

# params = [lion_optimizer(df4, NUMPARAMS, LR, LIMIT, BETA1, BETA2, LAMBDAA), adam_optimizer(df4, NUMPARAMS, LR, LIMIT, BETA1, BETA2, LAMBDAA)]
# print("Lion Optimizer : ", len(params[0]), "Adam Optimizer : ", len(params[1]))
# plot_subplots(f4, params)

# LR = 0.1
# LIMIT = 0.01
# GAMMA = 0.9
# BETA1 = 0.9
# BETA2 = 0.95
# LAMBDAA = 1e-8
# init_params = np.array([6, 5.5])

# params = [lion_optimizer(df5, NUMPARAMS, LR, LIMIT, BETA1, BETA2, LAMBDAA), adam_optimizer(df5, NUMPARAMS, LR, LIMIT, BETA1, BETA2, LAMBDAA)]
# print("Lion Optimizer : ", len(params[0]), "Adam Optimizer : ", len(params[1]))
# plot_subplots(f5, params)

LR = 1
LIMIT = 0.0001
GAMMA = 0.9
BETA1 = 0.9
BETA2 = 0.95
LAMBDAA = 1e-8
init_params = np.array([6, 5.5])

params = [lion_optimizer(df6, NUMPARAMS, LR, LIMIT, BETA1, BETA2, LAMBDAA), adam_optimizer(df6, NUMPARAMS, LR, LIMIT, BETA1, BETA2, LAMBDAA)]
print("Lion Optimizer : ", len(params[0]), "Adam Optimizer : ", len(params[1]))
plot_subplots(f6, params)