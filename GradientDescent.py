# Gradient Descent from Scratch

"""
Formulas :-
    - yhat = wx + b
    - loss = (y - yhat) ** 2 / N
"""


import numpy as np

x = np.random.randn(10, 1)
y = 5 * x + np.random.rand()

w = 0.0
b = 0.0

learningRate = 0.01


def gradientDescent(w, b, x, y, learningRate):
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]

    for xi, yi in zip(x, y):
        dldw += -2*xi*(yi-(w*xi+b))
        dldb += -2*(yi-(w*xi+b))

    w = w - learningRate*(1/N)*dldw
    b = b - learningRate*(1/N)*dldb

    return w, b


for iteration in range(500):
    w, b = gradientDescent(w, b, x, y, learningRate)
    yhat = w*x + b
    loss = np.divide(np.sum((y-yhat)**2, axis=0), x.shape[0])
    print(f'{iteration} loss is {loss}, paramters w:{w}, b:{b}')
print(x, y)