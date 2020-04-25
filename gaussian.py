# coding: UTF-8


import numpy as np



x = 3
y = 3
sigma = 1.3


def norm2d(x, y, sigma):

    z = np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2)) # / (np.sqrt(2 * np.pi) * sigma)
    return z


fil = [[np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2)) for j in range(y)] for i in range(x)]

fil = np.array(fil)

print(fil)
print(np.sqrt(2 * np.pi) * sigma)

size = 3
sigma = (size - 1) / 2.
print(sigma)

x = np.arange(0, size) - sigma
y = np.arange(0, size) - sigma
X, Y = np.meshgrid(x, y)
print(X)
print(Y)

fil = norm2d(X, Y, sigma)
print(2 * np.pi * sigma ** 2)
# fil = fil / fil.sum()
print(fil)
