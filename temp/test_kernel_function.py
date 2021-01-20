import numpy as np
from math import exp
import matplotlib.pyplot as plt

X = np.linspace(-10, 10, 100, endpoint=True)
Y = np.zeros(X.shape)

for i,x in enumerate(X):
    Y[i] = -1.0 / (exp(x) + 2 + exp(-x))
    temp = 1

plt.figure(1)
plt.plot(X,Y)
plt.show()

temp = 1