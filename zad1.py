import numpy as np
import matplotlib.pyplot as plt


def neuron_1(x, w, s):
    return 1 / (1 + np.abs(x - w) / np.abs(s))

x = np.linspace(-8, 10)

for s in [1, 0.25, 4]:
    y = lambda x: neuron_1(x, 2, s)
    plt.plot(x, y(x), label='s = ' + str(s))


plt.legend(loc='best')
plt.show()