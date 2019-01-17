import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_class(row):
    for c in data.columns:
        if row[c] == 1:
            return c

data = pd.read_table('zad7-dataset.txt', '\t', names=['x', 'y', 'a', 'b', 'c'])
class_a = data.loc[data.a == 1]
class_b = data.loc[data.b == 1]
class_c = data.loc[data.c == 1]

plt.scatter(class_a['x'], class_a['y'], label='class a')
plt.scatter(class_b['x'], class_b['y'], label='class b')
plt.scatter(class_c['x'], class_c['y'], label='class c')
plt.legend(loc='best')
plt.show()