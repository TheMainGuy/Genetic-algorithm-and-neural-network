import numpy as np
import neural_network
import dataset
import random
import matplotlib.pyplot as plt
import pandas as pd
nn = neural_network.NeuralNetwork([2, 8, 3])

# parameters = nn.get_parameters()
# print(parameters)
# parameters[0] = 0
# parameters[1] = 5
# parameters[10] = 10
# parameters[43] = parameters[44] + 1
# parameters[40] = 100
# parameters[57] = 200
# nn.set_parameters(parameters)
# print(nn.layers[0].w)
# print(nn.layers[0].s)
# print(nn.layers[1].weights)
# print(nn.layers[1].w0)
# print(nn.get_number_of_parameters())

data = dataset.Dataset('zad7-dataset.txt')
parameters = [8.67773783e-01, 7.32259736e-01, 1.28398696e-01, 7.39753406e-01, 8.73448409e-01, 2.58567249e-01,
              6.22389919e-01, 7.44043665e-01, 6.30455979e-01, 2.60093070e-01, 3.78412124e-01, 2.58334675e-01,
              1.31651259e-01, 2.61920133e-01, 3.67127942e-01, 7.45221202e-01, 2.77544608e-01, 4.31119625e-01,
              3.04022042e-01, 3.34643944e-01, -1.26283686e-01, -1.48948806e-01, 1.23435163e-01, 3.22733685e-01,
              -8.46105676e-02, -1.75098739e-01, 6.14378804e-02, 1.43710407e-01, -1.10594116e-01, -1.67289589e-01,
              8.46201175e-02, 2.36570645e-01, -3.21197109e+01, 3.51661982e+01, -1.67185305e+01, -2.34271214e+01,
              3.63863628e+01, -1.42044212e+01, 5.01697124e+01, -3.06110825e+01, -1.95300219e+01, 6.33983664e+01,
              -3.35611747e+01, -2.76334634e+01, -5.42383738e+01, -3.25739141e+01, 7.06558715e+01, -3.83752455e+01,
              8.13505270e+01, -3.87341347e+01, 4.78022985e+01, -4.14143241e+01, -3.70354666e+01, -1.33985010e+01,
              -5.34848507e+01, 6.91325373e+01, -1.67980991e-04, -7.56807829e-05, -4.06990603e-05]
error = nn.calc_error(parameters, data)
print(error)
output = nn.calc_multiple_output(parameters, data)
print(output)
zero_one_error = nn.calc_zero_one_error(parameters, data)
print(zero_one_error)
classified = nn.classify_dataset(parameters, data)
print(classified)
x = data.get_x_y()
# plt.plot(1, 2, 1)
# plot = plt.scatter(x[:, 0], x[:, 1], c=np.argmax(classified, axis=1))
def get_class(row):
    for c in data.columns:
        if row[c] == 1:
            return c

data32 = pd.read_table('zad7-dataset.txt', '\t', names=['x', 'y', 'a', 'b', 'c'])
class_a = data32.loc[data32.a == 1]
class_b = data32.loc[data32.b == 1]
class_c = data32.loc[data32.c == 1]

# plt.scatter(class_a['x'], class_a['y'], label='class a')
# plt.scatter(class_b['x'], class_b['y'], label='class b')
# plt.scatter(class_c['x'], class_c['y'], label='class c')
# plt.legend(loc='best')
# plt.show()
plt.scatter(nn.layers[0].w[:, 0], nn.layers[0].w[:, 1], label='w [2 , 8 , 3]')
x = range(0, 8)
print(nn.layers[1].weights[:,0])
plt.scatter(x, nn.layers[1].weights[:,0], label='Težine A')
plt.scatter(x, nn.layers[1].weights[:,1], label='Težine B')
plt.scatter(x, nn.layers[1].weights[:,2], label='Težine C')
print("weights")
print(nn.layers[1].weights)
plt.legend(loc='best')
plt.show()
print(nn.layers[0].s)

parameters2 = [4.02105624e-01, 2.78389424e-01, 1.22759760e+00, 5.92397696e-01, 1.03943280e-01, 7.32920428e-01,
               1.06904797e-01, 2.50502274e-01, 8.95257703e-01, 7.29911178e-01, -9.07814940e-01, 7.50136446e-01,
               -9.29827626e-02, 2.59823341e-01, -1.48812294e+00, 2.79647463e-01, 1.13672078e-01, -2.13714991e-01,
               1.62581543e+00, -2.80371397e+00, 7.45698661e-02, 1.40644193e-01, 1.07243953e+00, -3.79470213e-01,
               -4.81248508e+00, -1.26621192e+01, -4.02552301e+00, -4.32134078e+01, 5.33218234e+00, 1.83112984e+00,
               -2.62820520e-01, 1.67740200e+01, -6.03360497e+00, -1.98624058e-01, -2.69610042e+00, -8.86585567e-01,
               8.76681778e-01, 1.14932690e+00, 1.71258736e+00, -1.80641826e+01, -6.99462166e+00, 3.28566465e+00,
               -3.28293374e+00, 1.58566527e+02, 2.15936667e+00, 1.57303600e+00, 2.56208335e+00, 2.11722625e+01,
               -2.17292208e-01, -1.64902631e-01, -6.07504416e-01, -7.19583651e-02, 5.50234624e+01, -1.00309292e+02,
               4.00142835e+00, 5.83766074e+01, 3.16159397e+01, -9.94352149e+01, 2.08143263e+02, -1.06766473e+02,
               8.45334536e+01, -2.43672106e+01, 3.10090953e+00, 3.62831625e+01, 1.87070578e-04, -5.71706944e-06,
               -3.80630006e-05]

nn2 = neural_network.NeuralNetwork([2, 6, 4, 3])
# classified2 = nn2.classify_dataset(parameters2, data)
# plt.plot(1, 2, 2)
# print(nn2.calc_zero_one_error(parameters2, data))
# print(nn2.layers[0].s)
# plot = plt.scatter(x[:, 0], x[:, 1], c=np.argmax(classified2, axis=1))
# plt.scatter(nn2.layers[0].w[:, 0], nn2.layers[0].w[:, 1], label='w [2, 6, 4, 3')

parameters3 = [ 8.92721336e-01, 2.43593419e-01, 1.30923900e-01, 2.23939262e-01,
  2.89384487e-01, 5.16882651e-01, 1.21815173e-01, 3.34443804e-01,
  5.92040892e-01, 7.48883925e-01, 4.89330174e-01, 2.20798590e-01,
  8.50696826e-01, 7.49407046e-01, 1.03332453e+00, 8.12449787e-01,
  7.47546285e-02,-2.21493054e-01,-8.46632457e-02, 1.63732727e-01,
 -3.68207680e+00, 3.48907488e-02,-1.06818484e-01,-1.07755414e-01,
 -6.32628877e-02, 2.65494320e-01,-1.10777438e-01,-1.62441703e-01,
 -1.46840841e-01, 3.47544281e-01, 5.12650540e+00, 7.37870865e-01,
  5.98828759e-01, 4.25528380e+00,-9.81809764e+00,-2.96311288e+01,
  1.05375448e+01, 4.11924118e+00,-2.45651302e+00,-7.61098850e+00,
 -5.18198660e+00,-2.68548466e+00, 4.31958181e+00, 1.66914677e+01,
  1.00037678e+01, 2.97697876e+00, 6.10009649e-01, 6.80423143e-01,
 -2.45350629e+01, 9.09772967e+00,-1.55658991e+01, 2.47481615e+00,
  2.68665012e+00,-5.06235061e+00, 2.29313443e+01, 2.40236463e+00,
  5.51299190e+00,-1.87686654e+00,-6.74348257e-01, 1.05223946e+01,
 -1.03901886e+00,-2.73560401e+00, 1.65800416e+00, 1.87880383e+00,
 -1.67331599e-01,-2.05154355e-01, 2.04497791e-02, 2.54821985e+00,
  1.19576718e+01, 4.07394755e+01,-4.17789262e+01, 4.40326919e+01,
 -7.01508409e+01,-2.31386439e+01,-2.46120739e+01,-1.59109793e+00,
  2.50059573e+01,-1.66583296e+00,-2.56517067e-01,-2.20615178e+00,
  9.15251718e-06, 2.22106969e-04,-1.73409749e-04]

# nn3 = neural_network.NeuralNetwork([2, 8, 4, 3])
# classified3 = nn3.classify_dataset(parameters3, data)
# plt.scatter(nn3.layers[0].w[:, 0], nn3.layers[0].w[:, 1], label='w [2 , 8, 4 , 3]')
# plt.legend(loc='best')
# plt.show()
