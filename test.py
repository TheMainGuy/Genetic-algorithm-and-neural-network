import numpy as np
import neural_network
import dataset
import random
import matplotlib.pyplot as plt

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
# plot = plt.scatter(x[:, 0], x[:, 1], c=np.argmax(classified, axis=1))
# plt.scatter(nn.layers[0].w[:, 0], nn.layers[0].w[:, 1], label='w')
# plt.legend(loc='best')
# plt.show()
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
classified2 = nn2.classify_dataset(parameters2, data)
print(nn2.calc_zero_one_error(parameters2, data))
print(nn2.layers[0].s)
plot = plt.scatter(x[:, 0], x[:, 1], c=np.argmax(classified2, axis=1))
plt.scatter(nn2.layers[0].w[:, 0], nn2.layers[0].w[:, 1], label='w')
plt.legend(loc='best')
plt.show()
