import numpy as np
import neural_network
import dataset
import random

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