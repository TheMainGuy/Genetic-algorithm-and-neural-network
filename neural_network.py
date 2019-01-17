import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Layer:
    def __init__(self, number_of_inputs, number_of_outputs):
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.weights = np.random.rand(number_of_inputs, number_of_outputs)
        self.w0 = np.random.rand(number_of_outputs)

    def forward(self, x):
        return sigmoid(np.dot(x, self.weights)) + self.w0

    def get_number_of_parameters(self):
        return (self.number_of_inputs + 1) * self.number_of_outputs

    def set_parameter(self, index, parameter):
        if index < self.number_of_inputs * self.number_of_outputs:
            self.weights[index // self.number_of_outputs, index % self.number_of_outputs] = parameter
        else:
            self.w0[index - self.number_of_inputs * self.number_of_outputs] = parameter

    def get_parameter(self, index):
        if index < self.number_of_inputs * self.number_of_outputs:
            return self.weights[index // self.number_of_outputs, index % self.number_of_outputs]
        return self.w0[index - self.number_of_inputs * self.number_of_outputs]


class FirstLayer:
    def __init__(self, number_of_inputs, number_of_outputs):
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.w = np.random.rand(number_of_outputs, number_of_inputs)
        self.s = np.random.rand(number_of_outputs, number_of_inputs)

    def forward(self, x):
        y = 1 / (1 + np.sum(np.abs(x - self.w) / np.abs(self.s), axis=1))
        return y

    def get_number_of_parameters(self):
        return self.number_of_inputs * self.number_of_outputs * 2

    def set_parameter(self, index, parameter):
        if index < self.number_of_inputs * self.number_of_outputs:
            self.w[index // self.number_of_inputs, index % self.number_of_inputs] = parameter
        else:
            index -= self.number_of_inputs * self.number_of_outputs
            self.s[index // self.number_of_inputs, index % self.number_of_inputs] = parameter

    def get_parameter(self, index):
        if index < self.number_of_inputs * self.number_of_outputs:
            return self.w[index // self.number_of_inputs, index % self.number_of_inputs]

        index -= self.number_of_inputs * self.number_of_outputs
        return self.s[index // self.number_of_inputs, index % self.number_of_inputs]


class NeuralNetwork:
    def __init__(self, architecture):
        self.layers = []
        for i in range(len(architecture)):
            if i == 0:
                continue
            elif i == 1:
                self.layers.append(FirstLayer(architecture[i - 1], architecture[i]))
            else:
                self.layers.append(Layer(architecture[i - 1], architecture[i]))

    def forward_pass(self, x):
        for layer in self.layers:
            y = layer.forward(x)
            x = y

        return y

    def calc_output_with_parameters(self, parameters, x):
        self.set_parameters(parameters)
        return self.forward_pass(x)

    def calc_output(self, x):
        return self.forward_pass(x)

    def calc_error(self, parameters, dataset):
        y = self.calc_multiple_output(parameters, dataset)
        t = dataset.get_classification()
        error = np.sum(np.power(y - t, 2)) / dataset.size()
        # print(error)
        return error

    def calc_multiple_output(self, parameters, dataset):
        x = dataset.get_x_y()

        self.set_parameters(parameters)
        y = []
        for single_x in x:
            y.append(self.calc_output(single_x))

        return y

    def calc_zero_one_error(self, parameters, dataset):
        y = self.calc_multiple_output(parameters, dataset)
        t = dataset.get_classification()
        return np.sum(np.round(y) - t)



    def get_number_of_parameters(self):
        number = 0
        for layer in self.layers:
            number += layer.get_number_of_parameters()

        return number

    def set_parameters(self, parameters):
        if len(parameters) != self.get_number_of_parameters():
            raise ValueError('Number of parameters not equal to ' + str(self.get_number_of_parameters()))

        i = 0
        for layer in self.layers:
            for j in range(layer.get_number_of_parameters()):
                layer.set_parameter(j, parameters[i])
                i += 1

    def get_parameters(self):
        parameters = []
        for layer in self.layers:
            for j in range(layer.get_number_of_parameters()):
                parameters.append(layer.get_parameter(j))

        return parameters
