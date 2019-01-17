import numpy as np


class Dataset:
    def __init__(self, path_to_data):
        self.x = []
        self.y = []
        self.xy = []
        self.classification = []
        file = open(path_to_data, 'r')
        for line in file:
            line = line.strip()
            parts = line.split('\t')
            self.x.append(float(parts[0]))
            self.y.append(float(parts[1]))
            self.xy.append([float(parts[0]), float(parts[1])])
            self.classification.append([float(parts[2]), float(parts[3]), float(parts[4])])

    def size(self):
        return len(self.x)

    def get(self, index):
        return self.x[index], self.y[index], self.classification[index]

    def get_x_y(self):
        return np.array(self.xy)

    def get_classification(self):
        return np.array(self.classification)
