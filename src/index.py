import numpy as np

def distance_l2(a, b):
    return np.sum((a - b) ** 2)

def distance_inner_product(a, b):
    return np.dot(a, b)

class Index:
    def __init__(self, dimension, n, metric):
        self.dimension = dimension
        self.n = n
        self.data = None
        self.has_built = False
        if callable(metric):
            self.distance = metric
        elif metric == "L2":
            self.distance = distance_l2
        elif metric == "inner_product":
            self.distance = distance_inner_product
        else:
            raise ValueError("Unsupported metric type")

    def build(self, data, parameters):
        raise NotImplementedError("This method must be implemented by subclasses")

    def search(self, query, k, parameters):
        raise NotImplementedError("This method must be implemented by subclasses")

    def save(self, filename):
        raise NotImplementedError("This method must be implemented by subclasses")

    def load(self, filename):
        raise NotImplementedError("This method must be implemented by subclasses")
