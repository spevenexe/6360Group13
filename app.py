import numpy as np
from src.indexNSG import IndexNSG

def main():
    dimension = 100
    n = 1000
    data = np.random.random((n, dimension)).astype(np.float32)
    index = IndexNSG(dimension, n, "L2")
    parameters = Parameters()
    parameters.set("example_param", 123)
    index.build(data, parameters)
    query = np.random.random(dimension).astype(np.float32)
    index.search(query, 5, parameters)

if __name__ == "__main__":
    main()
