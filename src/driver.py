import numpy as np
import struct
import os
import sys
import time
from indexNSG import *

def load_data(filename):
    with open(filename, 'rb') as file:
        dim_bytes = file.read(4)
        dim = struct.unpack('<I', dim_bytes)[0]

        file.seek(0, os.SEEK_END)
        fsize = file.tell()
        num = fsize // ((dim + 1) * 4)

        data = np.empty((num, dim), dtype=np.int32)

        file.seek(0)
        for i in range(num):
            file.seek(4, os.SEEK_CUR)
            data_bytes = file.read(dim * 4)
            data[i] = np.array(struct.unpack('<' + 'i'*dim, data_bytes)) 

    return data, num, dim

def main():
    if len(sys.argv) != 7:
        print(f"Usage: {sys.argv[0]} data_file nn_graph_path L R C save_graph_file")
        sys.exit(-1)

    data_file = sys.argv[1]
    nn_graph_path = sys.argv[2]
    L = int(sys.argv[3])
    R = int(sys.argv[4])
    C = int(sys.argv[5])
    save_graph_file = sys.argv[6]

    data_load, points_num, dim = load_data(data_file)
    index = IndexNSG(dim, points_num, 'L2')

    start_time = time.time()
    paras = Parameters()
    paras.set('L', L)
    paras.set('R', R)
    paras.set('C', C)
    paras.set('nn_graph_path', nn_graph_path)
    index.build(points_num, data_load, paras)
    print(f"Building index with parameters L={L}, R={R}, C={C}, using graph {nn_graph_path}")
    end_time = time.time()

    indexing_time = end_time - start_time
    print(f"indexing time: {indexing_time}")
    index.save(save_graph_file)

if __name__ == "__main__":
    main()


# filename = "./benchmarks/sift.50NN.graph"
# data, num, dim = load_data(filename)
# print("Data loaded:")
# print(data)
# print(f"Number of data points: {num}, Dimension of each point: {dim}")
