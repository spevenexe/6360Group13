import numpy as np
import struct
import os

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

filename = "./benchmarks/sift.50NN.graph"
data, num, dim = load_data(filename)
print("Data loaded:")
print(data)
print(f"Number of data points: {num}, Dimension of each point: {dim}")
