# # import sys
# import os

# original_path = sys.path.copy()
# sys.path.append(os.path.abspath('../src'))
import indexNSG
from driver import load_data
# sys.path = original_path
graph = indexNSG.IndexNSG(128, 10000, "L2")

filename = "./benchmarks/sift.50NN.graph"
graph.load_nn_graph(filename)

filename = "./benchmarks/sift.50NN.graph"
data, num, dim = load_data(filename)
print("Data loaded:")
# print(data)
print(f"Number of data points: {num}, Dimension of each point: {dim}")
print(data == graph.final_graph)

# print(graph.final_graph)