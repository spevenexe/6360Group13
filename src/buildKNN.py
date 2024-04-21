from index import Index, distance_l2
from indexGraph import indexGraph
from indexRandom import indexRandom
from parameters import Parameters
import sys
import time

def load_data(argv : list[str]):
    
    return 1,2,3

# build a kNN and write it to save_graph
def buildKNN():
    if len(sys.argv) != 8:
        print(sys.argv[0], "data_file save_graph K L iter S R")
        exit(-1)
    data_load, points_num, dim = load_data(sys.argv[1])
    
    graph_filename,K,L,iter,S,R = [sys.argv[i] for i in range(2,8)]
    init_index = indexRandom(dim,points_num)
    index = indexGraph(dim,points_num, distance_l2,init_index)
    
    paras = Parameters()
    paras.set("K",K)
    paras.set("L",L)
    paras.set("iter",iter)
    paras.set("S",S)
    paras.set("R",R)
    
    start = time.process_time_ns()
    
    end = time.process_time_ns()
    
    return

def main():
    buildKNN()
    return

if __name__ == "__main__":
    main()