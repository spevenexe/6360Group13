import sys
import time
from read_knn import load_data
from indexNSG import IndexNSG
from parameters import Parameters

def main():
    if(len(sys.argv) != 7):
        print(sys.argv[0], " data_file query_file nsg_path search_L search_K result_path")
        exit(-1)
        
    filename = "./vecs/siftsmall/siftsmall_base.fvecs"
    data_load, points_num, dim = load_data(filename)

    filename = "./vecs/siftsmall/siftsmall_query.fvecs"
    query_load, query_num, query_dim = load_data(filename)
    assert dim == query_dim
    
    L = int(sys.argv[4])
    K = int(sys.argv[5])
    if( L < K):
        print("search_L cannot be smaller than search_K")
        exit(-1)
    
    index = IndexNSG(dim,points_num,"L2")
    index.Load(sys.argv[3])

    paras = Parameters()
    paras.set("L_search", L)
    paras.set("P_search", L)

    start = time.process_time()
    res = []
    for i in range(0,query_num):
        tmp = index.search(query_load[i:],data_load,K,paras)
        res.append(tmp)

    end = time.process_time()
    diff = end - start
    print("Search Time: ", diff)
    # for vector in res:
    #     print(vector)
    
def test(data_file, query_file, nsg_path, L, K):

    data_load, points_num, dim = load_data(data_file)
    query_load, query_num, query_dim = load_data(query_file)
    assert dim == query_dim
    
    if(L < K):
        print(L, " ", K)
        print("search_L cannot be smaller than search_K")
        exit(-1)
    
    index = IndexNSG(dim,points_num,"L2")
    index.Load(nsg_path)

    paras = Parameters()
    paras.set("L_search", L)
    paras.set("P_search", L)

    start = time.process_time()
    res = []
    for i in range(0,query_num):
        tmp = index.search(query_load[i:],data_load,K,paras)
        res.append(tmp)

    end = time.process_time()
    diff = end - start
    return diff
    

if __name__ == "__main__":
    main()