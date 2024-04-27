import matplotlib.pyplot as plt
import sys
import numpy as np
from nsg_serach import test

def main():
    if(len(sys.argv) != 6):
        # keepConstant:
        # 0: treat search_LK as the constant L
        # nonzero: treat search_LK as the constant K
        print(sys.argv[0], "data_file query_file nsg_path search_LK keepConstant")
        exit(-1)
    data_file = sys.argv[1]
    query_file = sys.argv[2]
    nsg_path = sys.argv[3]
    search_LK = int(sys.argv[4])
    keepConstant = int(sys.argv[5])
    
    x = []
    searchTime = []
    if(keepConstant == 0):
        x = [k for k in range(1,search_LK+1)]
        searchTime = [None]*len(x)
        for k in range(0,len(x)):
            searchTime[k] = test(data_file,query_file,nsg_path,search_LK,k)
    else:
        x = [l for l in range(search_LK,search_LK+20)]
        searchTime = [None]*len(x)
        for l in range(0,len(x)):
            searchTime[l] = test(data_file,query_file,nsg_path,l+search_LK,search_LK)   
    x = np.array(x)
    searchTime = np.array(searchTime)
    plt.scatter(x,searchTime,color="b")
    plt.show()
    
if __name__ == "__main__":
    main()