import matplotlib.pyplot as plt
import sys
import numpy as np
from nsg_search import test

def main():
    if(len(sys.argv) != 6):
        # keepConstant:
        # 0: treat search_LK as the constant L
        # nonzero: treat search_LK as the constant K
        print(sys.argv[0], "base_file query_file nsg_file search_constant setConstantK")
        exit(-1)
    base_file = sys.argv[1]
    query_file = sys.argv[2]
    nsg_file = sys.argv[3]
    search_constant = int(sys.argv[4])
    setConstantK = int(sys.argv[5])
    
    x = []
    searchTime = []
    if(setConstantK == 0):
        x = [k for k in range(1,search_constant+1)]
        searchTime = [None]*len(x)
        for k in range(0,len(x)):
            searchTime[k] = test(base_file,query_file,nsg_file,search_constant,k)
    else:
        x = [l for l in range(search_constant,search_constant+20)]
        searchTime = [None]*len(x)
        for l in range(0,len(x)):
            searchTime[l] = test(base_file,query_file,nsg_file,l+search_constant,search_constant)
    
    # x = x[1:]
    # searchTime = searchTime[1:]
    x = np.array(x)
    searchTime = np.array(searchTime)
    plt.scatter(x,searchTime,color="b")
    plt.show()
    
if __name__ == "__main__":
    main()