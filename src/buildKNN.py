from parameters import Parameters
import sys

def buildKNN():
    if len(sys.argv) != 7:
        print(sys.argv[0], "data_file save_graph K L iter S R")
        exit(-1)
    return