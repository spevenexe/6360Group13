from index import Index
from parameters import Parameters

class indexGraph(Index):
    def __init__(self, dimension : int, n : int, metric, initializer : Index):
        super.__init__(dimension,n,metric)
        assert dimension == initializer.dimension, "unmatched dimensions"
        self.initializer = initializer
    
    def initializeGraph(parameters : Parameters):
        pass
    
    def NNDescent(parameters : Parameters):
        pass
    
    def build(n : int, data : list[float], parameters : Parameters):
        pass