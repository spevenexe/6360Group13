from index import Index

class indexRandom(Index):
    def __init__(self, dimension : int, n : int):
        super.__init__(dimension,n,"L2")
        self.has_built = True
        