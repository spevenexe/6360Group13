class Neighbor:
    def __init__(self, id, distance, flag=False):
        self.id = id
        self.distance = distance
        self.flag = flag

    def __lt__(self, other):
        return self.distance < other.distance

def InsertIntoPool(addr : Neighbor, K : int, nn : Neighbor):
    raise NotImplementedError