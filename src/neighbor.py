class Neighbor:
    def __init__(self, id=0, distance=0, flag=False):
        self.id = id
        self.distance = distance
        self.flag = flag

    def __lt__(self, other):
        return self.distance < other.distance

def insert_into_pool(addr, K, nn):
    # Find the location to insert
    left, right = 0, K - 1
    if addr[left].distance > nn.distance:
        addr[left + 1 : left + K + 1] = addr[left : left + K]
        addr[left] = nn
        return left

    if addr[right].distance < nn.distance:
        addr[K] = nn
        return K

    while left < right - 1:
        mid = (left + right) // 2
        if addr[mid].distance > nn.distance:
            right = mid
        else:
            left = mid

    # Check equal ID
    while left > 0:
        if addr[left].distance < nn.distance:
            break
        if addr[left].id == nn.id:
            return K + 1
        left -= 1

    if addr[left].id == nn.id or addr[right].id == nn.id:
        return K + 1

    addr[right + 1 : K + 1] = addr[right : K]
    addr[right] = nn
    return right