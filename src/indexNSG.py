import threading
import heapq
import random
import struct
import numpy as np
from parameters import Parameters
from neighbor import Neighbor, insert_into_pool
from scipy.spatial import distance
from index import distance_l2,distance_inner_product

class IndexNSG:
    def __init__(self, dimension, n, metric):
        self.dimension = dimension
        self.n = n
        if callable(metric):
            self.distance = metric
        elif metric == "L2":
            self.distance = distance_l2
        elif metric == "inner_product":
            self.distance = distance_inner_product
        else:
            raise ValueError("Unsupported metric type")
        self.data = None
        self.final_graph = [[] for _ in range(n)]
        self.locks = [threading.Lock() for _ in range(n)]
        self.has_built = False
        self.ep = 0

    #test_nsg_search invokes index_nsg : load
    def Load(self, filename):
        with open(filename, 'rb') as f:
            self.width = struct.unpack('I', f.read(4))[0]
            self.ep = struct.unpack('I', f.read(4))[0]
            # cc = 0
            while True:
                k_bytes = f.read(4)
                if not k_bytes:
                    break
                k = struct.unpack('I', k_bytes)[0]

                tmp_bytes = f.read(k * 4)
                tmp = struct.unpack(f'{k}I', tmp_bytes)
                # cc += tmp

                self.final_graph.append(list(tmp))
        return self.final_graph

    # cc /= self.nd_

    def save(self, filename):
        assert len(self.final_graph) == self.n, "Final graph size does not match the expected size"
        with open(filename, 'wb') as f:
            f.write(struct.pack('I', self.width))
            f.write(struct.pack('I', self.ep))
            for neighbors in self.final_graph:
                GK = len(neighbors)
                f.write(struct.pack('I', GK))
                if GK > 0:
                    f.write(struct.pack(f'{GK}I', *neighbors))

    # tested and correct
    def load_nn_graph(self, filename):
        with open(filename, 'rb') as f:
            graph_data = np.fromfile(f, dtype=np.uint32)
            k = graph_data[0]
            self.final_graph = graph_data.reshape(-1, k+1)[:,1:].tolist()

    def get_neighbors(self, query, parameters):
        L = parameters.get("L", 10)
        retset = []
        fullset = []
        flags = np.zeros(self.nd, dtype=bool)

        # Initialize neighbors
        init_ids = list(self.final_graph[self.ep][:L])
        flags[init_ids] = True
        additional_ids = np.random.choice([i for i in range(self.nd) if not flags[i]], size=L-len(init_ids), replace=False)
        init_ids.extend(additional_ids)
        flags[additional_ids] = True

        for id in init_ids:
            if id < self.nd:
                dist = distance.euclidean(self.data[id], query)
                retset.append(Neighbor(id, dist, True))

        retset.sort()

        k = 0
        while k < len(retset):
            nk = len(retset)
            if retset[k].flag:
                current_id = retset[k].id
                retset[k].flag = False
                neighbors = self.final_graph[current_id]
                for neighbor_id in neighbors:
                    if flags[neighbor_id]:
                        continue
                    flags[neighbor_id] = True
                    dist = distance.euclidean(self.data[neighbor_id], query)
                    new_neighbor = Neighbor(neighbor_id, dist, True)
                    fullset.append(new_neighbor)
                    if dist < retset[-1].distance:
                        r = insert_into_pool(retset, L, new_neighbor)
                        if r < nk: nk = r
                        if len(retset) > L: retset.pop()
            if nk < k: k = nk
            else: k+=1

        return retset, fullset


    def init_graph(self, parameters):
        center = np.mean(self.data, axis=0)
        self.ep = random.randint(0, self.n - 1)
        retset, _ = self.get_neighbors(center, parameters)
        self.ep = retset[0][0]

    def sync_prune(self, q, pool, parameters, flags, cut_graph):
        R = parameters.get("R", 100)
        C = parameters.get("C", 100)
        sorted_pool = sorted(pool)[:R]

        with self.locks[q]:
            for neighbor in sorted_pool:
                if not flags[neighbor.id]:
                    cut_graph[q].append(neighbor)
                    flags[neighbor.id] = True
                    if len(cut_graph[q]) >= C:
                        break

    def build(self, data, parameters):

        if data is None or not data.shape[0]:
            raise ValueError("Data is empty or None")
        if data.ndim != 2:
            raise ValueError("Data must be a 2D array")
    
        self.data = data
        n, dimension = data.shape
        l = parameters.get("l", 40)
        m = parameters.get("m", 50)

        centroid = np.mean(self.data, axis=0)
        dists_to_centroid = np.linalg.norm(self.data - centroid, axis=1)
        navigating_node = np.argmin(dists_to_centroid)

        for i in range(n):

            candidate_pool = self.search_on_graph(i, l)
            candidate_pool.sort(key=lambda x: x.distance)
            
            if candidate_pool:
                nearest_neighbor = candidate_pool[0]
                self.final_graph[i].append(nearest_neighbor)

                for neighbor in candidate_pool[1:]:
                    if self.no_conflict(i, neighbor, m):
                        self.final_graph[i].append(neighbor)
                        if len(self.final_graph[i]) >= m:
                            break

        self.build_dfs_tree(navigating_node)
        self.has_built = True


    def search_on_graph(self, G, q, l, k):
        S = []
        checked = set()
        distance_to_query = {}

        start_node = random.choice(range(self.n))
        start_distance = self.distance.compare(self.data[start_node], q, self.dimension)
        heapq.heappush(S, (start_distance, start_node))
        distance_to_query[start_node] = start_distance

        while len(checked) < l and S:
            current_distance, current_node = heapq.heappop(S)
            checked.add(current_node)

            for neighbor in G[current_node]:
                if neighbor not in checked:
                    if neighbor not in distance_to_query:
                        neighbor_distance = self.distance.compare(self.data[neighbor], q, self.dimension)
                        distance_to_query[neighbor] = neighbor_distance
                    else:
                        neighbor_distance = distance_to_query[neighbor]

                    heapq.heappush(S, (neighbor_distance, neighbor))

            if len(checked) >= l:
                break

        top_k_neighbors = sorted(S, key=lambda x: x[0])[:k]

        return [node for _, node in top_k_neighbors]


    def no_conflict(self, node, candidate, m):
        if len(self.final_graph[node]) >= m:
            return False

        for existing_neighbor in self.final_graph[node]:
            if existing_neighbor.id == candidate.id:
                return False

        max_distance = max(self.final_graph[node], key=lambda n: n.distance).distance
        if candidate.distance >= max_distance:
            return False

        return True

    def build_dfs_tree(self, root):
        n = len(self.final_graph)
        visited = [False] * n
        stack = [root]

        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                for neighbor in self.final_graph[node]:
                    neighbor_id = neighbor.id
                    if not visited[neighbor_id]:
                        stack.append(neighbor_id)
                        if not self.is_connected(node, neighbor_id):
                            self.connect_nodes(node, neighbor_id)


    def is_connected(self, node1, node2):
        return any(neighbor.id == node2 for neighbor in self.final_graph[node1])


    def connect_nodes(self, node1, node2):
        if not self.is_connected(node1, node2):
            distance = self.distance.compare(self.data[node1], self.data[node2], self.dimension)
            self.final_graph[node1].append(Neighbor(node2, distance))
            self.final_graph[node2].append(Neighbor(node1, distance))

    # note: the function now returns indices
    # instead of receiving it as an arg and modifying directly
    # fuck c++ coding conventions
    # still need to check if this even works
    def search(self, query : list[float], x : list[float], K : list[float], parameters : Parameters):
        print(query)
        indices = np.empty(K,int)    
        # print("L: ",parameters.get("L_search"))
        L = int(parameters.get("L_search"))
        data = x
        retset = np.empty(L+1,Neighbor)
        init_ids = np.zeros(L,int)
        flags = np.empty(self.n,bool)
        
        tmp_l = 0
        while tmp_l < L and tmp_l < len(self.final_graph[self.ep]):
            init_ids[tmp_l] = self.final_graph[self.ep][tmp_l]
            flags[init_ids[tmp_l]] = True
            tmp_l +=1
        
        while tmp_l < L:
            id = random.randint(0,self.n-1)
            if flags[id]: continue
            flags[id] = True
            init_ids[tmp_l] = id
            tmp_l+=1
        
        # print("finished random")
        
        for i in range(0,len(init_ids)):
            id = init_ids[i]
            dist = self.distance(data[id],query)
            retset[i] = Neighbor(id, dist, True)
        
        retset[0:L] = sorted(retset[0:L])
        # print(retset)
        k = 0
        while k < L:
            nk = L
            
            if retset[k].flag:
                retset[k].flag = False
                n = retset[k].id
                
                for m in range(0,len(self.final_graph[n])):
                    id = self.final_graph[n][m]
                    if flags[id]: continue
                    flags[id] = True
                    dist = self.distance(query,data[id])
                    if dist >= retset[L-1].distance: continue
                    nn = Neighbor(id,dist,True)
                    # TODO: make InsertIntoPool
                    r = insert_into_pool(retset,L,nn)
                    
                    if r < nk: 
                        nk = r
            if nk <= k:
                k = nk
            else:
                k+=1    
        for i in range(0,K):
            indices[i] = retset[i].id      
        
        return indices
        
        # retset, _ = self.get_neighbors(query, parameters)
        # return sorted(retset, key=lambda x: x[1])[:k]

    def DFS(self, root, flag, cnt):
        tmp = root
        stack = [root]
        if not flag[root]:
            cnt[0] += 1
        flag[root] = True
        while stack:
            next_node = None
            for neighbor in self.final_graph[tmp]:
                if not flag[neighbor]:
                    next_node = neighbor
                    break
            if next_node is None:
                stack.pop()
                if not stack:
                    break
                tmp = stack[-1]
                continue
            tmp = next_node
            flag[tmp] = True
            stack.append(tmp)
            cnt[0] += 1
        return flag, cnt

    def findroot(self, flag, root, parameter):
        id = self.n
        for i in range(self.n):
            if not flag[i]:
                id = i
                break

        if id == self.n:
            return  # No Unlinked Node

        tmp, pool = self.get_neighbors(self.data + self.dimension * id, parameter)
        pool.sort()

        found = False
        for neighbor in pool:
            if flag[neighbor.id]:
                root[0] = neighbor.id
                found = True
                break

        if not found:
            while True:
                rid = random.randint(0, self.n - 1)
                if flag[rid]:
                    root[0] = rid
                    break

        self.final_graph[root[0]].append(id)
    

    def tree_grow(self, parameter):
        root = self.ep
        flags = [False] * self.n
        unlinked_cnt = 0
        while unlinked_cnt < self.n:
            flags, unlinked_cnt = self.DFS(root, flags, unlinked_cnt)
            if unlinked_cnt >= self.n:
                break
            self.findroot(flags, root, parameter)

        for i in range(self.n):
            if len(self.final_graph[i]) > width:
                width = len(self.final_graph[i])