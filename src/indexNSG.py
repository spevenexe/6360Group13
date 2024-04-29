import threading
import heapq
import itertools
import random
import struct
import numpy as np
from concurrent.futures import ThreadPoolExecutor
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
        flags = np.zeros(self.n, dtype=bool)

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


    def sync_prune(self, q, pool, parameter, flags, cut_graph):
        range_r = parameter['R']
        maxc = parameter['C']
        start = 0

        # Collect eligible neighbors
        for nn in self.final_graph[q]:
            if flags[nn]:
                continue
            dist = distance.euclidean(self.data[q], self.data[nn])
            pool.append(Neighbor(nn, dist))

        # Sort neighbors by distance
        pool.sort()

        # Start could be incremented if the closest neighbor is itself (q)
        if len(pool) > 0 and pool[start].id == q: start += 1
        if pool: result = [pool[start]]
        else: result = []

        # Prune the pool to meet the criteria
        while len(result) < range_r and start < len(pool) and start < maxc:
            p = pool[start]
            occlude = False
            for t in result:
                if p.id == t.id:
                    occlude = True
                    break
                # Check if another neighbor is closer to t than p is to q
                djk = distance.euclidean(self.data[t.id], self.data[p.id])
                if djk < p.distance:
                    occlude = True
                    break
            if not occlude:
                result.append(p)
            start += 1

        # Update the cut graph with the pruned results
        for t, neighbor in enumerate(result):
            cut_graph[q * range_r + t] = Neighbor(id=neighbor.id, distance=neighbor.distance)

        # Mark unused slots with a special flag (-1)
        if len(result) < range_r:
            cut_graph[q * range_r + len(result)].distance = -1

    # finish this
    def inter_insert(self, n, range_r, locks, cut_graph):

        for i in range(range_r):
            if cut_graph[n*range_r+i].distance == -1: break

            sn = Neighbor(n, cut_graph[n*range_r+i].distance)
            des = cut_graph[n*range_r+i].id

            temp_pool = []
            dup = False
            lock = locks[des]
            with lock:
                for j in range(range_r):
                    if cut_graph[des*range_r+j].distance == -1: break
                    if n == cut_graph[des*range_r+j].id: dup = True; break
                    temp_pool.append(cut_graph[des*range_r+j])

            if dup: continue

            temp_pool.append(sn)
            if len(temp_pool) > range_r:
                start = 0
                temp_pool.sort()
                result = [temp_pool[start]]
                while len(result) < range_r and start + 1 < len(temp_pool):
                    start += 1
                    p = temp_pool[start]
                    occlude = False
                    for t in result:
                        if p.id == t.id:
                            occlude = True
                            break
                        djk = self.euclidean_distance(t.id, p.id)
                        if djk < p.distance:
                            occlude = True
                            break
                    if not occlude:
                        result.append(p)

                with lock:
                    for i,t in enumerate(result):
                        cut_graph[des*range_r+i] = t
            else:
                with lock:
                    for t in range(range_r):
                        if cut_graph[t+des*range_r].distance == -1:
                            cut_graph[t+des*range_r] = sn
                            if t + 1 < range_r:
                                cut_graph[t + 1 + des*range_r].distance = -1
                            break


    def link(self, parameters, cut_graph):
        range_r = parameters['R']
        step_size = self.nd // 100
        locks = [threading.Lock() for _ in range(self.nd)]

        def process_node(n):
            point = self.data[n * self.dimension:(n + 1) * self.dimension]
            pool, tmp = [], []
            flags = np.zeros(self.nd, dtype=bool)

            self.get_neighbors(point, parameters, flags, tmp, pool)
            self.sync_prune(n, pool, parameters, flags, cut_graph)

        # First parallel execution block
        with ThreadPoolExecutor(max_workers=step_size) as executor:
            list(executor.map(process_node, range(self.nd)))

        # InterInsert now includes locks as a parameter
        def call_interinsert(n):
            self.inter_insert(n, range_r, locks, cut_graph)

        # Second parallel execution block
        with ThreadPoolExecutor(max_workers=step_size) as executor:
            list(executor.map(call_interinsert, range(self.nd)))



    def build(self, n, data, parameters):
        nn_graph_path = parameters['nn_graph_path']
        range_r = parameters['R']
        self.load_nn_graph(nn_graph_path)
        self.data = data
        self.init_graph(parameters)
        cut_graph = [Neighbor() for _ in range(n * range_r)]
        self.link(parameters, cut_graph)
        self.final_graph = [[] for _ in range(n)]

        for i in range(n):
            pool = cut_graph[i * range_r:(i + 1) * range_r]
            pool_size = 0
            for neighbor in pool:
                if neighbor.distance == -1:
                    break
                pool_size += 1

            self.final_graph[i] = [neighbor.id for neighbor in pool[:pool_size]]

        self.tree_grow(parameters)

        max_degree = max(len(g) for g in self.final_graph)
        min_degree = min(len(g) for g in self.final_graph)
        avg_degree = sum(len(g) for g in self.final_graph) / n
        print(f"Degree Statistics: Max = {max_degree}, Min = {min_degree}, Avg = {avg_degree:.2f}")

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
        indices = np.empty(K,int)    
        # print("L: ",parameters.get("L_search"))
        L = int(parameters.get("L_search"))
        data = x
        returnSet = np.empty(L+1,Neighbor)
        init_ids = np.zeros(L,int)
        flags = np.empty(self.n,bool)
        for b in flags:
            flags[b]=False
        # print(flags)
        
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
            returnSet[i] = Neighbor(id, dist, True)
        
        returnSet[0:L] = sorted(returnSet[0:L])
        # print(retset)
        k = 0
        while k < L:
            next_k = L
            
            if returnSet[k].flag:
                returnSet[k].flag = False
                n = returnSet[k].id
                
                for m in range(0,len(self.final_graph[n])):
                    id = self.final_graph[n][m]
                    if flags[id]: continue
                    flags[id] = True
                    dist = self.distance(query,data[id])
                    if dist >= returnSet[L-1].distance: continue
                    nn = Neighbor(id,dist,True)
                    # TODO: make InsertIntoPool
                    right = insert_into_pool(returnSet,L,nn)
                    
                    if right < next_k: 
                        next_k = right
            if next_k <= k:
                k = next_k
            else:
                k+=1    
        for i in range(0,K):
            indices[i] = returnSet[i].id      
        
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