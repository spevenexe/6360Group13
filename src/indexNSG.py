import threading
import heapq
import random
import numpy as np
from neighbor import Neighbor
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

    def load(self, filename):
        try:
            self.final_graph = np.load(filename, allow_pickle=True).tolist()
        except FileNotFoundError:
            print(f"Error: The file {filename} does not exist.")
        except IOError:
            print(f"Error: An I/O error occurred while reading {filename}.")
        except:
            print("An unexpected error occurred while loading the graph.")

    def save(self, filename):
        try:
            np.save(filename, np.array(self.final_graph, dtype=object))
        except IOError:
            print(f"Error: An I/O error occurred while saving {filename}.")
        except:
            print("An unexpected error occurred while saving the graph.")


    def load_nn_graph(self, filename):
        with open(filename, 'rb') as f:
            graph_data = np.fromfile(f, dtype=np.uint32)
            k = graph_data[0]
            self.final_graph = graph_data[1:].reshape(-1, k).tolist()

    def get_neighbors(self, query, parameters):
        L = parameters.get("L", 10)
        retset = []
        fullset = []
        if self.initializer:
            init_ids = self.initializer.search(query, L)
        else:
            init_ids = np.random.choice(self.n, L, replace=False)

        for idx in init_ids:
            dist = distance.euclidean(self.data[idx], query)
            neighbor = Neighbor(idx, dist)
            retset.append(neighbor)
            fullset.append(neighbor)

        retset.sort()
        return retset, fullset


    def init_graph(self, parameters):
        center = np.mean(self.data, axis=0)
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


    def search(self, query, k, parameters):
        retset, _ = self.get_neighbors(query, parameters)
        return sorted(retset, key=lambda x: x[1])[:k]