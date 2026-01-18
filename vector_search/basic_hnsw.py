"""
Basic implementation of Approximate nearest neighbors (ANN)

Source 1: https://weaviate.io/blog/vector-search-explained

Note that most vector databases allow you to configure how your ANN algorithm should behave.
This lets you find the right balance between the
recall tradeoff (the fraction of results that are the true top-k nearest neighbors),
latency, throughput (queries per second) and import time.

Common ANN methods:
1. Tree based - ANNOY
2. proximity graphs - HNSW
3. Clustering - FAISS
4. Hashing - LSH

For each of the above algos, we need to strike a balance amongst - 
1. Latency - time it takes to process one query 
2. Throughput - Queries per second
3. Build time - time it takes to build the vector index
4. Recall - #  of items relevant items retrieved out of all the relevant items present in the dataset

Weaviate maintains a recall of >95%
"""

import numpy as np
from typing import List, Dict, Tuple
from loguru import logger
import visualise_hnsw as vis_utils
import heapq

np.random.seed(42)


class Node:
    """The Node class representing the vector"""
    def __init__(self, vector, node_id, max_layer):
        self.id = node_id                             # Unique identifier
        self.vector = vector                  # the actual embedding
        self.max_layer = max_layer                     # Max level this node exists in
        self.neighbors: Dict[int, List[int]] = {layer: [] for layer in range(self.max_layer + 1)}            # Dictionary containing neighbors of this node at every layer where this node is present in


class HNSW:
    """The main HNSFW class"""
    def __init__(self, M=16, M_max=16, ef_construct=200, ml=1.0) -> None:
        self.M = M          # Max connections per node
        self.M_max = M_max          # max connections at layer 0 (the lowest layer)
        self.ef_construction = ef_construct    # Beam search width
        self.ml = ml                            # Layer normalisation factor


        self.nodes: Dict[int, Node] = {}         # node_id -> Node
        self.entrypoint = None      # the highest node (starting point)
        self.max_layer = -1             # Current max layer in graph

        self.distance = self.l2

    def l2(self, a, b):
        return np.linalg.norm(a-b)  # (a-b)^2
    
    def assign_layer(self, p=0.5, max_level=32):
        # Formula: floor(-ln(uniform(0,1)) * mL)
        return int(-np.log(np.random.uniform(0, 1)) * self.ml)
    
    def search_layer(self, query: int, entrypoints: List[int], layer: int, ef: int=1) -> List[Tuple[int, int]]:
        """
        Search for nearest neighbors at a specific layer
        
        :param query: The query vector
        :param entrypoints: List of node_ids to start search from
        :param layer: which layer to search in
        :param ef: # of candidates to track (1 for greedy, >1 for beam search)

        Returns:
            List of (node_id, distance) tuples - the ef nearest neigbours
        """

        visited = set()
        candidates = []     # Min-heap: (distance, node_id)     # closest node always at the top
        nearest = []        # Max-heap: (-distance, node_id)    ???


        # Computing distance of the query node with all the candidate nodes that will serve as entrypoints for the next layer
        for ep_id in entrypoints:
            dist = self.distance(query, self.nodes[ep_id].vector)
            heapq.heappush(candidates, (dist, ep_id))
            heapq.heappush(nearest, (-dist, ep_id))
            visited.add(ep_id)

        while candidates:
            # get closest unvisited candidate
            current_dist, current_id = heapq.heappop(candidates)

            # If this is worse than our ef-th best, stop
            if current_dist > -nearest[0][0]:
                break

            # If the current distance is better than the ef-th best, then we can explore neigbours
            for neigbour_id in self.nodes[current_id].neighbors[layer]:
                if neigbour_id not in visited:
                    visited.add(neigbour_id)
                
                neigbour_dist = self.distance(query, self.nodes[neigbour_id].vector)

                # We add this neigbour to the list of candidates if it's better than the current worst or we add it anyways if we have room in the heap
                if neigbour_dist < -nearest[0][0] or len(nearest) < ef:
                    heapq.heappush(candidates, (neigbour_dist, neigbour_id))
                    heapq.heappush(nearest, (-neigbour_dist, neigbour_id))

                    # Keep only top ef
                    if len(nearest) > ef:
                        heapq.heappop(nearest)

        
        # Isn't this wrong? nearest contains distance with a negative sign
        # sorted(nearest) will return distances in ascending order
        # So output will be [-5, -4, -2, -1]
        # And when we iterate over the sorted list and put a -ve sign again in the distance
        # The output list that will be returned is [5, 4, 2, 1]
        return [(node_id, -dist) for dist, node_id in sorted(nearest)]
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __repr__(self) -> str:
        return (f"HNSW(nodes={len(self.nodes)}, layers={self.max_layer + 1}, "
                f"M={self.M}, ef_construction={self.ef_construction})")

        

    #####################################
    #### Graph building utils ###########
    #####################################

    def find_neighbors(self, candidates, M):
        """
        Select M best and diverse neighbors
        
        :param candidates: the candidate nodes that might be marked as neighbors for the query node
        :param M: Maximum # of neighbors this query node can have
        """

        candidates = sorted(candidates, key=lambda x: x[1])

        selected = []

        for candidate_id, candidate_dist in candidates:
            if len(selected) >= M:
                break

            should_add = True

            for selected_id in selected:
                # if candidate_id is closer to selected that to query, then skip
                dist_to_selected = self.distance(self.nodes[candidate_id].vector, self.nodes[selected_id].vector)

                if dist_to_selected < candidate_dist:
                    should_add = False
                    break
            
            if should_add:
                selected.append(candidate_id)

        # If we still need more (heuristic was too strict), add closest remaining
        # if len(selected) < M:
        #     for candidate_id, _ in candidates:
        #         if candidate_id not in selected:
        #             selected.append(candidate_id)
        #             if len(selected) >= M:
        #                 break

        return selected

    def insert(self, vector, node_id=None):
        """
        Insert a new vector (node) into the HNSW graph
        
        :param vector: actual vector embedding
        :param node_id: node_id
        """
        if node_id is None:
            node_id = len(self.nodes)   # Why are we assigning the last index as the ID for the first node. Can't we do node_id = 0?
        
        # Step 1: Assign random layer
        node_layer = self.assign_layer()
        logger.info(f"Node: {node_id} assigned layer: {node_layer}")

        # Create the node
        node = Node(vector=vector, node_id=node_id, max_layer=node_layer)
        logger.info(f"Node {node.id} created." )
        assert node.id == node_id, "Node ID mismatch."
        self.nodes[node_id] = node


        # If this is the first node
        if self.entrypoint is None:
            self.entrypoint = node_id   # We only have one node so this will be our entrypoint for the current max layer
            self.max_layer = node_layer
            return node_id
        
        # If the current node is taller than the entrypoint
        if node_layer > self.max_layer:
            self.max_layer = node_layer
            self.entrypoint = node_id

        # Step 2: Search for insertion point
        current_nearest = [self.entrypoint]
        for layer in range(self.max_layer, node_layer, -1):
            # Greedy search (ef=1) to find entrypoint for next layer
            current_nearest = self.search_layer(vector, current_nearest, layer, ef=1)
            current_nearest = [current_nearest[0][0]]       # Why this step is required? self.search_layer (ef = 1) will anyways return only 1 entrypoint to the next layer

        # Step 3: Now we have reached the max_layer for this node
        # So from layer = max_layer -> layer = 0, we have to insert current node (node_id) at each layer
        # And insertion means that we also need to connect it with some other "relevant" nodes

        for layer in range(node_layer, -1, -1):
            # Beam search to find candidates
            candidates = self.search_layer(vector, current_nearest, layer, ef=self.ef_construction)

            # Filter out the current node from the candidate list
            candidates = [(cid, dist) for cid, dist in candidates if cid != node_id]

            # Select M neighbors
            M = self.M if layer > 0 else self.M_max
            neighbors = self.find_neighbors(candidates=candidates, M=M)

            # Add bidirectional connections
            for neighbor_id in neighbors:
                # Connect new node (node_id) to neighbor_id
                node.neighbors[layer].append(neighbor_id)

                # Connect neighbor_id to new node (node_id)
                self.nodes[neighbor_id].neighbors[layer].append(node_id)

                # Prune logic will come later

            
            # Update current_nearest for next layer
            current_nearest = neighbors

        # Update entry point if this node is taller
        if node_layer > self.max_layer:
            self.max_layer = node_layer
            self.entrypoint = node_id

        return node_id
    

if __name__ == "__main__":
    index = HNSW(M=4, M_max=8, ef_construct=4)
    logger.info("Created index with params:\n")

    # Insert vectors
    vectors = np.random.randn(10000, 32)
    for i, vec in enumerate(vectors):
        index.insert(vector=vec, node_id=i)
        logger.info(f"Vector {i} inserted successfully.")

    print("Stop")

    # Query
    # query = np.random.randn(1, 32)
    # results = index.search_knn(query, k=10, ef=4)    
    
