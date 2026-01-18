"""
hnsw_utils.py - Utility functions for HNSW visualization and debugging
"""

import numpy as np


def print_graph_structure(hnsw_index):
    """
    Print a detailed view of the current HNSW graph structure
    Shows layers, nodes, and their connections
    
    Args:
        hnsw_index: HNSW instance
    """
    
    if hnsw_index.entrypoint is None:
        print("Graph is empty!")
        return
    
    print("=" * 80)
    print("HNSW GRAPH STRUCTURE")
    print("=" * 80)
    print(f"Total nodes: {len(hnsw_index.nodes)}")
    print(f"Max layer: {hnsw_index.max_layer}")
    print(f"Entry point: Node {hnsw_index.entrypoint}")
    print(f"Parameters: M={hnsw_index.M}, M_max={hnsw_index.M_max}, ef_construction={hnsw_index.ef_construction}")
    print("=" * 80)
    
    # Organize nodes by layer
    layers = {}
    for node_id, node in hnsw_index.nodes.items():
        for layer in range(node.max_layer + 1):
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node_id)
    
    # Print layer by layer (top to bottom)
    for layer in range(hnsw_index.max_layer, -1, -1):
        print(f"\n{'─' * 80}")
        print(f"LAYER {layer}")
        print(f"{'─' * 80}")
        
        if layer not in layers:
            print("  (empty)")
            continue
        
        nodes_at_layer = sorted(layers[layer])
        print(f"  Nodes: {nodes_at_layer}")
        print(f"  Total: {len(nodes_at_layer)} nodes")
        print()
        
        # Print connections for each node at this layer
        for node_id in nodes_at_layer:
            node = hnsw_index.nodes[node_id]
            neighbors = node.neighbors[layer]
            
            # Calculate some statistics
            if neighbors:
                neighbor_distances = [
                    hnsw_index.distance(node.vector, hnsw_index.nodes[n].vector)
                    for n in neighbors
                ]
                avg_dist = np.mean(neighbor_distances)
                min_dist = np.min(neighbor_distances)
                max_dist = np.max(neighbor_distances)
            else:
                avg_dist = min_dist = max_dist = 0
            
            # Print node info
            print(f"  Node {node_id}:")
            print(f"    Max layer: {node.max_layer}")
            print(f"    Neighbors ({len(neighbors)}): {neighbors}")
            if neighbors:
                print(f"    Distances: avg={avg_dist:.3f}, min={min_dist:.3f}, max={max_dist:.3f}")
            print()
    
    print("=" * 80)


def print_graph_summary(hnsw_index):
    """
    Print a compact summary of the graph structure
    Useful for quick checks
    
    Args:
        hnsw_index: HNSW instance
    """
    
    if hnsw_index.entrypoint is None:
        print("Graph is empty!")
        return
    
    print("\nHNSW Graph Summary:")
    print(f"  Nodes: {len(hnsw_index.nodes)}")
    print(f"  Layers: 0 to {hnsw_index.max_layer}")
    print(f"  Entry: Node {hnsw_index.entrypoint}")
    
    # Count nodes per layer
    for layer in range(hnsw_index.max_layer, -1, -1):
        count = sum(1 for node in hnsw_index.nodes.values() if node.max_layer >= layer)
        print(f"  Layer {layer}: {count} nodes")


def print_graph_ascii(hnsw_index):
    """
    Print a visual ASCII representation of the graph
    Shows the hierarchical structure
    
    Args:
        hnsw_index: HNSW instance
    """
    
    if hnsw_index.entrypoint is None:
        print("Graph is empty!")
        return
    
    print("\nHNSW Graph Visualization (ASCII):")
    print("=" * 80)
    
    # Organize nodes by their max layer
    layer_nodes = {}
    for node_id, node in hnsw_index.nodes.items():
        if node.max_layer not in layer_nodes:
            layer_nodes[node.max_layer] = []
        layer_nodes[node.max_layer].append(node_id)
    
    # Print from top to bottom
    for layer in range(hnsw_index.max_layer, -1, -1):
        # Nodes that exist at this layer
        nodes_at_layer = [
            nid for nid, node in hnsw_index.nodes.items() 
            if node.max_layer >= layer
        ]
        
        print(f"Layer {layer}: ", end="")
        
        # Show nodes
        for nid in sorted(nodes_at_layer):
            node = hnsw_index.nodes[nid]
            
            # Mark entry point
            if nid == hnsw_index.entrypoint and layer == hnsw_index.max_layer:
                print(f"[{nid}*]", end=" ")
            # Mark nodes that "start" at this layer (max_layer)
            elif node.max_layer == layer:
                print(f"[{nid}]", end=" ")
            # Nodes that extend through this layer
            else:
                print(f" {nid} ", end=" ")
        
        print()
        
        # Show connections at this layer
        if layer > 0:
            print("         ", end="")
            for nid in sorted(nodes_at_layer):
                neighbors = hnsw_index.nodes[nid].neighbors[layer]
                if neighbors:
                    print(f" ↓{len(neighbors)} ", end=" ")
                else:
                    print("    ", end=" ")
            print()
    
    print("=" * 80)
    print("Legend: [N*] = entry point, [N] = node starts here, N = extends through")


def print_node_details(hnsw_index, node_id):
    """
    Print detailed information about a specific node
    
    Args:
        hnsw_index: HNSW instance
        node_id: The node to inspect
    """
    
    if node_id not in hnsw_index.nodes:
        print(f"Node {node_id} not found!")
        return
    
    node = hnsw_index.nodes[node_id]
    
    print("=" * 80)
    print(f"NODE {node_id} DETAILS")
    print("=" * 80)
    print(f"Vector shape: {node.vector.shape}")
    print(f"Max layer: {node.max_layer}")
    print(f"Is entry point: {node_id == hnsw_index.entrypoint}")
    print()
    
    # Print neighbors at each layer
    for layer in range(node.max_layer, -1, -1):
        neighbors = node.neighbors[layer]
        print(f"Layer {layer}:")
        print(f"  Neighbors ({len(neighbors)}): {neighbors}")
        
        if neighbors:
            # Show distances to neighbors
            print("  Distances:")
            for neighbor_id in neighbors:
                dist = hnsw_index.distance(node.vector, hnsw_index.nodes[neighbor_id].vector)
                print(f"    → Node {neighbor_id}: {dist:.4f}")
        print()
    
    print("=" * 80)


def print_connectivity_stats(hnsw_index):
    """
    Print statistics about graph connectivity
    
    Args:
        hnsw_index: HNSW instance
    """
    
    if not hnsw_index.nodes:
        print("Graph is empty!")
        return
    
    print("\nConnectivity Statistics:")
    print("=" * 80)
    
    for layer in range(hnsw_index.max_layer, -1, -1):
        # Nodes at this layer
        nodes_at_layer = [
            nid for nid, node in hnsw_index.nodes.items()
            if node.max_layer >= layer
        ]
        
        if not nodes_at_layer:
            continue
        
        # Connection stats
        connection_counts = [
            len(hnsw_index.nodes[nid].neighbors[layer])
            for nid in nodes_at_layer
        ]
        
        avg_connections = np.mean(connection_counts)
        min_connections = np.min(connection_counts)
        max_connections = np.max(connection_counts)
        
        # Distance stats
        all_distances = []
        for nid in nodes_at_layer:
            node = hnsw_index.nodes[nid]
            for neighbor_id in node.neighbors[layer]:
                dist = hnsw_index.distance(node.vector, hnsw_index.nodes[neighbor_id].vector)
                all_distances.append(dist)
        
        print(f"\nLayer {layer}:")
        print(f"  Nodes: {len(nodes_at_layer)}")
        print(f"  Connections per node: avg={avg_connections:.2f}, min={min_connections}, max={max_connections}")
        
        if all_distances:
            print(f"  Edge distances: avg={np.mean(all_distances):.4f}, "
                  f"min={np.min(all_distances):.4f}, max={np.max(all_distances):.4f}")
    
    print("=" * 80)


def to_networkx(hnsw_index, layer=0):
    """
    Export a specific layer to NetworkX for visualization
    
    Args:
        hnsw_index: HNSW instance
        layer: Which layer to export (default: 0)
    
    Returns:
        NetworkX graph object or None if NetworkX not installed
    """
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX not installed. Install with: pip install networkx")
        return None
    
    G = nx.Graph()
    
    # Add nodes
    for node_id, node in hnsw_index.nodes.items():
        if node.max_layer >= layer:
            G.add_node(node_id, max_layer=node.max_layer)
    
    # Add edges
    for node_id, node in hnsw_index.nodes.items():
        if node.max_layer >= layer:
            for neighbor_id in node.neighbors[layer]:
                G.add_edge(node_id, neighbor_id)
    
    return G


def visualize_layer(hnsw_index, layer=0, figsize=(12, 8)):
    """
    Visualize a specific layer using matplotlib
    
    Args:
        hnsw_index: HNSW instance
        layer: Which layer to visualize
        figsize: Figure size tuple
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install networkx and matplotlib: pip install networkx matplotlib")
        return
    
    G = to_networkx(hnsw_index, layer)
    
    if G is None:
        return
    
    plt.figure(figsize=figsize)
    
    # Color nodes by their max_layer
    node_colors = [hnsw_index.nodes[n].max_layer for n in G.nodes()]
    
    # Highlight entry point
    node_sizes = [1000 if n == hnsw_index.entrypoint else 300 for n in G.nodes()]
    
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, cmap='viridis', alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5)
    
    plt.title(f'HNSW Graph - Layer {layer}')
    plt.axis('off')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                               norm=plt.Normalize(vmin=0, vmax=hnsw_index.max_layer))
    sm.set_array([])
    plt.colorbar(sm, label='Max Layer', ax=plt.gca())
    
    plt.tight_layout()
    plt.show()


def print_insertion_trace(hnsw_index, vector, node_id=None):
    """
    Show what would happen during insertion without actually inserting
    Useful for debugging and understanding the algorithm
    
    Args:
        hnsw_index: HNSW instance
        vector: Vector to trace insertion for
        node_id: Optional node ID
    """
    
    if node_id is None:
        node_id = len(hnsw_index.nodes)
    
    print("=" * 80)
    print(f"INSERTION TRACE FOR NODE {node_id}")
    print("=" * 80)
    
    # Simulate layer assignment
    node_layer = hnsw_index.get_random_layer()
    print(f"Assigned layer: {node_layer}")
    print()
    
    if hnsw_index.entrypoint is None:
        print("This would be the first node (entry point)")
        return
    
    print(f"Starting from entry point: Node {hnsw_index.entrypoint} at layer {hnsw_index.max_layer}")
    print()
    
    # Navigate upper layers
    current_nearest = [hnsw_index.entrypoint]
    
    for layer in range(hnsw_index.max_layer, node_layer, -1):
        print(f"{'─' * 60}")
        print(f"NAVIGATING Layer {layer} (Greedy Search, ef=1)")
        print(f"{'─' * 60}")
        print(f"Starting from: {current_nearest}")
        
        candidates = hnsw_index.search_layer(vector, current_nearest, layer, ef=1)
        current_nearest = [candidates[0][0]]
        
        print(f"Found closest: Node {current_nearest[0]}")
        print()
    
    # Insert at relevant layers
    for layer in range(node_layer, -1, -1):
        print(f"{'─' * 60}")
        print(f"INSERTING at Layer {layer} (Beam Search, ef={hnsw_index.ef_construction})")
        print(f"{'─' * 60}")
        print(f"Starting search from: {current_nearest}")
        
        candidates = hnsw_index.search_layer(
            vector, current_nearest, layer, ef=hnsw_index.ef_construction
        )
        
        print(f"Found {len(candidates)} candidates:")
        for i, (nid, dist) in enumerate(candidates[:5]):
            print(f"  {i+1}. Node {nid}: distance={dist:.4f}")
        if len(candidates) > 5:
            print(f"  ... and {len(candidates) - 5} more")
        
        M = hnsw_index.M if layer > 0 else hnsw_index.M_max
        neighbors = hnsw_index.select_neighbors_heuristic(vector, candidates, M)
        
        print(f"\nWould connect to {len(neighbors)} neighbors: {neighbors}")
        print()
        
        current_nearest = neighbors
    
    print("=" * 80)