# backend_router.py - Enhanced version with additional fixes and optimizations

import logging
import networkx as nx
from geopy.distance import geodesic
import pickle
import os
from typing import Dict, List, Optional, Tuple, Any, Union
import time
from collections import defaultdict
import json
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
GRAPH_CACHE = None
GRAPH_LAST_LOADED = 0
GRAPH_CACHE_TTL = 300  # 5 minutes

def load_and_fix_graph(force_reload: bool = False) -> nx.MultiGraph:
    """
    Load graph with caching and automatic connectivity fixing
    """
    global GRAPH_CACHE, GRAPH_LAST_LOADED
    
    current_time = time.time()
    
    # Check cache
    if (not force_reload and 
        GRAPH_CACHE is not None and 
        (current_time - GRAPH_LAST_LOADED) < GRAPH_CACHE_TTL):
        logger.debug("Returning cached graph")
        return GRAPH_CACHE
    
    logger.info("Loading multimodal graph...")
    
    # Try to load existing graph
    graph_paths = [
        "multimodal_graph.pkl",
        "multimodal_graph_fixed.pkl",
        "graph.pkl",
        "network.pkl"
    ]
    
    graph = None
    for graph_path in graph_paths:
        if os.path.exists(graph_path):
            try:
                with open(graph_path, 'rb') as f:
                    graph = pickle.load(f)
                logger.info(f"Loaded graph from {graph_path}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
                break
            except Exception as e:
                logger.warning(f"Error loading graph from {graph_path}: {e}")
                continue
    
    if graph is None:
        logger.warning("No valid graph file found, creating fallback graph")
        graph = create_fallback_graph()
    
    # Validate and repair graph data
    graph = validate_and_repair_graph(graph)
    
    # Check connectivity and fix if needed
    if not nx.is_connected(graph):
        logger.warning("Graph is not connected, applying fixes...")
        
        # Apply connectivity fixes
        try:
            edges_added = fix_graph_connectivity_inline(graph)
            logger.info(f"Added {edges_added} connectivity edges")
            
            # Save fixed graph
            fixed_path = "multimodal_graph_fixed.pkl"
            try:
                with open(fixed_path, 'wb') as f:
                    pickle.dump(graph, f)
                logger.info(f"Saved fixed graph to {fixed_path}")
            except Exception as e:
                logger.warning(f"Could not save fixed graph: {e}")
            
        except Exception as e:
            logger.error(f"Error fixing graph connectivity: {e}")
    
    # Update cache
    GRAPH_CACHE = graph
    GRAPH_LAST_LOADED = current_time
    
    return graph

def validate_and_repair_graph(graph: nx.MultiGraph) -> nx.MultiGraph:
    """
    Validate and repair graph data integrity
    """
    logger.info("Validating and repairing graph data...")
    
    nodes_to_remove = []
    edges_to_repair = []
    
    # Validate nodes
    for node_id, node_data in graph.nodes(data=True):
        # Check for required coordinates
        if 'y' not in node_data or 'x' not in node_data:
            logger.debug(f"Node {node_id} missing coordinates")
            nodes_to_remove.append(node_id)
            continue
        
        # Validate coordinate values
        try:
            lat, lon = float(node_data['y']), float(node_data['x'])
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                logger.debug(f"Node {node_id} has invalid coordinates: ({lat}, {lon})")
                nodes_to_remove.append(node_id)
                continue
        except (ValueError, TypeError):
            logger.debug(f"Node {node_id} has non-numeric coordinates")
            nodes_to_remove.append(node_id)
            continue
        
        # Ensure mode is set
        if 'mode' not in node_data:
            if '_' in str(node_id):
                node_data['mode'] = str(node_id).split('_')[-1]
            else:
                node_data['mode'] = 'walk'
    
    # Remove invalid nodes
    for node in nodes_to_remove:
        graph.remove_node(node)
    
    if nodes_to_remove:
        logger.info(f"Removed {len(nodes_to_remove)} invalid nodes")
    
    # Validate and repair edges
    for node1, node2, key, edge_data in graph.edges(data=True, keys=True):
        if node1 not in graph or node2 not in graph:
            continue
        
        # Calculate distance if missing
        if 'length' not in edge_data or edge_data['length'] is None:
            try:
                node1_data = graph.nodes[node1]
                node2_data = graph.nodes[node2]
                distance = geodesic(
                    (node1_data['y'], node1_data['x']),
                    (node2_data['y'], node2_data['x'])
                ).meters
                edge_data['length'] = distance
            except Exception as e:
                logger.debug(f"Could not calculate distance for edge {node1}-{node2}: {e}")
                edge_data['length'] = 100  # Default distance
        
        # Calculate time if missing
        if 'time' not in edge_data or edge_data['time'] is None:
            length = edge_data.get('length', 100)
            mode = edge_data.get('mode', 'walk')
            
            # Speed in m/s
            speed_map = {
                'walk': 1.39,    # 5 km/h
                'bike': 4.17,    # 15 km/h
                'car': 13.89,    # 50 km/h
                'bus': 8.33,     # 30 km/h
                'tram': 8.33,    # 30 km/h
                'metro': 16.67,  # 60 km/h
                'train': 27.78   # 100 km/h
            }
            
            speed = speed_map.get(mode, 1.39)
            edge_data['time'] = length / speed
        
        # Set cost if missing
        if 'cost' not in edge_data:
            edge_data['cost'] = edge_data.get('time', 0)
    
    logger.info(f"Graph validation complete: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph

def fix_graph_connectivity_inline(graph: nx.MultiGraph, max_distance: float = 3000) -> int:
    """
    Enhanced connectivity fixer with better algorithms
    """
    logger.info("Fixing graph connectivity...")
    
    # Find connected components
    components = list(nx.connected_components(graph))
    if len(components) <= 1:
        logger.info("Graph is already connected")
        return 0
    
    logger.info(f"Found {len(components)} disconnected components")
    
    # Sort components by size (largest first)
    components_sorted = sorted(components, key=len, reverse=True)
    edges_added = 0
    
    # Strategy 1: Connect all components to the largest one
    main_component = components_sorted[0]
    
    for comp_idx, component in enumerate(components_sorted[1:], 1):
        best_connections = find_best_connections(graph, component, main_component, max_distance)
        
        if best_connections:
            # Add the best connection
            node1, node2, distance = best_connections[0]
            
            # Add edge with proper attributes
            travel_time = distance / 1.39  # Walking speed
            graph.add_edge(node1, node2, 
                         length=distance,
                         time=travel_time,
                         cost=travel_time,
                         mode='walk',
                         highway='footway',
                         connection_type='auto_generated')
            
            edges_added += 1
            logger.info(f"Connected component {comp_idx} to main component: {node1} -> {node2} ({distance:.1f}m)")
            
            # Update main component
            main_component = main_component.union(component)
        else:
            logger.warning(f"Could not connect component {comp_idx} - no nodes within {max_distance}m")
    
    # Strategy 2: Add intermodal connections
    intermodal_edges = add_intermodal_connections(graph, max_distance=800)
    edges_added += intermodal_edges
    
    # Strategy 3: Add redundant connections for reliability
    redundant_edges = add_redundant_connections(graph, max_distance=1000)
    edges_added += redundant_edges
    
    logger.info(f"Total edges added: {edges_added}")
    return edges_added

def find_best_connections(graph: nx.MultiGraph, component1: set, component2: set, 
                         max_distance: float, max_connections: int = 3) -> List[Tuple[str, str, float]]:
    """
    Find the best connections between two components
    """
    connections = []
    
    # Sample nodes to avoid performance issues
    nodes1 = list(component1)[:min(50, len(component1))]
    nodes2 = list(component2)[:min(100, len(component2))]
    
    for node1 in nodes1:
        if not graph.has_node(node1):
            continue
        
        node1_data = graph.nodes[node1]
        if 'x' not in node1_data or 'y' not in node1_data:
            continue
        
        for node2 in nodes2:
            if not graph.has_node(node2):
                continue
            
            node2_data = graph.nodes[node2]
            if 'x' not in node2_data or 'y' not in node2_data:
                continue
            
            # Calculate distance
            distance = geodesic(
                (node1_data['y'], node1_data['x']),
                (node2_data['y'], node2_data['x'])
            ).meters
            
            if distance <= max_distance:
                connections.append((node1, node2, distance))
    
    # Sort by distance and return best connections
    connections.sort(key=lambda x: x[2])
    return connections[:max_connections]

def add_intermodal_connections(graph: nx.MultiGraph, max_distance: float = 800) -> int:
    """
    Add connections between different transport modes
    """
    logger.info("Adding intermodal connections...")
    
    # Group nodes by mode
    mode_nodes = defaultdict(list)
    for node_id, node_data in graph.nodes(data=True):
        mode = node_data.get('mode', 'walk')
        if '_' in str(node_id) and mode == 'walk':  # Extract mode from node ID
            mode = str(node_id).split('_')[-1]
        mode_nodes[mode].append(node_id)
    
    logger.info(f"Found transport modes: {list(mode_nodes.keys())}")
    
    edges_added = 0
    
    # Add connections between complementary modes
    mode_pairs = [
        ('walk', 'bike'),
        ('walk', 'bus'),
        ('walk', 'tram'),
        ('walk', 'metro'),
        ('bike', 'train'),
        ('bus', 'metro'),
        ('tram', 'metro')
    ]
    
    for mode1, mode2 in mode_pairs:
        if mode1 not in mode_nodes or mode2 not in mode_nodes:
            continue
        
        # Find close nodes between these modes
        nodes1 = mode_nodes[mode1][:20]  # Limit for performance
        nodes2 = mode_nodes[mode2][:20]
        
        for node1 in nodes1:
            if not graph.has_node(node1):
                continue
            
            node1_data = graph.nodes[node1]
            if 'x' not in node1_data or 'y' not in node1_data:
                continue
            
            closest_distance = float('inf')
            closest_node2 = None
            
            for node2 in nodes2:
                if not graph.has_node(node2):
                    continue
                
                node2_data = graph.nodes[node2]
                if 'x' not in node2_data or 'y' not in node2_data:
                    continue
                
                distance = geodesic(
                    (node1_data['y'], node1_data['x']),
                    (node2_data['y'], node2_data['x'])
                ).meters
                
                if distance < closest_distance and distance <= max_distance:
                    closest_distance = distance
                    closest_node2 = node2
            
            # Add transfer edge
            if closest_node2 and not graph.has_edge(node1, closest_node2):
                transfer_time = closest_distance / 1.39 + 120  # Walking time + 2 min transfer
                
                graph.add_edge(node1, closest_node2,
                             length=closest_distance,
                             time=transfer_time,
                             cost=transfer_time,
                             mode='transfer',
                             connection_type='intermodal')
                
                edges_added += 1
                
                if edges_added >= 20:  # Limit to avoid too many edges
                    break
        
        if edges_added >= 20:
            break
    
    logger.info(f"Added {edges_added} intermodal connections")
    return edges_added

def add_redundant_connections(graph: nx.MultiGraph, max_distance: float = 1000) -> int:
    """
    Add redundant connections to improve routing reliability
    """
    logger.info("Adding redundant connections...")
    
    edges_added = 0
    
    # Find nodes with low connectivity
    low_connectivity_nodes = []
    for node_id in graph.nodes():
        if graph.degree(node_id) <= 2:
            low_connectivity_nodes.append(node_id)
    
    logger.info(f"Found {len(low_connectivity_nodes)} low-connectivity nodes")
    
    # Add connections for low-connectivity nodes
    for node1 in low_connectivity_nodes[:50]:  # Limit for performance
        if not graph.has_node(node1):
            continue
        
        node1_data = graph.nodes[node1]
        if 'x' not in node1_data or 'y' not in node1_data:
            continue
        
        # Find nearby nodes
        nearby_nodes = []
        for node2 in graph.nodes():
            if node2 == node1 or graph.has_edge(node1, node2):
                continue
            
            node2_data = graph.nodes[node2]
            if 'x' not in node2_data or 'y' not in node2_data:
                continue
            
            distance = geodesic(
                (node1_data['y'], node1_data['x']),
                (node2_data['y'], node2_data['x'])
            ).meters
            
            if distance <= max_distance:
                nearby_nodes.append((node2, distance))
        
        # Sort by distance and add connection to closest node
        nearby_nodes.sort(key=lambda x: x[1])
        
        if nearby_nodes:
            node2, distance = nearby_nodes[0]
            travel_time = distance / 1.39  # Walking speed
            
            graph.add_edge(node1, node2,
                         length=distance,
                         time=travel_time,
                         cost=travel_time,
                         mode='walk',
                         connection_type='redundant')
            
            edges_added += 1
            
            if edges_added >= 10:  # Limit redundant edges
                break
    
    logger.info(f"Added {edges_added} redundant connections")
    return edges_added

def create_fallback_graph() -> nx.MultiGraph:
    """
    Create a comprehensive fallback graph for testing
    """
    logger.info("Creating fallback graph...")
    
    graph = nx.MultiGraph()
    
    # Amsterdam area coordinates
    base_lat, base_lon = 52.3676, 4.9041
    
    # Create a connected grid of nodes
    grid_size = 5
    spacing = 0.005  # Roughly 500m
    
    nodes = []
    
    # Create grid nodes
    for i in range(grid_size):
        for j in range(grid_size):
            lat = base_lat + (i - grid_size//2) * spacing
            lon = base_lon + (j - grid_size//2) * spacing
            
            # Create nodes for different modes
            walk_node = f"{i*grid_size+j}_walk"
            bike_node = f"{i*grid_size+j}_bike"
            
            graph.add_node(walk_node, x=lon, y=lat, mode='walk')
            graph.add_node(bike_node, x=lon + spacing*0.1, y=lat + spacing*0.1, mode='bike')
            
            nodes.append((walk_node, bike_node))
    
    # Connect adjacent nodes
    for i in range(grid_size):
        for j in range(grid_size):
            current_idx = i * grid_size + j
            walk_node, bike_node = nodes[current_idx]
            
            # Connect to adjacent nodes
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    adjacent_idx = ni * grid_size + nj
                    adj_walk_node, adj_bike_node = nodes[adjacent_idx]
                    
                    # Walking connections
                    distance = geodesic(
                        (graph.nodes[walk_node]['y'], graph.nodes[walk_node]['x']),
                        (graph.nodes[adj_walk_node]['y'], graph.nodes[adj_walk_node]['x'])
                    ).meters
                    
                    graph.add_edge(walk_node, adj_walk_node,
                                 length=distance,
                                 time=distance/1.39,
                                 cost=distance/1.39,
                                 mode='walk')
                    
                    # Bike connections
                    bike_distance = geodesic(
                        (graph.nodes[bike_node]['y'], graph.nodes[bike_node]['x']),
                        (graph.nodes[adj_bike_node]['y'], graph.nodes[adj_bike_node]['x'])
                    ).meters
                    
                    graph.add_edge(bike_node, adj_bike_node,
                                 length=bike_distance,
                                 time=bike_distance/4.17,
                                 cost=bike_distance/4.17,
                                 mode='bike')
            
            # Connect walk and bike modes
            transfer_distance = geodesic(
                (graph.nodes[walk_node]['y'], graph.nodes[walk_node]['x']),
                (graph.nodes[bike_node]['y'], graph.nodes[bike_node]['x'])
            ).meters
            
            graph.add_edge(walk_node, bike_node,
                         length=transfer_distance,
                         time=transfer_distance/1.39 + 60,  # 1 min transfer
                         cost=transfer_distance/1.39 + 60,
                         mode='transfer')
    
    logger.info(f"Created fallback graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph

def find_nearest_nodes_all_modes(graph: nx.MultiGraph, lat: float, lon: float, 
                                max_distance: float = 1000) -> List[str]:
    """
    Find nearest nodes across all transport modes
    """
    distances = []
    
    for node_id, node_data in graph.nodes(data=True):
        if 'x' not in node_data or 'y' not in node_data:
            continue
        
        distance = geodesic(
            (lat, lon),
            (node_data['y'], node_data['x'])
        ).meters
        
        if distance <= max_distance:
            distances.append((node_id, distance))
    
    # Sort by distance
    distances.sort(key=lambda x: x[1])
    return [node_id for node_id, _ in distances]

def find_path_with_fallback(graph: nx.MultiGraph, start_node: str, end_node: str) -> Optional[List[str]]:
    """
    Find path with multiple fallback strategies
    """
    logger.debug(f"Finding path from {start_node} to {end_node}")
    
    # Check if nodes exist
    if start_node not in graph or end_node not in graph:
        logger.warning(f"Nodes not found in graph: start={start_node in graph}, end={end_node in graph}")
        return None
    
    # Strategy 1: Weighted shortest path
    try:
        if nx.has_path(graph, start_node, end_node):
            path = nx.shortest_path(graph, start_node, end_node, weight='time')
            logger.debug(f"Found weighted path with {len(path)} nodes")
            return path
    except Exception as e:
        logger.debug(f"Weighted path failed: {e}")
    
    # Strategy 2: Unweighted shortest path
    try:
        if nx.has_path(graph, start_node, end_node):
            path = nx.shortest_path(graph, start_node, end_node)
            logger.debug(f"Found unweighted path with {len(path)} nodes")
            return path
    except Exception as e:
        logger.debug(f"Unweighted path failed: {e}")
    
    # Strategy 3: Multi-source shortest path (find intermediate nodes)
    try:
        # Find nodes connected to both start and end
        start_neighbors = set(nx.single_source_shortest_path_length(graph, start_node, cutoff=3).keys())
        end_neighbors = set(nx.single_source_shortest_path_length(graph, end_node, cutoff=3).keys())
        
        common_neighbors = start_neighbors.intersection(end_neighbors)
        if common_neighbors:
            # Find best intermediate node
            best_intermediate = min(common_neighbors, 
                                  key=lambda n: nx.shortest_path_length(graph, start_node, n) + 
                                               nx.shortest_path_length(graph, n, end_node))
            
            path1 = nx.shortest_path(graph, start_node, best_intermediate)
            path2 = nx.shortest_path(graph, best_intermediate, end_node)
            
            # Combine paths (avoid duplicating intermediate node)
            full_path = path1 + path2[1:]
            logger.debug(f"Found path via intermediate node {best_intermediate}: {len(full_path)} nodes")
            return full_path
    except Exception as e:
        logger.debug(f"Intermediate path failed: {e}")
    
    logger.warning(f"No path found between {start_node} and {end_node}")
    return None

def get_multimodal_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> Dict[str, Any]:
    """
    Calculate the optimal multimodal route between two points
    """
    try:
        # Load and fix graph
        graph = load_and_fix_graph()
        
        logger.info(f"Calculating route from ({start_lat}, {start_lon}) to ({end_lat}, {end_lon})")
        
        # Find nearest nodes with multiple candidates
        start_candidates = find_nearest_nodes_all_modes(graph, start_lat, start_lon, 2000)
        end_candidates = find_nearest_nodes_all_modes(graph, end_lat, end_lon, 2000)
        
        if not start_candidates or not end_candidates:
            logger.warning("Could not find nearby nodes, using direct route")
            return create_direct_route(start_lat, start_lon, end_lat, end_lon)
        
        logger.info(f"Found {len(start_candidates)} start candidates and {len(end_candidates)} end candidates")
        
        # Try multiple start-end combinations
        best_path = None
        best_start_node = None
        best_end_node = None
        
        for start_node in start_candidates[:5]:  # Try top 5 candidates
            for end_node in end_candidates[:5]:
                path = find_path_with_fallback(graph, start_node, end_node)
                if path:
                    best_path = path
                    best_start_node = start_node
                    best_end_node = end_node
                    break
            if best_path:
                break
        
        if not best_path:
            logger.warning("No path found between any candidate nodes")
            return create_direct_route(start_lat, start_lon, end_lat, end_lon)
        
        logger.info(f"Found path: {best_start_node} -> {best_end_node} ({len(best_path)} nodes)")
        
        # Convert path to route segments
        segments = path_to_segments(graph, best_path, start_lat, start_lon, end_lat, end_lon)
        
        # Calculate totals
        total_distance = sum(seg['distance'] for seg in segments)
        total_time = sum(seg['time'] for seg in segments)
        total_cost = sum(seg.get('cost', 0) for seg in segments)
        
        # Create coordinates for visualization
        path_coordinates = [[start_lat, start_lon]]
        for segment in segments:
            path_coordinates.append(segment['end'])
        
        # Get transport mode breakdown
        mode_breakdown = defaultdict(float)
        for segment in segments:
            mode_breakdown[segment['mode']] += segment['distance']
        
        result = {
            'segments': segments,
            'total_distance': total_distance,
            'total_time': total_time,
            'total_cost': total_cost,
            'path_coordinates': path_coordinates,
            'mode_breakdown': dict(mode_breakdown),
            'path_nodes': best_path
        }
        
        logger.info(f"Route calculated: {len(segments)} segments, {total_distance:.1f}m, {total_time:.1f}s")
        return result
        
    except Exception as e:
        logger.error(f"Error in route calculation: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return create_direct_route(start_lat, start_lon, end_lat, end_lon)

def path_to_segments(graph: nx.MultiGraph, path: List[str], start_lat: float, start_lon: float, 
                    end_lat: float, end_lon: float) -> List[Dict[str, Any]]:
    """
    Convert a path to route segments with enhanced mode detection
    """
    segments = []
    
    if not path:
        return segments
    
    # Add walking segment from start to first node
    first_node_data = graph.nodes[path[0]]
    first_distance = geodesic(
        (start_lat, start_lon),
        (first_node_data['y'], first_node_data['x'])
    ).meters
    
    segments.append({
        'mode': 'walk',
        'start': [start_lat, start_lon],
        'end': [first_node_data['y'], first_node_data['x']],
        'distance': first_distance,
        'time': first_distance / 1.39,
        'cost': 0,
        'description': 'Walk to start point'
    })
    
    # Add segments for each edge in the path
    for i in range(len(path) - 1):
        node1_data = graph.nodes[path[i]]
        node2_data = graph.nodes[path[i + 1]]
        
        # Get edge data
        edge_data = graph.get_edge_data(path[i], path[i + 1])
        if edge_data:
            # MultiGraph returns dict of edge keys, get the best edge
            edge_attrs = list(edge_data.values())[0]
            
            # If multiple edges, pick the fastest
            if len(edge_data) > 1:
                edge_attrs = min(edge_data.values(), key=lambda x: x.get('time', float('inf')))
        else:
            # Create default edge attributes
            distance = geodesic(
                (node1_data['y'], node1_data['x']),
                (node2_data['y'], node2_data['x'])
            ).meters
            
            edge_attrs = {
                'length': distance,
                'time': distance / 1.39,
                'cost': 0,
                'mode': 'walk'
            }
        
        # Determine mode from edge or nodes
        mode = edge_attrs.get('mode', 'walk')
        if mode == 'walk':
            # Try to infer from node IDs
            if '_' in path[i]:
                mode = path[i].split('_')[-1]
            elif '_' in path[i + 1]:
                mode = path[i + 1].split('_')[-1]
        
        segments.append({
            'mode': mode,
            'start': [node1_data['y'], node1_data['x']],
            'end': [node2_data['y'], node2_data['x']],
            'distance': edge_attrs.get('length', 0),
            'time': edge_attrs.get('time', 0),
            'cost': edge_attrs.get('cost', 0),
            'description': f'Travel by {mode}'
        })
    
    # Add walking segment from last node to end
    last_node_data = graph.nodes[path[-1]]
    last_distance = geodesic(
        (last_node_data['y'], last_node_data['x']),
        (end_lat, end_lon)
    ).meters
    
    segments.append({
        'mode': 'walk',
        'start': [last_node_data['y'], last_node_data['x']],
        'end': [end_lat, end_lon],
        'distance': last_distance,
        'time': last_distance / 1.39,
        'cost': 0,
        'description': 'Walk to destination'
    })
    
    return segments




def create_direct_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> Dict[str, Any]:
    """
    Create a direct walking route as fallback
    """
    distance = geodesic((start_lat, start_lon), (end_lat, end_lon)).meters
    travel_time = distance / 1.39  # Walking speed: 1.39 m/s (5 km/h)
    
    logger.info(f"Creating direct route: {distance:.1f}m, {travel_time:.1f}s")
    
    segment = {
        'mode': 'walk',
        'start': [start_lat, start_lon],
        'end': [end_lat, end_lon],
        'distance': distance,
        'time': travel_time,
        'cost': 0,
        'description': 'Direct walking route'
    }
    
    return {
        'segments': [segment],
        'total_distance': distance,
        'total_time': travel_time,
        'total_cost': 0,
        'path_coordinates': [[start_lat, start_lon], [end_lat, end_lon]],
        'mode_breakdown': {'walk': distance},
        'path_nodes': []
    }

def get_route_alternatives(start_lat: float, start_lon: float, end_lat: float, end_lon: float, 
                          max_alternatives: int = 3) -> List[Dict[str, Any]]:
    """
    Get multiple route alternatives with different optimization criteria
    """
    logger.info(f"Calculating {max_alternatives} route alternatives")
    
    alternatives = []
    
    try:
        graph = load_and_fix_graph()
        
        # Find candidate nodes
        start_candidates = find_nearest_nodes_all_modes(graph, start_lat, start_lon, 2000)
        end_candidates = find_nearest_nodes_all_modes(graph, end_lat, end_lon, 2000)
        
        if not start_candidates or not end_candidates:
            logger.warning("No candidates found for alternatives")
            return [create_direct_route(start_lat, start_lon, end_lat, end_lon)]
        
        # Different optimization strategies
        strategies = [
            ('time', 'time'),      # Fastest route
            ('length', 'length'),  # Shortest route
            ('cost', 'cost')       # Cheapest route
        ]
        
        found_paths = set()
        
        for strategy_name, weight_attr in strategies:
            if len(alternatives) >= max_alternatives:
                break
            
            logger.debug(f"Trying strategy: {strategy_name}")
            
            # Try different start-end combinations
            for start_node in start_candidates[:3]:
                for end_node in end_candidates[:3]:
                    try:
                        if nx.has_path(graph, start_node, end_node):
                            path = nx.shortest_path(graph, start_node, end_node, weight=weight_attr)
                            path_key = tuple(path)
                            
                            # Skip if we already found this path
                            if path_key in found_paths:
                                continue
                            
                            found_paths.add(path_key)
                            
                            # Convert to route
                            segments = path_to_segments(graph, path, start_lat, start_lon, end_lat, end_lon)
                            
                            # Calculate totals
                            total_distance = sum(seg['distance'] for seg in segments)
                            total_time = sum(seg['time'] for seg in segments)
                            total_cost = sum(seg.get('cost', 0) for seg in segments)
                            
                            # Create coordinates
                            path_coordinates = [[start_lat, start_lon]]
                            for segment in segments:
                                path_coordinates.append(segment['end'])
                            
                            # Get mode breakdown
                            mode_breakdown = defaultdict(float)
                            for segment in segments:
                                mode_breakdown[segment['mode']] += segment['distance']
                            
                            route = {
                                'segments': segments,
                                'total_distance': total_distance,
                                'total_time': total_time,
                                'total_cost': total_cost,
                                'path_coordinates': path_coordinates,
                                'mode_breakdown': dict(mode_breakdown),
                                'path_nodes': path,
                                'strategy': strategy_name
                            }
                            
                            alternatives.append(route)
                            logger.info(f"Found {strategy_name} route: {len(segments)} segments, "
                                      f"{total_distance:.1f}m, {total_time:.1f}s")
                            
                            if len(alternatives) >= max_alternatives:
                                break
                    except Exception as e:
                        logger.debug(f"Strategy {strategy_name} failed for {start_node}-{end_node}: {e}")
                        continue
                
                if len(alternatives) >= max_alternatives:
                    break
        
        # If we don't have enough alternatives, add direct route
        if len(alternatives) == 0:
            alternatives.append(create_direct_route(start_lat, start_lon, end_lat, end_lon))
        
        # Sort alternatives by total time
        alternatives.sort(key=lambda x: x['total_time'])
        
        logger.info(f"Returning {len(alternatives)} route alternatives")
        return alternatives[:max_alternatives]
        
    except Exception as e:
        logger.error(f"Error calculating alternatives: {e}")
        return [create_direct_route(start_lat, start_lon, end_lat, end_lon)]

def get_node_info(node_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific node
    """
    try:
        graph = load_and_fix_graph()
        
        if node_id not in graph:
            return {'error': f'Node {node_id} not found'}
        
        node_data = graph.nodes[node_id]
        
        # Get connections
        connections = []
        for neighbor in graph.neighbors(node_id):
            edge_data = graph.get_edge_data(node_id, neighbor)
            if edge_data:
                # Get best edge if multiple
                best_edge = min(edge_data.values(), key=lambda x: x.get('time', float('inf')))
                connections.append({
                    'neighbor': neighbor,
                    'distance': best_edge.get('length', 0),
                    'time': best_edge.get('time', 0),
                    'mode': best_edge.get('mode', 'unknown')
                })
        
        return {
            'node_id': node_id,
            'coordinates': [node_data.get('y', 0), node_data.get('x', 0)],
            'mode': node_data.get('mode', 'unknown'),
            'degree': graph.degree(node_id),
            'connections': connections,
            'attributes': dict(node_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting node info: {e}")
        return {'error': str(e)}

def get_graph_stats() -> Dict[str, Any]:
    """
    Get statistics about the loaded graph
    """
    try:
        graph = load_and_fix_graph()
        
        # Basic stats
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        is_connected = nx.is_connected(graph)
        
        # Mode breakdown
        mode_counts = defaultdict(int)
        for node_id, node_data in graph.nodes(data=True):
            mode = node_data.get('mode', 'unknown')
            if mode == 'unknown' and '_' in str(node_id):
                mode = str(node_id).split('_')[-1]
            mode_counts[mode] += 1
        
        # Connectivity stats
        components = list(nx.connected_components(graph))
        largest_component_size = max(len(comp) for comp in components) if components else 0
        
        # Degree distribution
        degrees = [graph.degree(node) for node in graph.nodes()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'is_connected': is_connected,
            'num_components': len(components),
            'largest_component_size': largest_component_size,
            'mode_breakdown': dict(mode_counts),
            'degree_stats': {
                'average': avg_degree,
                'maximum': max_degree,
                'minimum': min_degree
            },
            'graph_type': str(type(graph).__name__)
        }
        
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        return {'error': str(e)}

def export_route_geojson(route: Dict[str, Any]) -> Dict[str, Any]:
    """
    Export route as GeoJSON for visualization
    """
    try:
        features = []
        
        # Add route line
        if 'path_coordinates' in route:
            route_line = {
                'type': 'Feature',
                'properties': {
                    'type': 'route',
                    'total_distance': route.get('total_distance', 0),
                    'total_time': route.get('total_time', 0),
                    'total_cost': route.get('total_cost', 0)
                },
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[coord[1], coord[0]] for coord in route['path_coordinates']]
                }
            }
            features.append(route_line)
        
        # Add segments with different modes
        if 'segments' in route:
            for i, segment in enumerate(route['segments']):
                segment_line = {
                    'type': 'Feature',
                    'properties': {
                        'type': 'segment',
                        'segment_index': i,
                        'mode': segment['mode'],
                        'distance': segment['distance'],
                        'time': segment['time'],
                        'cost': segment.get('cost', 0),
                        'description': segment.get('description', '')
                    },
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': [
                            [segment['start'][1], segment['start'][0]],
                            [segment['end'][1], segment['end'][0]]
                        ]
                    }
                }
                features.append(segment_line)
        
        # Add start and end points
        if 'path_coordinates' in route and route['path_coordinates']:
            start_point = {
                'type': 'Feature',
                'properties': {
                    'type': 'start',
                    'name': 'Start'
                },
                'geometry': {
                    'type': 'Point',
                    'coordinates': [route['path_coordinates'][0][1], route['path_coordinates'][0][0]]
                }
            }
            features.append(start_point)
            
            end_point = {
                'type': 'Feature',
                'properties': {
                    'type': 'end',
                    'name': 'End'
                },
                'geometry': {
                    'type': 'Point',
                    'coordinates': [route['path_coordinates'][-1][1], route['path_coordinates'][-1][0]]
                }
            }
            features.append(end_point)
        
        return {
            'type': 'FeatureCollection',
            'features': features
        }
        
    except Exception as e:
        logger.error(f"Error exporting GeoJSON: {e}")
        return {'error': str(e)}

def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate latitude and longitude coordinates
    """
    try:
        lat_float = float(lat)
        lon_float = float(lon)
        
        if not (-90 <= lat_float <= 90):
            return False
        if not (-180 <= lon_float <= 180):
            return False
        
        return True
        
    except (ValueError, TypeError):
        return False

def clear_graph_cache():
    """
    Clear the graph cache to force reload
    """
    global GRAPH_CACHE, GRAPH_LAST_LOADED
    GRAPH_CACHE = None
    GRAPH_LAST_LOADED = 0
    logger.info("Graph cache cleared")

def optimize_graph_performance(graph: nx.MultiGraph) -> nx.MultiGraph:
    """
    Optimize graph for better routing performance
    """
    logger.info("Optimizing graph performance...")
    
    # Remove redundant edges with same endpoints and mode
    edges_to_remove = []
    edge_dict = defaultdict(list)
    
    # Group edges by endpoints and mode
    for node1, node2, key, data in graph.edges(data=True, keys=True):
        mode = data.get('mode', 'unknown')
        edge_key = (min(node1, node2), max(node1, node2), mode)
        edge_dict[edge_key].append((node1, node2, key, data))
    
    # Keep only the best edge for each group
    for edge_group in edge_dict.values():
        if len(edge_group) > 1:
            # Sort by time (or length if time not available)
            edge_group.sort(key=lambda x: x[3].get('time', x[3].get('length', float('inf'))))
            
            # Remove all but the best edge
            for node1, node2, key, data in edge_group[1:]:
                edges_to_remove.append((node1, node2, key))
    
    # Remove redundant edges
    for edge in edges_to_remove:
        try:
            graph.remove_edge(*edge)
        except:
            pass
    
    if edges_to_remove:
        logger.info(f"Removed {len(edges_to_remove)} redundant edges")
    
    # Add spatial index for faster nearest neighbor searches
    # This would require additional libraries like rtree or similar
    
    logger.info("Graph optimization complete")
    return graph

# Error handling wrapper
def safe_route_calculation(func):
    """
    Decorator for safe route calculation with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Route calculation failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return fallback result
            if len(args) >= 4:
                return create_direct_route(args[0], args[1], args[2], args[3])
            else:
                return {'error': str(e)}
    
    return wrapper

# Apply decorator to main functions
get_multimodal_route = safe_route_calculation(get_multimodal_route)
get_route_alternatives = safe_route_calculation(get_route_alternatives)

if __name__ == "__main__":
    # Test the routing system
    logger.info("Testing routing system...")
    
    # Test coordinates (Amsterdam area)
    start_lat, start_lon = 52.3676, 4.9041  # Amsterdam Central
    end_lat, end_lon = 52.3702, 4.8952      # Dam Square
    
    # Test basic routing
    route = get_multimodal_route(start_lat, start_lon, end_lat, end_lon)
    print(f"Route found: {len(route.get('segments', []))} segments")
    print(f"Total distance: {route.get('total_distance', 0):.1f}m")
    print(f"Total time: {route.get('total_time', 0):.1f}s")
    
    # Test alternatives
    alternatives = get_route_alternatives(start_lat, start_lon, end_lat, end_lon)
    print(f"Found {len(alternatives)} alternatives")
    
    # Test graph stats
    stats = get_graph_stats()
    print(f"Graph stats: {stats.get('num_nodes', 0)} nodes, {stats.get('num_edges', 0)} edges")
    
    # Test GeoJSON export
    geojson = export_route_geojson(route)
    print(f"GeoJSON features: {len(geojson.get('features', []))}")
    
    logger.info("Testing complete")