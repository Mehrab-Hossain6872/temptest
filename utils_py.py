# utils_py.py - Utility functions for routing

import logging
from geopy.distance import geodesic
import networkx as nx
from typing import Tuple, List, Optional, Dict, Any

logger = logging.getLogger(__name__)

def nearest_node(graph: nx.Graph, lat: float, lon: float, max_distance: float = 1000) -> Optional[str]:
    """
    Find the nearest node to a given latitude and longitude
    
    Args:
        graph: NetworkX graph with nodes containing 'y' (lat) and 'x' (lon) attributes
        lat: Target latitude
        lon: Target longitude  
        max_distance: Maximum search distance in meters
    
    Returns:
        Node ID of nearest node within max_distance, or None if not found
    """
    try:
        min_distance = float('inf')
        nearest_node_id = None
        
        for node_id, node_data in graph.nodes(data=True):
            # Skip nodes without coordinates
            if 'y' not in node_data or 'x' not in node_data:
                continue
                
            try:
                node_lat = float(node_data['y'])
                node_lon = float(node_data['x'])
                
                # Calculate distance
                distance = geodesic((lat, lon), (node_lat, node_lon)).meters
                
                if distance < min_distance and distance <= max_distance:
                    min_distance = distance
                    nearest_node_id = node_id
                    
            except (ValueError, TypeError) as e:
                logger.debug(f"Invalid coordinates for node {node_id}: {e}")
                continue
        
        if nearest_node_id:
            logger.debug(f"Requested: ({lat:.6f}, {lon:.6f}), "
                        f"Nearest: ({graph.nodes[nearest_node_id]['y']:.6f}, {graph.nodes[nearest_node_id]['x']:.6f}), "
                        f"Distance: {min_distance:.2f}m")
            return nearest_node_id
        else:
            logger.warning(f"No nodes found within {max_distance}m of ({lat:.6f}, {lon:.6f})")
            return None
            
    except Exception as e:
        logger.error(f"Error finding nearest node: {e}")
        return None

def calc_cost(graph: nx.Graph, path: List[str]) -> Dict[str, float]:
    """
    Calculate the total cost, time, and distance for a path
    
    Args:
        graph: NetworkX graph with edge weights
        path: List of node IDs representing the path
    
    Returns:
        Dictionary with 'cost', 'time', and 'distance' keys
    """
    try:
        if not path or len(path) < 2:
            return {'cost': 0, 'time': 0, 'distance': 0}
        
        total_cost = 0
        total_time = 0
        total_distance = 0
        
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            
            # Get edge data
            edge_data = graph.get_edge_data(current_node, next_node)
            
            if edge_data:
                total_cost += edge_data.get('cost', 0)
                total_time += edge_data.get('time', 0)
                total_distance += edge_data.get('length', 0)
            else:
                logger.warning(f"No edge data found between {current_node} and {next_node}")
        
        return {
            'cost': total_cost,
            'time': total_time,
            'distance': total_distance
        }
        
    except Exception as e:
        logger.error(f"Error calculating path cost: {e}")
        return {'cost': float('inf'), 'time': float('inf'), 'distance': float('inf')}

def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate latitude and longitude coordinates
    
    Args:
        lat: Latitude
        lon: Longitude
    
    Returns:
        True if valid, False otherwise
    """
    try:
        lat = float(lat)
        lon = float(lon)
        
        return -90 <= lat <= 90 and -180 <= lon <= 180
    except (ValueError, TypeError):
        return False

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the distance between two geographic points
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
    
    Returns:
        Distance in meters
    """
    try:
        return geodesic((lat1, lon1), (lat2, lon2)).meters
    except Exception as e:
        logger.error(f"Error calculating distance: {e}")
        return float('inf')

def get_node_mode(node_id: str, node_data: Dict[str, Any]) -> str:
    """
    Determine the transport mode for a node
    
    Args:
        node_id: Node identifier
        node_data: Node attributes dictionary
    
    Returns:
        Transport mode string ('walk', 'bike', 'car', etc.)
    """
    # Check if mode is explicitly set
    if 'mode' in node_data:
        return node_data['mode']
    
    # Extract mode from node ID if available
    if '_' in node_id:
        return node_id.split('_')[-1]
    
    # Default to walking
    return 'walk'

def estimate_travel_time(distance: float, mode: str) -> float:
    """
    Estimate travel time based on distance and transport mode
    
    Args:
        distance: Distance in meters
        mode: Transport mode ('walk', 'bike', 'car', etc.)
    
    Returns:
        Estimated time in seconds
    """
    # Speed in meters per second
    speeds = {
        'walk': 1.39,    # 5 km/h
        'bike': 4.17,    # 15 km/h
        'car': 13.89,    # 50 km/h
        'bus': 8.33,     # 30 km/h
        'tram': 11.11,   # 40 km/h
        'metro': 16.67,  # 60 km/h
        'train': 22.22   # 80 km/h
    }
    
    speed = speeds.get(mode, speeds['walk'])  # Default to walking speed
    return distance / speed

def find_nodes_in_radius(graph: nx.Graph, lat: float, lon: float, radius: float) -> List[Tuple[str, float]]:
    """
    Find all nodes within a given radius of a point
    
    Args:
        graph: NetworkX graph
        lat: Center latitude
        lon: Center longitude
        radius: Search radius in meters
    
    Returns:
        List of (node_id, distance) tuples, sorted by distance
    """
    nodes_in_radius = []
    
    for node_id, node_data in graph.nodes(data=True):
        if 'y' not in node_data or 'x' not in node_data:
            continue
            
        try:
            distance = calculate_distance(lat, lon, node_data['y'], node_data['x'])
            if distance <= radius:
                nodes_in_radius.append((node_id, distance))
        except Exception as e:
            logger.debug(f"Error processing node {node_id}: {e}")
            continue
    
    # Sort by distance
    nodes_in_radius.sort(key=lambda x: x[1])
    return nodes_in_radius

def get_path_segments(graph: nx.Graph, path: List[str]) -> List[Dict[str, Any]]:
    """
    Convert a path to detailed segments with mode information
    
    Args:
        graph: NetworkX graph
        path: List of node IDs
    
    Returns:
        List of segment dictionaries
    """
    segments = []
    
    if not path or len(path) < 2:
        return segments
    
    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        
        current_data = graph.nodes[current_node]
        next_data = graph.nodes[next_node]
        edge_data = graph.get_edge_data(current_node, next_node)
        
        # Determine mode
        mode = get_node_mode(current_node, current_data)
        
        segment = {
            'from_node': current_node,
            'to_node': next_node,
            'mode': mode,
            'start_coords': [current_data['y'], current_data['x']],
            'end_coords': [next_data['y'], next_data['x']],
            'distance': edge_data.get('length', 0) if edge_data else 0,
            'time': edge_data.get('time', 0) if edge_data else 0,
            'cost': edge_data.get('cost', 0) if edge_data else 0
        }
        
        segments.append(segment)
    
    return segments

def interpolate_path(start_coords: List[float], end_coords: List[float], num_points: int = 10) -> List[List[float]]:
    """
    Interpolate points between start and end coordinates
    
    Args:
        start_coords: [lat, lon] of start point
        end_coords: [lat, lon] of end point
        num_points: Number of intermediate points to generate
    
    Returns:
        List of [lat, lon] coordinates
    """
    if num_points <= 0:
        return [start_coords, end_coords]
    
    points = [start_coords]
    
    lat_step = (end_coords[0] - start_coords[0]) / (num_points + 1)
    lon_step = (end_coords[1] - start_coords[1]) / (num_points + 1)
    
    for i in range(1, num_points + 1):
        lat = start_coords[0] + lat_step * i
        lon = start_coords[1] + lon_step * i
        points.append([lat, lon])
    
    points.append(end_coords)
    return points

def merge_consecutive_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive segments with the same transport mode
    
    Args:
        segments: List of segment dictionaries
    
    Returns:
        List of merged segments
    """
    if not segments:
        return []
    
    merged = []
    current_segment = segments[0].copy()
    
    for i in range(1, len(segments)):
        next_segment = segments[i]
        
        # If same mode, merge with current segment
        if next_segment['mode'] == current_segment['mode']:
            current_segment['end_coords'] = next_segment['end_coords']
            current_segment['to_node'] = next_segment['to_node']
            current_segment['distance'] += next_segment['distance']
            current_segment['time'] += next_segment['time']
            current_segment['cost'] += next_segment['cost']
        else:
            # Different mode, save current and start new
            merged.append(current_segment)
            current_segment = next_segment.copy()
    
    # Add the last segment
    merged.append(current_segment)
    
    return merged

def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{int(seconds)} sec"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} min"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}min"

def format_distance(meters: float) -> str:
    """
    Format distance in meters to human-readable string
    
    Args:
        meters: Distance in meters
    
    Returns:
        Formatted distance string
    """
    if meters < 1000:
        return f"{int(meters)} m"
    else:
        kilometers = meters / 1000
        return f"{kilometers:.1f} km"