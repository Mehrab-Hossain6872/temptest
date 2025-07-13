# backend_main.py - Enhanced backend with debugging and error handling

import logging
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import our router module
try:
    from backend_router import (
        get_multimodal_route, 
        get_graph_stats, 
        debug_node_connectivity,
        load_and_fix_graph
    )
    logger.info("Successfully imported backend_router module")
except ImportError as e:
    logger.error(f"Failed to import backend_router: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# Global variables for debugging
DEBUG_MODE = True
ROUTE_CACHE = {}

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        return send_from_directory('.', 'index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return f"Error: {e}", 500

@app.route('/debug')
def debug_info():
    """Debug endpoint to check system status"""
    try:
        # Get graph statistics
        stats = get_graph_stats()
        
        # System info
        debug_data = {
            'system_status': 'OK',
            'debug_mode': DEBUG_MODE,
            'graph_stats': stats,
            'cache_entries': len(ROUTE_CACHE),
            'python_version': sys.version,
            'working_directory': os.getcwd()
        }
        
        return jsonify(debug_data)
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/test-connectivity')
def test_connectivity():
    """Test connectivity between specific nodes"""
    try:
        node1 = request.args.get('node1', '65216_walk')
        node2 = request.args.get('node2', '65223_walk')
        
        graph = load_and_fix_graph()
        
        # Debug both nodes
        node1_info = debug_node_connectivity(graph, node1)
        node2_info = debug_node_connectivity(graph, node2)
        
        # Check if path exists
        import networkx as nx
        has_path = nx.has_path(graph, node1, node2) if node1 in graph and node2 in graph else False
        
        result = {
            'node1_info': node1_info,
            'node2_info': node2_info,
            'has_path': has_path,
            'graph_connected': nx.is_connected(graph),
            'components': len(list(nx.connected_components(graph)))
        }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Connectivity test error: {e}")
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/route')
def calculate_route():
    """Calculate multimodal route between two points"""
    try:
        # Get parameters
        start_lat = float(request.args.get('start_lat'))
        start_lon = float(request.args.get('start_lon'))
        end_lat = float(request.args.get('end_lat'))
        end_lon = float(request.args.get('end_lon'))
        
        logger.info(f"Calculating route from ({start_lat}, {start_lon}) to ({end_lat}, {end_lon})")
        
        # Create cache key
        cache_key = f"{start_lat:.6f},{start_lon:.6f}_{end_lat:.6f},{end_lon:.6f}"
        
        # Check cache
        if cache_key in ROUTE_CACHE:
            logger.info("Returning cached route")
            return jsonify(ROUTE_CACHE[cache_key])
        
        # Calculate route
        route = get_multimodal_route(start_lat, start_lon, end_lat, end_lon)
        
        if route:
            # Cache the result
            ROUTE_CACHE[cache_key] = route
            logger.info(f"Route calculated successfully: {route.get('total_distance', 0):.1f}m")
            return jsonify(route)
        else:
            logger.error("Route calculation returned None")
            return jsonify({'error': 'No route found'}), 404
            
    except ValueError as e:
        logger.error(f"Invalid parameters: {e}")
        return jsonify({'error': 'Invalid coordinates provided'}), 400
    except Exception as e:
        logger.error(f"Error calculating route: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return detailed error in debug mode
        if DEBUG_MODE:
            return jsonify({
                'error': str(e),
                'traceback': traceback.format_exc(),
                'parameters': {
                    'start_lat': request.args.get('start_lat'),
                    'start_lon': request.args.get('start_lon'),
                    'end_lat': request.args.get('end_lat'),
                    'end_lon': request.args.get('end_lon')
                }
            }), 500
        else:
            return jsonify({'error': 'Internal server error'}), 500

@app.route('/clear-cache')
def clear_cache():
    """Clear the route cache"""
    global ROUTE_CACHE
    cache_size = len(ROUTE_CACHE)
    ROUTE_CACHE.clear()
    logger.info(f"Cleared {cache_size} cached routes")
    return jsonify({'message': f'Cleared {cache_size} cached routes'})

@app.route('/graph-info')
def graph_info():
    """Get detailed graph information"""
    try:
        stats = get_graph_stats()
        
        # Add more detailed info
        graph = load_and_fix_graph()
        import networkx as nx
        
        components = list(nx.connected_components(graph))
        component_sizes = [len(comp) for comp in components]
        
        detailed_info = {
            **stats,
            'component_sizes': sorted(component_sizes, reverse=True),
            'is_connected': nx.is_connected(graph),
            'average_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
        }
        
        return jsonify(detailed_info)
    except Exception as e:
        logger.error(f"Graph info error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/find-nearest')
def find_nearest():
    """Find nearest nodes to a given point"""
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        max_distance = int(request.args.get('max_distance', 1000))
        
        from backend_router import find_nearest_nodes_all_modes
        graph = load_and_fix_graph()
        
        nearest_nodes = find_nearest_nodes_all_modes(graph, lat, lon, max_distance)
        
        # Get detailed info for each node
        node_details = []
        for node_id in nearest_nodes[:10]:  # Limit to 10 nodes
            if node_id in graph:
                node_data = graph.nodes[node_id]
                from geopy.distance import geodesic
                distance = geodesic((lat, lon), (node_data['y'], node_data['x'])).meters
                
                node_details.append({
                    'id': node_id,
                    'lat': node_data['y'],
                    'lon': node_data['x'],
                    'mode': node_data.get('mode', 'unknown'),
                    'distance': round(distance, 2),
                    'neighbors': len(list(graph.neighbors(node_id)))
                })
        
        return jsonify({
            'requested_point': [lat, lon],
            'max_distance': max_distance,
            'nodes_found': len(nearest_nodes),
            'node_details': node_details
        })
        
    except Exception as e:
        logger.error(f"Find nearest error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/force-reconnect')
def force_reconnect():
    """Force reconnection of graph components"""
    try:
        from graph_connectivity_fixer import fix_graph_connectivity, add_intermodal_connections
        
        graph = load_and_fix_graph()
        
        # Force reconnection
        edges_added = fix_graph_connectivity(graph, max_connection_distance=5000)
        intermodal_edges = add_intermodal_connections(graph, max_transfer_distance=1000)
        
        # Get updated stats
        import networkx as nx
        components = list(nx.connected_components(graph))
        
        result = {
            'edges_added': edges_added,
            'intermodal_edges_added': intermodal_edges,
            'components_after': len(components),
            'is_connected': nx.is_connected(graph),
            'largest_component': len(max(components, key=len)) if components else 0
        }
        
        logger.info(f"Force reconnect completed: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Force reconnect error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting enhanced backend server...")
    
    # Test graph loading on startup
    try:
        stats = get_graph_stats()
        logger.info(f"Graph loaded successfully: {stats}")
        
        # Test connectivity
        graph = load_and_fix_graph()
        import networkx as nx
        if not nx.is_connected(graph):
            logger.warning("Graph is not fully connected - some routes may fail")
            components = list(nx.connected_components(graph))
            logger.info(f"Graph has {len(components)} components")
            
    except Exception as e:
        logger.error(f"Graph loading test failed: {e}")
        logger.error("Server may not work properly without a valid graph")
    
    # Start server
    app.run(host='0.0.0.0', port=5000, debug=DEBUG_MODE)