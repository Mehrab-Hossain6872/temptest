import os
import networkx as nx
from shapely.geometry import Point
from geopy.distance import geodesic
import logging
from pyrosm import OSM
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, box
from scipy.spatial import cKDTree
import numpy as np

logger = logging.getLogger(__name__)

class MultimodalGraphBuilder:
    """
    Builds or loads a multimodal graph combining walking, biking, and driving networks
    """
    
    def __init__(self, place_or_bbox=None, walk_speed=5, bike_speed=15, car_speed=40, graphml_path=None, osm_file=None, network_type='drive', bbox=None):
        """
        Initialize the multimodal graph builder
        Args:
            place_or_bbox: Either a place name string or bbox tuple (north, south, east, west)
            walk_speed: Walking speed in km/h (default: 5)
            bike_speed: Biking speed in km/h (default: 15)
            car_speed: Car speed in km/h (default: 40)
            graphml_path: Path to pre-downloaded GraphML file (optional)
            osm_file: Path to a local OSM file (.osm.pbf or .osm.xml) (optional)
            network_type: 'drive', 'walk', 'bike', etc. (default: 'drive')
            bbox: Bounding box tuple (north, south, east, west) to filter data (optional)
        """
        self.place_or_bbox = place_or_bbox
        self.walk_speed = walk_speed
        self.bike_speed = bike_speed
        self.car_speed = car_speed
        self.graph = None
        self.graphml_path = graphml_path
        self.osm_file = osm_file
        self.network_type = network_type
        self.bbox = bbox  # (north, south, east, west)

    def build(self):
        """
        Build or load the complete multimodal graph
        Returns:
            nx.MultiDiGraph: The complete multimodal graph
        """
        # 1. Load from GraphML if available
        if self.graphml_path and os.path.exists(self.graphml_path):
            logger.info(f"Loading graph from {self.graphml_path} ...")
            self.graph = nx.read_graphml(self.graphml_path)
            logger.info(f"Graph loaded from {self.graphml_path}: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
            return self.graph
        
        # 2. Load from OSM PBF file using Pyrosm
        if self.osm_file and os.path.exists(self.osm_file):
            logger.info(f"Loading graph from OSM PBF file: {self.osm_file} ...")
            if self.bbox:
                logger.info(f"Filtering data to bounding box: {self.bbox}")
            
            # Extract networks for each mode
            walk_graph = self._extract_graph('walking')
            bike_graph = self._extract_graph('cycling')
            car_graph = self._extract_graph('driving')
        
            # Relabel nodes to make them unique per mode
            walk_graph = self._relabel_nodes(walk_graph, 'walk')
            bike_graph = self._relabel_nodes(bike_graph, 'bike')
            car_graph = self._relabel_nodes(car_graph, 'car')
        
            # Add mode attributes and calculate travel times
            self._add_mode_attributes(walk_graph, 'walk', self.walk_speed)
            self._add_mode_attributes(bike_graph, 'bike', self.bike_speed)
            self._add_mode_attributes(car_graph, 'car', self.car_speed)
        
            # Merge all graphs
            logger.info("Merging individual mode graphs...")
            merged_graph = nx.compose_all([walk_graph, bike_graph, car_graph])
        
            # Add interlayer transfer edges
            self._add_interlayer_edges(merged_graph, walk_graph, bike_graph, car_graph)
        
            self.graph = merged_graph
            logger.info(f"Multimodal graph built successfully: {len(merged_graph.nodes)} nodes, {len(merged_graph.edges)} edges")

            # Save the graph if a path is provided
            if self.graphml_path:
                logger.info(f"Saving graph to {self.graphml_path} ...")
                # Clean the graph before saving
                cleaned_graph = self._clean_graph_for_graphml(self.graph)
                nx.write_graphml(cleaned_graph, self.graphml_path)
                logger.info(f"Graph saved to {self.graphml_path}")
        
            return merged_graph
    
    def _filter_by_bbox(self, gdf):
        """
        Filter GeoDataFrame by bounding box if specified
        """
        if self.bbox is None:
            return gdf
        
        north, south, east, west = self.bbox
        bbox_geom = box(west, south, east, north)
        
        # Filter geometries that intersect with the bounding box
        filtered_gdf = gdf[gdf.geometry.intersects(bbox_geom)]
        
        logger.info(f"Filtered from {len(gdf)} to {len(filtered_gdf)} features using bbox {self.bbox}")
        return filtered_gdf
    
    def _clean_graph_for_graphml(self, graph):
        """
        Clean the graph by removing None values and converting problematic data types
        """
        logger.info("Cleaning graph data for GraphML export...")
        
        # Create a copy to avoid modifying the original
        cleaned_graph = graph.copy()
        
        # Clean node attributes
        for node_id, data in cleaned_graph.nodes(data=True):
            cleaned_data = {}
            for key, value in data.items():
                if value is None:
                    # Replace None with appropriate default values
                    if key in ['x', 'y']:
                        cleaned_data[key] = 0.0
                    elif key in ['name']:
                        cleaned_data[key] = ""
                    else:
                        cleaned_data[key] = ""
                elif isinstance(value, (int, float, str, bool)):
                    cleaned_data[key] = value
                else:
                    # Convert complex objects to strings
                    cleaned_data[key] = str(value)
            
            # Update node data
            cleaned_graph.nodes[node_id].clear()
            cleaned_graph.nodes[node_id].update(cleaned_data)
        
        # Clean edge attributes
        for u, v, key, data in cleaned_graph.edges(data=True, keys=True):
            cleaned_data = {}
            for attr_key, value in data.items():
                if value is None:
                    # Replace None with appropriate default values
                    if attr_key in ['length', 'time', 'weight']:
                        cleaned_data[attr_key] = 0.0
                    elif attr_key in ['mode', 'highway', 'name']:
                        cleaned_data[attr_key] = ""
                    elif attr_key in ['oneway']:
                        cleaned_data[attr_key] = False
                    else:
                        cleaned_data[attr_key] = ""
                elif isinstance(value, (int, float, str, bool)):
                    cleaned_data[attr_key] = value
                elif hasattr(value, '__geo_interface__'):
                    # Handle geometry objects by converting to WKT
                    cleaned_data[attr_key] = str(value)
                else:
                    # Convert other objects to strings
                    cleaned_data[attr_key] = str(value)
            
            # Update edge data
            cleaned_graph.edges[u, v, key].clear()
            cleaned_graph.edges[u, v, key].update(cleaned_data)
        
        return cleaned_graph
    
    def _extract_graph(self, network_type):
        """
        Extract graph for a specific network type using Pyrosm
        """
        logger.info(f"Extracting {network_type} graph using Pyrosm...")
        try:
            osm = OSM(self.osm_file)
            gdf = osm.get_network(network_type=network_type)
            
            if gdf.empty:
                logger.warning(f"No {network_type} network found in OSM data")
                return nx.MultiDiGraph()
            
            # Apply bounding box filter if specified
            gdf = self._filter_by_bbox(gdf)
            
            if gdf.empty:
                logger.warning(f"No {network_type} network found in specified bounding box")
                return nx.MultiDiGraph()
            
            print(f"GDF columns for {network_type}:", gdf.columns.tolist())
            print(f"GDF shape: {gdf.shape}")
            
            # Create nodes and edges from the geometry
            graph = self._create_graph_from_gdf(gdf)
                
            logger.info(f"{network_type} graph extracted: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
            return graph
            
        except Exception as e:
            logger.error(f"Failed to extract {network_type} graph: {str(e)}")
            # Return empty graph instead of raising error
            return nx.MultiDiGraph()

    def _create_graph_from_gdf(self, gdf):
        """
        Create NetworkX graph from GeoDataFrame by extracting nodes from geometry
        """
        graph = nx.MultiDiGraph()
        
        # Dictionary to store unique nodes
        nodes_dict = {}
        node_id = 0
        
        def get_or_create_node(lat, lon):
            """Get existing node or create new one"""
            nonlocal node_id
            # Round coordinates to avoid floating point issues
            lat_rounded = round(lat, 6)
            lon_rounded = round(lon, 6)
            coord_key = (lat_rounded, lon_rounded)
            
            if coord_key not in nodes_dict:
                nodes_dict[coord_key] = node_id
                graph.add_node(node_id, x=lon_rounded, y=lat_rounded)
                node_id += 1
            
            return nodes_dict[coord_key]
        
        # Process each edge in the GeoDataFrame
        for idx, row in gdf.iterrows():
            try:
                geom = row.geometry
                if geom is None:
                    continue
                
                # Extract coordinates from geometry
                coords = []
                if isinstance(geom, LineString):
                    coords = list(geom.coords)
                elif isinstance(geom, MultiLineString):
                    # Flatten all parts into one list of coordinates
                    for part in geom.geoms:
                        coords.extend(list(part.coords))
                else:
                    continue
                
                if len(coords) < 2:
                    continue
                
                # Get start and end points
                start_lon, start_lat = coords[0]
                end_lon, end_lat = coords[-1]
                
                # Create or get nodes
                start_node = get_or_create_node(start_lat, start_lon)
                end_node = get_or_create_node(end_lat, end_lon)
                
                # Skip self-loops
                if start_node == end_node:
                    continue
                
                # Add edge with attributes, handling None values
                edge_attrs = {
                    'length': self._safe_get_value(row, 'length', self._calculate_length(coords)),
                    'highway': self._safe_get_value(row, 'highway', 'unclassified'),
                    'name': self._safe_get_value(row, 'name', ''),
                    'maxspeed': self._safe_get_value(row, 'maxspeed', ''),
                    'oneway': self._safe_get_value(row, 'oneway', False),
                    # Don't store geometry in GraphML as it's not supported
                }
                
                # Add any other relevant attributes
                for col in ['surface', 'lanes', 'bicycle', 'foot', 'motor_vehicle']:
                    if col in row and pd.notna(row[col]):
                        edge_attrs[col] = self._safe_get_value(row, col, '')
                
                graph.add_edge(start_node, end_node, **edge_attrs)
                
                # Add reverse edge if not oneway
                if not self._is_oneway(row):
                    graph.add_edge(end_node, start_node, **edge_attrs)
                
            except Exception as e:
                logger.warning(f"Error processing edge {idx}: {str(e)}")
                continue
        
        return graph

    def _safe_get_value(self, row, key, default):
        """
        Safely get value from row, handling None and NaN values
        """
        if key in row:
            value = row[key]
            if pd.isna(value) or value is None:
                return default
            return value
        return default

    def _calculate_length(self, coords):
        """Calculate length of a line from coordinates"""
        if len(coords) < 2:
            return 0
        
        total_length = 0
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]
            total_length += geodesic((lat1, lon1), (lat2, lon2)).meters
        
        return total_length

    def _is_oneway(self, row):
        """Check if the way is oneway"""
        oneway = row.get('oneway', False)
        if isinstance(oneway, str):
            return oneway.lower() in ['yes', 'true', '1']
        return bool(oneway)
    
    def _relabel_nodes(self, graph, mode_suffix):
        """
        Relabel nodes to include mode suffix
        """
        if graph.number_of_nodes() == 0:
            return graph
        
        logger.info(f"Relabeling nodes for {mode_suffix} mode...")
        return nx.relabel_nodes(graph, lambda n: f"{n}_{mode_suffix}")
    
    def _add_mode_attributes(self, graph, mode, speed_kmh):
        """
        Add mode and time attributes to edges
        """
        if graph.number_of_edges() == 0:
            return
        
        logger.info(f"Adding attributes for {mode} mode...")
        
        for u, v, key, data in graph.edges(data=True, keys=True):
            # Add mode attribute
            data['mode'] = mode
            
            # Calculate travel time in minutes
            length = data.get('length', 0)
            if length > 0:
                # Convert length from meters to km, then calculate time
                distance_km = length / 1000
                time_hours = distance_km / speed_kmh
                time_minutes = time_hours * 60
                data['time'] = time_minutes
            else:
                # Fallback time
                data['time'] = 1.0
                
            # Set weight for shortest path algorithm
            data['weight'] = data['time']
    
    def _add_interlayer_edges(self, merged_graph, walk_graph, bike_graph, car_graph):
        logger.info("Adding interlayer transfer edges...")

        # Collect all nodes and their positions
        node_positions = {}
        for mode_graph, suffix in [(walk_graph, '_walk'), (bike_graph, '_bike'), (car_graph, '_car')]:
            for node_id, node_data in mode_graph.nodes(data=True):
                if 'y' in node_data and 'x' in node_data:
                    node_positions[node_id] = (node_data['y'], node_data['x'])

        # Build arrays for KDTree
        node_ids = list(node_positions.keys())
        coords = np.array([node_positions[nid] for nid in node_ids])

        # Build KDTree for fast neighbor search
        tree = cKDTree(coords)
        max_transfer_distance = 10  # meters

        transfer_edges_added = 0
        for idx, node1 in enumerate(node_ids):
            pos1 = coords[idx]
            # Find all nodes within max_transfer_distance (including itself)
            indices = tree.query_ball_point(pos1, max_transfer_distance / 111_139)  # ~meters to degrees
            for j in indices:
                node2 = node_ids[j]
                if node1 == node2:
                    continue
                # Only add transfer edges between different modes
                if node1.split('_')[-1] != node2.split('_')[-1]:
                    merged_graph.add_edge(
                        node1, node2,
                        weight=2.0,
                        time=2.0,
                        mode='transfer',
                        length=0
                    )
                    transfer_edges_added += 1

        logger.info(f"Added {transfer_edges_added} transfer edges")
    
    def get_graph_stats(self):
        """
        Get statistics about the built graph
        """
        if self.graph is None:
            return {"error": "Graph not built yet"}
        
        # Count nodes and edges by mode
        mode_stats = {}
        for u, v, data in self.graph.edges(data=True):
            mode = data.get('mode', 'unknown')
            if mode not in mode_stats:
                mode_stats[mode] = 0
            mode_stats[mode] += 1
        
        return {
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
            "edges_by_mode": mode_stats,
            "is_directed": self.graph.is_directed(),
            "is_multigraph": self.graph.is_multigraph()
        }