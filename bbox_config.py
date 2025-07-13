# Bounding Box Configuration for Different Areas
# Format: (north, south, east, west) - latitude/longitude coordinates

# Amsterdam City Center (small area for fast loading)
AMSTERDAM_CENTER = (52.42, 52.34, 4.95, 4.85)

# Amsterdam Metropolitan Area (larger area)
AMSTERDAM_METRO = (52.45, 52.30, 5.05, 4.75)

# Haarlem Area
HAARLEM = (52.40, 52.35, 4.70, 4.60)

# Utrecht City Center
UTRECHT = (52.12, 52.05, 5.15, 5.05)

# The Hague City Center
THE_HAGUE = (52.08, 52.02, 4.35, 4.25)

# Rotterdam City Center
ROTTERDAM = (51.95, 51.88, 4.55, 4.45)

# Leiden City Center
LEIDEN = (52.18, 52.12, 4.52, 4.46)

# Delft City Center
DELFT = (52.02, 51.96, 4.40, 4.34)

# Gouda City Center
GOUDA = (52.03, 51.97, 4.75, 4.69)

# Default to Amsterdam Center for smaller, faster loading
DEFAULT_BBOX = AMSTERDAM_CENTER

def get_bbox_by_name(area_name):
    """
    Get bounding box by area name
    """
    bbox_map = {
        'amsterdam_center': AMSTERDAM_CENTER,
        'amsterdam_metro': AMSTERDAM_METRO,
        'haarlem': HAARLEM,
        'utrecht': UTRECHT,
        'the_hague': THE_HAGUE,
        'rotterdam': ROTTERDAM,
        'leiden': LEIDEN,
        'delft': DELFT,
        'gouda': GOUDA,
    }
    
    return bbox_map.get(area_name.lower(), DEFAULT_BBOX)

def print_available_areas():
    """
    Print all available areas and their bounding boxes
    """
    areas = {
        'amsterdam_center': 'Amsterdam City Center (small, fast)',
        'amsterdam_metro': 'Amsterdam Metropolitan Area (medium)',
        'haarlem': 'Haarlem',
        'utrecht': 'Utrecht',
        'the_hague': 'The Hague',
        'rotterdam': 'Rotterdam',
        'leiden': 'Leiden',
        'delft': 'Delft',
        'gouda': 'Gouda',
    }
    
    print("Available areas:")
    for name, description in areas.items():
        bbox = get_bbox_by_name(name)
        print(f"  {name}: {description}")
        print(f"    BBox: {bbox}")
        print() 