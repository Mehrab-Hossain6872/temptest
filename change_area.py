#!/usr/bin/env python3
"""
Utility script to change the area being loaded in the backend
"""

import sys
import os
from bbox_config import get_bbox_by_name, print_available_areas

def change_area(area_name):
    """
    Change the area in backend_main.py
    """
    if not area_name:
        print("Error: Please provide an area name")
        print_available_areas()
        return False
    
    # Check if area exists
    bbox = get_bbox_by_name(area_name)
    if bbox is None:
        print(f"Error: Area '{area_name}' not found")
        print_available_areas()
        return False
    
    # Read the current backend_main.py
    main_file = "backend_main.py"
    if not os.path.exists(main_file):
        print(f"Error: {main_file} not found")
        return False
    
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Replace the area name
    import re
    pattern = r"area_name = \"[^\"]+\""
    replacement = f'area_name = "{area_name}"'
    
    if re.search(pattern, content):
        new_content = re.sub(pattern, replacement, content)
        
        # Write back to file
        with open(main_file, 'w') as f:
            f.write(new_content)
        
        print(f"Successfully changed area to: {area_name}")
        print(f"Bounding box: {bbox}")
        print("\nTo apply changes, restart your backend server.")
        return True
    else:
        print("Error: Could not find area_name line in backend_main.py")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python change_area.py <area_name>")
        print("\nAvailable areas:")
        print_available_areas()
        return
    
    area_name = sys.argv[1]
    change_area(area_name)

if __name__ == "__main__":
    main() 