import json
import re

def parse_wall_position(wall_str):
    """Parse the wall position string into coordinates."""
    # Extract wall name and coordinates
    parts = wall_str.split(": ")
    if len(parts) != 2:
        return None, None
    
    wall_name = parts[0]
    coords_str = parts[1]
    
    # Extract the coordinates
    coords = re.findall(r'\(([\d.]+),\s*([\d.]+)\)', coords_str)
    
    if len(coords) == 2:
        start = (float(coords[0][0]), float(coords[0][1]))
        end = (float(coords[1][0]), float(coords[1][1]))
        return wall_name, [start, end]
    return None, None

def map_coordinates(coord, original_resolution, new_bbox):
    """
    Map coordinates from the original space to the new bounding box without stretching.
    Preserves the original scale/aspect ratio.
    
    Args:
        coord: Tuple (x, y) of the original coordinate
        original_resolution: Tuple (width, height) of the original resolution
        new_bbox: Dictionary with new bounding box coordinates
    
    Returns:
        Tuple (x, y) of the mapped coordinate
    """
    x, y = coord
    
    # Calculate original width and height
    original_width, original_height = original_resolution
    
    # Calculate the dimensions of the new bounding box
    new_width = new_bbox["top_right"][0] - new_bbox["top_left"][0]
    new_height = new_bbox["bottom_left"][1] - new_bbox["top_left"][1]
    
    # Calculate the scaling factor while preserving aspect ratio
    # Use the smaller scaling factor to ensure the entire floorplan fits
    scale_x = new_width / original_width
    scale_y = new_height / original_height
    scale = min(scale_x, scale_y)
    
    # Calculate the scaled dimensions
    scaled_width = original_width * scale
    scaled_height = original_height * scale
    
    # Calculate the offset to center the floorplan in the new bounding box
    offset_x = new_bbox["top_left"][0] + (new_width - scaled_width) / 2
    offset_y = new_bbox["top_left"][1] + (new_height - scaled_height) / 2
    
    # Map the coordinates using the consistent scale and offset
    new_x = offset_x + x * scale
    new_y = offset_y + y * scale
    
    return (new_x, new_y)

def map_floorplan_to_new_bbox(floorplan_data, original_resolution, new_bbox):
    """
    Map the entire floorplan to the new bounding box.
    
    Args:
        floorplan_data: The original floorplan data (dictionary)
        original_resolution: Tuple (width, height) of the original resolution
        new_bbox: Dictionary with new bounding box coordinates
    
    Returns:
        Dictionary with the mapped floorplan data
    """
    # Create a deep copy of the floorplan data to modify
    mapped_floorplan = {}
    
    # Map external walls
    mapped_floorplan["external_wall_position"] = []
    for wall_str in floorplan_data["external_wall_position"]:
        wall_name, wall_coords = parse_wall_position(wall_str)
        
        if wall_coords:
            start_mapped = map_coordinates(wall_coords[0], original_resolution, new_bbox)
            end_mapped = map_coordinates(wall_coords[1], original_resolution, new_bbox)
            
            mapped_wall_str = f"{wall_name}: ({start_mapped[0]:.1f}, {start_mapped[1]:.1f}) to ({end_mapped[0]:.1f}, {end_mapped[1]:.1f})"
            mapped_floorplan["external_wall_position"].append(mapped_wall_str)
    # Map windows
    mapped_floorplan["slab_position"] = []
    for slab_pos in floorplan_data["slab_position"]:
        mapped_slab = map_coordinates(slab_pos, original_resolution, new_bbox)
        mapped_floorplan["slab_position"].append([round(mapped_slab[0], 1), round(mapped_slab[1], 1)])
    
    # Map internal walls
    mapped_floorplan["internal_wall_position"] = []
    for wall_str in floorplan_data["internal_wall_position"]:
        wall_name, wall_coords = parse_wall_position(wall_str)
        
        if wall_coords:
            start_mapped = map_coordinates(wall_coords[0], original_resolution, new_bbox)
            end_mapped = map_coordinates(wall_coords[1], original_resolution, new_bbox)
            
            mapped_wall_str = f"{wall_name}: ({start_mapped[0]:.1f}, {start_mapped[1]:.1f}) to ({end_mapped[0]:.1f}, {end_mapped[1]:.1f})"
            mapped_floorplan["internal_wall_position"].append(mapped_wall_str)
    
    # Map doors
    mapped_floorplan["doors_position"] = []
    for door_pos in floorplan_data["doors_position"]:
        mapped_door = map_coordinates(door_pos, original_resolution, new_bbox)
        mapped_floorplan["doors_position"].append([round(mapped_door[0], 1), round(mapped_door[1], 1)])
    
    # Map windows
    mapped_floorplan["windows_position"] = []
    for window_pos in floorplan_data["windows_position"]:
        mapped_window = map_coordinates(window_pos, original_resolution, new_bbox)
        mapped_floorplan["windows_position"].append([round(mapped_window[0], 1), round(mapped_window[1], 1)])
        

    
    return mapped_floorplan
