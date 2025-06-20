import json
import os
import numpy as np
import matplotlib.pyplot as plt
from conf.config import Config
from collections import defaultdict
import math

config = Config()

# Geometry helper functions
def distance(p, q):
    """Compute Euclidean distance between two points."""
    return np.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)

def project_point_to_segment(P, A, B):
    """Project point P onto the line segment AB.
       If the projection falls outside AB, return the closest endpoint.
    """
    Ax, Ay = A
    Bx, By = B
    Px, Py = P
    AB = (Bx - Ax, By - Ay)
    AB_sq = AB[0]**2 + AB[1]**2
    if AB_sq == 0:
        return A
    t = ((Px - Ax) * AB[0] + (Py - Ay) * AB[1]) / AB_sq
    if t < 0:
        return A
    elif t > 1:
        return B
    else:
        return (Ax + t * AB[0], Ay + t * AB[1])

def project_point_to_line(P, A, B):
    """
    Project point P onto the infinite line defined by points A and B.
    Returns the projected point.
    """
    Ax, Ay = A
    Bx, By = B
    Px, Py = P
    ABx, ABy = Bx - Ax, By - Ay
    APx, APy = Px - Ax, Py - Ay
    AB_sq = ABx**2 + ABy**2
    if AB_sq == 0:
        return A
    t = (APx * ABx + APy * ABy) / AB_sq
    proj = (Ax + t * ABx, Ay + t * ABy)
    return proj

# Parsing and basic geometric functions
def parse_coordinate_string(coord_str):
    """Parse a wall position string into start and end coordinates."""
    parts = coord_str.split(": ")[1].replace("(", "").replace(")", "").split(" to ")
    start = tuple(map(float, parts[0].split(", ")))
    end = tuple(map(float, parts[1].split(", ")))
    return start, end

def get_wall_id(coord_str):
    """Extract wall ID from a wall position string."""
    return coord_str.split(":")[0].strip()

def is_close(p1, p2, tolerance=2.0):
    """Check if two points are close enough to be considered the same."""
    return distance(p1, p2) < tolerance

def is_horizontal(start, end, tolerance=2.0):
    """Check if a wall is approximately horizontal."""
    return abs(start[1] - end[1]) < tolerance

def is_vertical(start, end, tolerance=2.0):
    """Check if a wall is approximately vertical."""
    return abs(start[0] - end[0]) < tolerance
def align_coordinates(walls_data, tolerance=5.0):
    """
    Align coordinates for nearly horizontal and nearly vertical walls.
    
    For walls that are nearly horizontal, the average y coordinate (rounded)
    is used for both endpoints. For walls that are nearly vertical, the average 
    x coordinate (rounded) is used for both endpoints.
    
    Parameters:
      walls_data: List of wall strings.
      tolerance: Tolerance for determining if a wall is nearly horizontal or vertical.
    
    Returns:
      A list of wall strings with adjusted coordinates.
    """
    aligned_walls = []
    for wall_str in walls_data:
        wall_id = get_wall_id(wall_str)
        start, end = parse_coordinate_string(wall_str)
        # If nearly horizontal, use the average y coordinate.
        if is_horizontal(start, end, tolerance=tolerance):
            avg_y = round((start[1] + end[1]) / 2.0)
            new_start = (start[0], avg_y)
            new_end = (end[0], avg_y)
        # If nearly vertical, use the average x coordinate.
        elif is_vertical(start, end, tolerance=tolerance):
            avg_x = round((start[0] + end[0]) / 2.0)
            new_start = (avg_x, start[1])
            new_end = (avg_x, end[1])
        else:
            new_start = start
            new_end = end
        aligned_wall = f"{wall_id}: ({new_start[0]:.1f}, {new_start[1]:.1f}) to ({new_end[0]:.1f}, {new_end[1]:.1f})"
        aligned_walls.append(aligned_wall)
    return aligned_walls


def group_by_position(walls, key, tolerance=2.0):
    """Group walls by a position key within tolerance."""
    if not walls:
        return []
    sorted_walls = sorted(walls, key=key)
    groups = []
    current_group = [sorted_walls[0]]
    current_key = key(sorted_walls[0])
    for wall in sorted_walls[1:]:
        if abs(key(wall) - current_key) < tolerance:
            current_group.append(wall)
        else:
            groups.append(current_group)
            current_group = [wall]
            current_key = key(wall)
    groups.append(current_group)
    return groups

def merge_aligned_walls(walls, proximity_threshold=5.0):
    """Merge a group of aligned walls (horizontal or vertical)."""
    if not walls:
        return []
    is_horiz = is_horizontal(walls[0]["start"], walls[0]["end"])
    if is_horiz:
        walls.sort(key=lambda w: min(w["start"][0], w["end"][0]))
    else:
        walls.sort(key=lambda w: min(w["start"][1], w["end"][1]))
    merged = []
    current = walls[0].copy()
    for next_wall in walls[1:]:
        if is_horiz:
            current_xmin = min(current["start"][0], current["end"][0])
            current_xmax = max(current["start"][0], current["end"][0])
            next_xmin = min(next_wall["start"][0], next_wall["end"][0])
            next_xmax = max(next_wall["start"][0], next_wall["end"][0])
            if next_xmin - current_xmax <= proximity_threshold:
                new_xmin = min(current_xmin, next_xmin)
                new_xmax = max(current_xmax, next_xmax)
                y_coord = current["start"][1]
                current["start"] = (new_xmin, y_coord)
                current["end"] = (new_xmax, y_coord)
            else:
                merged.append(current)
                current = next_wall.copy()
        else:
            current_ymin = min(current["start"][1], current["end"][1])
            current_ymax = max(current["start"][1], current["end"][1])
            next_ymin = min(next_wall["start"][1], next_wall["end"][1])
            next_ymax = max(next_wall["start"][1], next_wall["end"][1])
            if next_ymin - current_ymax <= proximity_threshold:
                new_ymin = min(current_ymin, next_ymin)
                new_ymax = max(current_ymax, next_ymax)
                x_coord = current["start"][0]
                current["start"] = (x_coord, new_ymin)
                current["end"] = (x_coord, new_ymax)
            else:
                merged.append(current)
                current = next_wall.copy()
    merged.append(current)
    return merged
def merge_collinear_walls(walls_data, merge_gap_threshold=10.0):
    """Merge walls that are collinear and adjacent, overlapping, or nearly touching."""
    parsed_walls = []
    for wall_str in walls_data:
        wall_id = get_wall_id(wall_str)
        start, end = parse_coordinate_string(wall_str)
        parsed_walls.append({
            "id": wall_id,
            "start": start,
            "end": end
        })
    # Separate walls by orientation.
    horizontal_walls = [w for w in parsed_walls if is_horizontal(w["start"], w["end"])]
    vertical_walls = [w for w in parsed_walls if is_vertical(w["start"], w["end"])]
    diagonal_walls = [w for w in parsed_walls if not is_horizontal(w["start"], w["end"]) and not is_vertical(w["start"], w["end"])]
    
    # Sort walls for grouping.
    horizontal_walls.sort(key=lambda w: (w["start"][1], min(w["start"][0], w["end"][0])))
    vertical_walls.sort(key=lambda w: (w["start"][0], min(w["start"][1], w["end"][1])))
    
    merged_horizontal = []
    for group in group_by_position(horizontal_walls, key=lambda w: w["start"][1]):
        merged_horizontal.extend(merge_aligned_walls(group, proximity_threshold=merge_gap_threshold))
    
    merged_vertical = []
    for group in group_by_position(vertical_walls, key=lambda w: w["start"][0]):
        merged_vertical.extend(merge_aligned_walls(group, proximity_threshold=merge_gap_threshold))
    
    merged_walls = merged_horizontal + merged_vertical + diagonal_walls
    
    result = []
    for wall in merged_walls:
        wall_str = f"{wall['id']}: ({wall['start'][0]:.1f}, {wall['start'][1]:.1f}) to ({wall['end'][0]:.1f}, {wall['end'][1]:.1f})"
        result.append(wall_str)
    return result
def point_distance(p1, p2):
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def distance_point_to_segment(A, B, P):
    """
    Returns the minimum distance from point P to the finite segment AB.
    """
    # If A == B, just return distance to the point
    ABx = B[0] - A[0]
    ABy = B[1] - A[1]
    length_sq = ABx*ABx + ABy*ABy
    if length_sq < 1e-12:
        return point_distance(A, P)
    
    # Consider the line param t where:
    #   A + t*(B - A)
    # t in [0,1] means P projects inside segment
    # t < 0 or t > 1 means P projects outside the segment
    APx = P[0] - A[0]
    APy = P[1] - A[1]
    t = (APx*ABx + APy*ABy) / length_sq
    
    if t < 0:
        # Closer to A
        return point_distance(A, P)
    elif t > 1:
        # Closer to B
        return point_distance(B, P)
    else:
        # Closer to the projection on AB
        proj_x = A[0] + t*ABx
        proj_y = A[1] + t*ABy
        return point_distance((proj_x, proj_y), P)

def line_length(A, B):
    """Length of the segment from A to B."""
    return point_distance(A, B)

def is_near_duplicate(shortA, shortB, longA, longB, dist_tol):
    """
    Checks if the 'short' segment (shortA->shortB) is very close to
    the 'long' segment (longA->longB). The criteria:
      1) Both endpoints of the short segment are within dist_tol
         of the long segment.
      2) The short segment isn't drastically longer than the long one
         (usually we only remove the short one if it's truly short).
    """
    dA = distance_point_to_segment(longA, longB, shortA)
    dB = distance_point_to_segment(longA, longB, shortB)
    return (dA < dist_tol and dB < dist_tol)

def remove_duplicates(walls_data, tol=1, coord_tol=5, dist_tol=5):
    """
    Remove duplicate walls from the list, including walls that are
    almost (but not fully) overlapping.

    - tol, coord_tol: as before
    - dist_tol: extra tolerance for "near-duplicate" checks
    """
    # --- 1) Parse the walls (same as your original approach) ---
    parsed_walls = []
    for wall_str in walls_data:
        wall_id = get_wall_id(wall_str)
        start, end = parse_coordinate_string(wall_str)
        parsed_walls.append({
            "id": wall_id,
            "start": start,
            "end": end,
            "wall_str": wall_str
        })

    keep = [True] * len(parsed_walls)
    n = len(parsed_walls)
    
    # --- 2) Original overlap pass (same logic you already have) ---
    for i in range(n):
        if not keep[i]:
            continue
        wall_i = parsed_walls[i]
        A, B = wall_i["start"], wall_i["end"]
        
        if is_horizontal(A, B, tolerance=coord_tol):
            x_i1, x_i2 = sorted([A[0], B[0]])
            y_i = (A[1] + B[1]) / 2.0
            for j in range(n):
                if i == j or not keep[i]:
                    continue
                wall_j = parsed_walls[j]
                C, D = wall_j["start"], wall_j["end"]
                if is_horizontal(C, D, tolerance=coord_tol):
                    y_j = (C[1] + D[1]) / 2.0
                    if abs(y_i - y_j) <= coord_tol:
                        x_j1, x_j2 = sorted([C[0], D[0]])
                        # If wall_i is completely inside wall_j, mark wall_i as duplicate
                        if x_j1 - tol <= x_i1 and x_i2 <= x_j2 + tol:
                            keep[i] = False
                            break
        elif is_vertical(A, B, tolerance=coord_tol):
            y_i1, y_i2 = sorted([A[1], B[1]])
            x_i = (A[0] + B[0]) / 2.0
            for j in range(n):
                if i == j or not keep[i]:
                    continue
                wall_j = parsed_walls[j]
                C, D = wall_j["start"], wall_j["end"]
                if is_vertical(C, D, tolerance=coord_tol):
                    x_j = (C[0] + D[0]) / 2.0
                    if abs(x_i - x_j) <= coord_tol:
                        y_j1, y_j2 = sorted([C[1], D[1]])
                        if y_j1 - tol <= y_i1 and y_i2 <= y_j2 + tol:
                            keep[i] = False
                            break
        else:
            # Diagonal case: check near-identical endpoints
            sorted_i = tuple(sorted([A, B]))
            for j in range(n):
                if i == j or not keep[i]:
                    continue
                wall_j = parsed_walls[j]
                C, D = wall_j["start"], wall_j["end"]
                sorted_j = tuple(sorted([C, D]))
                if (abs(sorted_i[0][0] - sorted_j[0][0]) < tol and
                    abs(sorted_i[0][1] - sorted_j[0][1]) < tol and
                    abs(sorted_i[1][0] - sorted_j[1][0]) < tol and
                    abs(sorted_i[1][1] - sorted_j[1][1]) < tol):
                    keep[i] = False
                    break

    # --- 3) Additional pass for near-duplicates ---
    #     If line i is significantly shorter and "hugs" line j, remove i.
    #     (Or vice versa, depending on your preference.)
    for i in range(n):
        if not keep[i]:
            continue
        A, B = parsed_walls[i]["start"], parsed_walls[i]["end"]
        len_i = line_length(A, B)
        for j in range(n):
            if i == j or not keep[i] or not keep[j]:
                continue
            C, D = parsed_walls[j]["start"], parsed_walls[j]["end"]
            len_j = line_length(C, D)

            # Decide which line is "shorter" for removal
            # We only remove the short one if it's hugging the longer line
            if len_i < len_j:
                # If i is short and near j, remove i
                if is_near_duplicate(A, B, C, D, dist_tol):
                    keep[i] = False
                    break
            else:
                # If j is short and near i, remove j
                if is_near_duplicate(C, D, A, B, dist_tol):
                    keep[j] = False

    # --- 4) Return only those walls we keep ---
    result = [parsed_walls[i]["wall_str"] for i in range(n) if keep[i]]
    return result

def connect_walls(walls_data, tolerance=3.0):
    """Connect walls that should meet at corners by snapping endpoints together."""
    parsed_walls = []
    for wall_str in walls_data:
        wall_id = get_wall_id(wall_str)
        start, end = parse_coordinate_string(wall_str)
        parsed_walls.append({
            "id": wall_id,
            "start": start,
            "end": end
        })
    endpoints = []
    for wall in parsed_walls:
        endpoints.append(wall["start"])
        endpoints.append(wall["end"])
    connected_points = {}
    processed = set()
    for i, p1 in enumerate(endpoints):
        if i in processed:
            continue
        cluster = [p1]
        processed.add(i)
        for j, p2 in enumerate(endpoints):
            if j in processed:
                continue
            if is_close(p1, p2, tolerance):
                cluster.append(p2)
                processed.add(j)
        if len(cluster) > 1:
            avg_x = sum(p[0] for p in cluster) / len(cluster)
            avg_y = sum(p[1] for p in cluster) / len(cluster)
            avg_point = (avg_x, avg_y)
            for p in cluster:
                connected_points[p] = avg_point
    updated_walls = []
    for wall in parsed_walls:
        new_start = connected_points.get(wall["start"], wall["start"])
        new_end = connected_points.get(wall["end"], wall["end"])
        updated_wall = f"{wall['id']}: ({new_start[0]:.1f}, {new_start[1]:.1f}) to ({new_end[0]:.1f}, {new_end[1]:.1f})"
        updated_walls.append(updated_wall)
    return updated_walls

def find_super_close_walls(parsed_walls, proximity_threshold=3.0):
    """Merge walls that are extremely close and should be combined (within the same room)."""
    merged_walls = []
    used_indices = set()
    for i, wall1 in enumerate(parsed_walls):
        if i in used_indices:
            continue
        start1, end1 = wall1["start"], wall1["end"]
        merged_group = [wall1]
        used_indices.add(i)
        for j, wall2 in enumerate(parsed_walls):
            if j in used_indices or i == j:
                continue
            start2, end2 = wall2["start"], wall2["end"]
            is_wall1_horiz = is_horizontal(start1, end1)
            is_wall1_vert = is_vertical(start1, end1)
            is_wall2_horiz = is_horizontal(start2, end2)
            is_wall2_vert = is_vertical(start2, end2)
            if (is_wall1_horiz and is_wall2_horiz) or (is_wall1_vert and is_wall2_vert):
                if is_wall1_horiz and abs(start1[1] - start2[1]) < proximity_threshold:
                    x1_min = min(start1[0], end1[0])
                    x1_max = max(start1[0], end1[0])
                    x2_min = min(start2[0], end2[0])
                    x2_max = max(start2[0], end2[0])
                    if (x1_min <= x2_max + proximity_threshold and x2_min <= x1_max + proximity_threshold):
                        merged_group.append(wall2)
                        used_indices.add(j)
                elif is_wall1_vert and abs(start1[0] - start2[0]) < proximity_threshold:
                    y1_min = min(start1[1], end1[1])
                    y1_max = max(start1[1], end1[1])
                    y2_min = min(start2[1], end2[1])
                    y2_max = max(start2[1], end2[1])
                    if (y1_min <= y2_max + proximity_threshold and y2_min <= y1_max + proximity_threshold):
                        merged_group.append(wall2)
                        used_indices.add(j)
        if len(merged_group) > 1:
            if is_horizontal(merged_group[0]["start"], merged_group[0]["end"]):
                y_coord = sum(w["start"][1] for w in merged_group) / len(merged_group)
                x_min = min(min(w["start"][0], w["end"][0]) for w in merged_group)
                x_max = max(max(w["start"][0], w["end"][0]) for w in merged_group)
                merged_wall = {
                    "id": merged_group[0]["id"],
                    "start": (x_min, y_coord),
                    "end": (x_max, y_coord)
                }
            else:
                x_coord = sum(w["start"][0] for w in merged_group) / len(merged_group)
                y_min = min(min(w["start"][1], w["end"][1]) for w in merged_group)
                y_max = max(max(w["start"][1], w["end"][1]) for w in merged_group)
                merged_wall = {
                    "id": merged_group[0]["id"],
                    "start": (x_coord, y_min),
                    "end": (x_coord, y_max)
                }
            merged_walls.append(merged_wall)
        else:
            merged_walls.append(wall1)
    result = []
    for wall in merged_walls:
        wall_str = f"{wall['id']}: ({wall['start'][0]:.1f}, {wall['start'][1]:.1f}) to ({wall['end'][0]:.1f}, {wall['end'][1]:.1f})"
        result.append(wall_str)
    return result
def snap_wall_endpoints(cleaned_data, tolerance=10):
    """
    For each wall endpoint (both start and end) in the single floor plan,
    compare its perpendicular distance to the infinite line of every other wall.
    If that distance is less than tolerance, snap the current endpoint
    to its projection onto that line, but only moving along the direction of the original wall.
    """
    # Build a global list of walls
    global_walls = []
    for wall_str in cleaned_data["walls"]:
        wall_id = get_wall_id(wall_str)
        start, end = parse_coordinate_string(wall_str)
        global_walls.append({
            "id": wall_id,
            "start": start,
            "end": end,
            "original_start": start[:],  # Store original values using slice for tuples
            "original_end": end[:]       # Store original values using slice for tuples
        })
    
    # For each wall, check both endpoints against every other wall's line
    for i, w1 in enumerate(global_walls):
        for endpoint in ["start", "end"]:
            pt = w1[endpoint]
            other_endpoint = "end" if endpoint == "start" else "start"
            original_pt = w1[f"original_{endpoint}"]
            other_pt = w1[other_endpoint]
            
            best_distance = float('inf')
            best_proj_on_wall = pt
            
            for j, w2 in enumerate(global_walls):
                if i == j:
                    continue
                    
                A, B = w2["start"], w2["end"]
                proj = project_point_to_line(pt, A, B)
                d = distance(pt, proj)
                
                if d < best_distance and d < tolerance:
                    # We found a potential snap point, but we need to constrain it
                    # to the direction of the original wall
                    
                    # Calculate the direction vector of the original wall
                    wall_dir = [other_pt[0] - original_pt[0], other_pt[1] - original_pt[1]]
                    wall_length = math.sqrt(wall_dir[0]**2 + wall_dir[1]**2)
                    if wall_length > 0:
                        wall_dir = [wall_dir[0]/wall_length, wall_dir[1]/wall_length]
                    
                    # Vector from original point to projection
                    proj_vector = [proj[0] - original_pt[0], proj[1] - original_pt[1]]
                    
                    # Project this vector onto the wall direction to get the scalar distance
                    scalar_proj = proj_vector[0]*wall_dir[0] + proj_vector[1]*wall_dir[1]
                    
                    # Compute the constrained projection point along the wall
                    constrained_proj = (
                        original_pt[0] + scalar_proj * wall_dir[0],
                        original_pt[1] + scalar_proj * wall_dir[1]
                    )
                    
                    # Calculate how far this is from the original projection
                    dp = distance(proj, constrained_proj)
                    
                    # Only use this projection if it's still close enough
                    if dp < tolerance:
                        best_distance = d
                        best_proj_on_wall = constrained_proj
            
            w1[endpoint] = best_proj_on_wall  # Snap endpoint if within tolerance
    
    # Rebuild the walls using updated endpoints
    updated_walls = []
    for w in global_walls:
        wall_str = f"{w['id']}: ({w['start'][0]:.1f}, {w['start'][1]:.1f}) to ({w['end'][0]:.1f}, {w['end'][1]:.1f})"
        updated_walls.append(wall_str)
    cleaned_data["walls"] = updated_walls
    return cleaned_data
def segment_intersection(A, B, C, D, tol=1e-6):
    """
    Compute the intersection point between two line segments AB and CD.
    Returns the intersection point as (x, y) if they intersect; otherwise, returns None.
    """
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    x4, y4 = D
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < tol:
        return None  # Parallel or collinear
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        # Intersection point along AB
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None

def split_wall_by_intersections(wall, all_walls, tol=1e-6):
    """
    For a given wall (a dict with keys "id", "start", and "end"),
    find all intersection points with every other wall in all_walls.
    If any intersection falls strictly within the wall (i.e. not at endpoints),
    split the wall into multiple segments. Each new segment's id is the original id
    with a suffix (_1, _2, etc.).
    """
    A, B = wall["start"], wall["end"]
    if distance(A, B) < tol:
        return [wall]  # Degenerate wall, return as is.

    intersections = []
    for other in all_walls:
        # Use id comparison rather than "is" in case the same wall appears as a separate dict.
        if wall["id"] == other["id"]:
            continue
        pt = segment_intersection(A, B, other["start"], other["end"], tol)
        if pt is not None:
            # Exclude intersections that are nearly the endpoints of the wall.
            if distance(pt, A) > tol and distance(pt, B) > tol:
                intersections.append(pt)
    
    # Remove duplicate intersection points.
    unique_points = []
    for pt in intersections:
        if not any(distance(pt, q) < tol for q in unique_points):
            unique_points.append(pt)
    
    if not unique_points:
        return [wall]
    
    # Sort the intersection points along the wall based on distance from A.
    unique_points.sort(key=lambda pt: distance(A, pt))
    points = [A] + unique_points + [B]
    
    segments = []
    for i in range(len(points) - 1):
        seg = {
            "id": f"{wall['id']}_{i+1}",
            "start": points[i],
            "end": points[i+1]
        }
        segments.append(seg)
    return segments

def split_all_walls_by_intersections(walls_list, tol=1e-6):
    """
    Given a list of wall dictionaries, repeatedly check each wall for intersections
    with all other walls and split it if necessary. This process is repeated until
    no new splits occur.
    """
    changed = True
    current_walls = walls_list[:]
    while changed:
        changed = False
        new_walls = []
        # For each wall, attempt to split it using the current list of walls.
        for wall in current_walls:
            segments = split_wall_by_intersections(wall, current_walls, tol)
            if len(segments) > 1:
                changed = True
            new_walls.extend(segments)
        current_walls = new_walls
    return current_walls

def split_walls_by_intersection(walls_data, tol=1e-6):
    """
    Given a list of wall strings, split each wall by its intersections with all other walls.
    This function repeatedly splits walls until no additional intersections are found.
    Returns a new list of wall strings.
    
    For example, a wall "Wall1: (100, 100) to (100, 500)" intersected at (100, 250)
    will be split into:
      "Wall1_1: (100.0, 100.0) to (100.0, 250.0)" and
      "Wall1_2: (100.0, 250.0) to (100.0, 500.0)"
    """
    # Parse wall strings into dictionaries.
    parsed_walls = []
    for wall_str in walls_data:
        wall_id = get_wall_id(wall_str)
        start, end = parse_coordinate_string(wall_str)
        parsed_walls.append({
            "id": wall_id,
            "start": start,
            "end": end
        })
    
    # Repeatedly split walls until no further splits occur.
    final_walls = split_all_walls_by_intersections(parsed_walls, tol)
    
    # Convert the resulting wall segments back to string format.
    result = []
    for wall in final_walls:
        wall_str = f"{wall['id']}: ({wall['start'][0]:.1f}, {wall['start'][1]:.1f}) to ({wall['end'][0]:.1f}, {wall['end'][1]:.1f})"
        result.append(wall_str)
    return result

def debug_visualize_step(walls, openings=None, save_dir=None):
    """
    Visualize a list of walls (and optionally openings) for debugging.
    
    Parameters:
      step_name: A string title for the plot.
      walls: A list of wall strings.
      openings: (Optional) A list of [x, y] coordinates.
      save_dir: (Optional) Directory to save the image. Defaults to config.work_dir.
    
    Returns:
      Path to the saved image.
    """
    import matplotlib.pyplot as plt
    import os
    
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Plot each wall
    for wall_str in walls:
        start, end = parse_coordinate_string(wall_str)
        plt.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=2)
        
        # Label the wall by its ID at its midpoint
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        plt.text(mid_x, mid_y, get_wall_id(wall_str), color='blue', fontsize=10, 
                 ha='center', va='center')
    
    # Plot openings if provided
    if openings:
        for op in openings:
            plt.plot(op[0], op[1], 'ro', markersize=6)
    
    # Set plot properties
    #plt.title(step_name)
    plt.axis('equal')
    plt.grid(False)
    plt.gca().invert_yaxis()  # Optional: Invert y-axis if needed
    
    # Save the figure before showing it
    output_name = "cleaned.png"
    
    # Use provided save_dir or default to config.work_dir
    if save_dir is None:
        try:
            save_dir = config.work_dir
        except (NameError, AttributeError):
            save_dir = '.'  # Default to current directory if config.work_dir is not available
    
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    image_path = os.path.join(config.work_dir, output_name)

    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    
    print(f"The processed floorplan image is saved in {image_path}")
    
    # Now display the plot (optional)
    plt.show()
    
    return image_path
def clean_floor_plan_single(data):
    """
    Process the single floor plan (with keys 'walls' and 'openings') and
    visualize each step for debugging.
    
      1. Align, merge, connect, and remove duplicate walls.
      2. Split walls by intersections.
      3. Snap wall endpoints.
      4. Adjust opening positions by projecting them onto the nearest wall if within tolerance.
      5. Final split of walls by intersections and duplicate removal.
    """
    # Step 1: Align, merge, connect, and remove duplicate walls.
    aligned = align_coordinates(data["walls"])

    #debug_visualize_step("After Alignment", aligned)
    
    merged = merge_collinear_walls(aligned)
    #debug_visualize_step("After Merging", merged)

    unique = remove_duplicates(merged)
    #debug_visualize_step("After Duplicate Removal", unique)
    
    # connected = connect_walls(unique, tolerance=20)
    # debug_visualize_step("After Connecting", connected)
    
    # # Step 2: Snap wall endpoints.
    snapped = snap_wall_endpoints({"walls": unique}, tolerance=20)["walls"]
    #debug_visualize_step("After Snapping Endpoints", snapped)


    # Step 3: Split walls by intersections.
    split_walls = split_walls_by_intersection(snapped)
    #debug_visualize_step("After Splitting Walls (Step 2)", split_walls)
    
    # # Step 5: Final split of walls by intersections, then remove duplicates.
    # final_walls = split_walls_by_intersection(snapped)
    unique_final = remove_duplicates(split_walls)
    #debug_visualize_step("After Splitting Walls (Step 2)", unique_final)

    # Step 4: Adjust opening positions.
    adjusted_openings = []
    for op in data.get("openings", []):
        op_pt = tuple(op)
        best_distance = float('inf')
        best_proj = op_pt
        for wall_str in unique_final:
            A, B = parse_coordinate_string(wall_str)
            proj = project_point_to_segment(op_pt, A, B)
            d = distance(op_pt, proj)
            if d < best_distance:
                best_distance = d
                best_proj = proj
        if best_distance < 30:
            adjusted_openings.append([round(best_proj[0], 1), round(best_proj[1], 1)])
        else:
            adjusted_openings.append(list(op_pt))
            
    #debug_visualize_step("After Adjusting Openings", unique_final, adjusted_openings)

    
    image_path = debug_visualize_step(unique_final, adjusted_openings)
    
    cleaned_data = {"walls": unique_final, "openings": adjusted_openings}
    return cleaned_data, image_path



def visualize_floor_plan_single(data, title="Floor Plan"):
    """Visualize the single floor plan with walls (labeled with their wall numbers) and openings."""
    plt.figure(figsize=(10, 10))
    # Plot walls as black lines with wall number labels.
    for wall_str in data["walls"]:
        start, end = parse_coordinate_string(wall_str)
        plt.plot([start[0], end[0]], [start[1], end[1]], color='black', linewidth=2)
        # Calculate midpoint for placing the label.
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        wall_label = get_wall_id(wall_str)
        plt.text(mid_x, mid_y, wall_label, color='blue', fontsize=12, fontweight='bold',
                 horizontalalignment='center', verticalalignment='center')
    # Plot openings as red markers.
    for op in data.get("openings", []):
        plt.plot(op[0], op[1], 'o', color='red', markersize=7)
    plt.title(title)
    plt.grid(False)
    plt.axis('off')
    plt.show()
    image_name = "cleanded_floorplan.png"
    image_path = os.path.join(config.work_dir, image_name)
    plt.savefig(image_path)
    print(f"The processed segmented floorplan image is saved in {image_path}")

    
    return image_path

if __name__ == "__main__":
    # Example input in the new format:
    input_data = {'walls': ['Wall1: (388, 483) to (76, 483)', 'Wall2: (243, 482) to (243, 27)', 'Wall3: (134, 26) to (38, 26)', 'Wall4: (37, 318) to (37, 27)', 'Wall5: (242, 248) to (38, 248)', 'Wall6: (276, 26) to (188, 26)', 'Wall7: (388, 318) to (207, 318)', 'Wall8: (206, 482) to (206, 319)', 'Wall9: (78, 318) to (79, 261)', 'Wall10: (389, 171) to (389, 80)', 'Wall11: (338, 27) to (135, 27)', 'Wall12: (314, 317) to (314, 183)', 'Wall13: (389, 482) to (389, 319)', 'Wall14: (75, 482) to (75, 319)', 'Wall15: (472, 319) to (390, 319)', 'Wall16: (74, 319) to (38, 319)', 'Wall17: (473, 318) to (473, 172)', 'Wall18: (334, 182) to (245, 182)', 'Wall19: (472, 172) to (380, 172)', 'Wall20: (388, 77) to (339, 28)', 'Wall21: (244, 319) to (244, 282)', 'Wall22: (77, 318) to (79, 266)', 'Wall23: (79, 317) to (78, 249)'], 'openings': [[295, 27], [139, 27], [351, 39], [281, 182], [243, 218], [473, 228], [243, 278], [78, 282], [37, 286], [271, 318], [243, 380], [243, 398], [136, 483], [312, 483]]}

    print("Cleaning floor plan data...")
    cleaned = clean_floor_plan_single(input_data)
    
    # print("cleaned floor plan...")
    # visualize_floor_plan_single(cleaned, "Cleaned Floor Plan")
    
    # Optionally, save the cleaned data
    output_file = 'cleaned_floor_plan_data.json'
    with open(output_file, 'w') as f:
        json.dump(cleaned, f, indent=2)
    
    print(f"Cleaned floor plan data saved to '{output_file}'")
