import json
import re
import os
import numpy as np
import cv2
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tabulate import tabulate

# ================= CONFIG PARAMETERS =================
THRESHOLD_PX = 20   # Max distance threshold for a successful match (pixels)
CANVAS_SIZE = (512, 512)  # Canvas size for IoU calculation
LINE_THICKNESS = 10    # Simulated wall thickness for IoU calculation

# ================= 1. DATA PARSING MODULE =================

def parse_wall_string(wall_str):
    """
    Parse string: "Wall3: (91, 88) to (400, 88)" -> ((91, 88), (400, 88))
    """
    pattern = r"\(([\d\.]+),\s*([\d\.]+)\).*?\(([\d\.]+),\s*([\d\.]+)\)"
    match = re.search(pattern, wall_str)
    if match:
        coords = list(map(float, match.groups()))
        return ((coords[0], coords[1]), (coords[2], coords[3]))
    return None

def load_data_from_json(json_path):
    """
    Read specific JSON format, extracting walls (lines) and doors/windows (points)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract walls (merge external and internal)
    walls = []
    if 'external_wall_position' in data:
        for w in data['external_wall_position']:
            parsed = parse_wall_string(w)
            if parsed: walls.append(parsed)
    if 'internal_wall_position' in data:
        for w in data['internal_wall_position']:
            parsed = parse_wall_string(w)
            if parsed: walls.append(parsed)
            
    # Extract doors and windows (points)
    windows = data.get('windows_position', [])
    doors = data.get('doors_position', [])
    
    # Ensure points are tuple format (x, y)
    windows = [tuple(p) for p in windows]
    doors = [tuple(p) for p in doors]
    
    return walls, windows, doors

# ================= 1.5 ALIGNMENT MODULE (NEW) =================

def get_points_from_walls(walls):
    """Flatten wall start/end points into a list of (x,y) coordinates."""
    points = []
    for w in walls:
        points.append(w[0]) # Start point
        points.append(w[1]) # End point
    return np.array(points)

def align_layout(pred_walls, pred_windows, pred_doors, gt_walls):
    """
    Calculates the translation shift required to move the center of 
    prediction walls to the center of GT walls to solve bias issues.
    """
    # Safety check: if no walls, cannot align
    if not pred_walls or not gt_walls:
        return pred_walls, pred_windows, pred_doors

    # 1. Get all coordinates from walls to find the "Center of Mass"
    p_points = get_points_from_walls(pred_walls)
    g_points = get_points_from_walls(gt_walls)
    
    if len(p_points) == 0 or len(g_points) == 0:
        return pred_walls, pred_windows, pred_doors

    # 2. Calculate Centroids (Mean X, Mean Y)
    p_center = np.mean(p_points, axis=0)
    g_center = np.mean(g_points, axis=0)

    # 3. Calculate the shift vector (Translation bias)
    shift = g_center - p_center
    
    # Only print if shift is significant (> 1 pixel)
    if abs(shift[0]) > 1 or abs(shift[1]) > 1:
        print(f"   -> Auto-aligning: shifting prediction by X={shift[0]:.2f}, Y={shift[1]:.2f}")

    # 4. Apply shift to Walls
    aligned_walls = []
    for w in pred_walls:
        # w is ((x1, y1), (x2, y2))
        start = (w[0][0] + shift[0], w[0][1] + shift[1])
        end   = (w[1][0] + shift[0], w[1][1] + shift[1])
        aligned_walls.append((start, end))

    # 5. Apply same shift to Windows and Doors (Points)
    aligned_windows = [(p[0] + shift[0], p[1] + shift[1]) for p in pred_windows]
    aligned_doors   = [(p[0] + shift[0], p[1] + shift[1]) for p in pred_doors]

    return aligned_walls, aligned_windows, aligned_doors

# ================= 2. CORE CALCULATION MODULE =================

def calculate_iou(pred_walls, gt_walls, size=CANVAS_SIZE):
    """Calculate Wall Intersection over Union (IoU)"""
    canvas_pred = np.zeros(size, dtype=np.uint8)
    canvas_gt = np.zeros(size, dtype=np.uint8)
    
    for p in pred_walls:
        cv2.line(canvas_pred, (int(p[0][0]), int(p[0][1])), (int(p[1][0]), int(p[1][1])), 255, LINE_THICKNESS)
    for g in gt_walls:
        cv2.line(canvas_gt, (int(g[0][0]), int(g[0][1])), (int(g[1][0]), int(g[1][1])), 255, LINE_THICKNESS)
        
    intersection = np.logical_and(canvas_pred, canvas_gt).sum()
    union = np.logical_or(canvas_pred, canvas_gt).sum()
    
    return intersection / union if union > 0 else 0

def evaluate_walls(pred_walls, gt_walls, threshold=THRESHOLD_PX):
    m, n = len(pred_walls), len(gt_walls)
    if m == 0 and n == 0: return 1.0, 1.0, 1.0, 0.0
    if m == 0 or n == 0: return 0.0, 0.0, 0.0, 0.0

    # 1. Build Cost Matrix
    cost_matrix = np.zeros((m, n))
    for i, p in enumerate(pred_walls):
        for j, g in enumerate(gt_walls):
            p_s, p_e = np.array(p[0]), np.array(p[1])
            g_s, g_e = np.array(g[0]), np.array(g[1])
            
            # Check both directions (A->B vs B->A)
            d1 = np.linalg.norm(p_s - g_s) + np.linalg.norm(p_e - g_e)
            d2 = np.linalg.norm(p_s - g_e) + np.linalg.norm(p_e - g_s)
            cost_matrix[i, j] = min(d1, d2) / 2.0

    # 2. Hungarian Algorithm Matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 3. Calculate True Positives (TP)
    tp = 0
    total_epe = 0
    for r, c in zip(row_ind, col_ind):
        dist = cost_matrix[r, c]
        if dist < threshold:
            tp += 1
            total_epe += dist
            
    fp = m - tp
    fn = n - tp
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    epe = total_epe / tp if tp > 0 else 0
    
    return prec, rec, f1, epe

def evaluate_points(pred_points, gt_points, threshold=THRESHOLD_PX):
    m, n = len(pred_points), len(gt_points)
    if m == 0 and n == 0: return 1.0, 1.0, 1.0, 0.0
    if m == 0 or n == 0: return 0.0, 0.0, 0.0, 0.0

    cost_matrix = np.zeros((m, n))
    for i, p in enumerate(pred_points):
        for j, g in enumerate(gt_points):
            cost_matrix[i, j] = np.linalg.norm(np.array(p) - np.array(g))
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    tp = 0
    total_dist = 0
    for r, c in zip(row_ind, col_ind):
        dist = cost_matrix[r, c]
        if dist < threshold:
            tp += 1
            total_dist += dist
            
    fp = m - tp
    fn = n - tp
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    avg_dist = total_dist / tp if tp > 0 else 0
    
    return prec, rec, f1, avg_dist

# ================= 3. MAIN EVALUATION FUNCTION =================

def run_evaluation(pred_json_path, gt_json_path, file_id):
    """
    Runs evaluation for a single pair of files and returns a list of result dictionaries.
    """
    print(f"[{file_id}] Processing...")
    
    try:
        p_walls, p_wins, p_doors = load_data_from_json(pred_json_path)
        g_walls, g_wins, g_doors = load_data_from_json(gt_json_path)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return []
    
    # --- STEP 2: ALIGNMENT / REGISTRATION ---
    # Shift Prediction to match GT based on wall centroids
    # This fixes the bias problem (e.g., (100,100) vs (200,200))
    p_walls, p_wins, p_doors = align_layout(p_walls, p_wins, p_doors, g_walls)
    
    # --- STEP 3: CALCULATE METRICS ---
    w_prec, w_rec, w_f1, w_epe = evaluate_walls(p_walls, g_walls)
    w_iou = calculate_iou(p_walls, g_walls)
    
    d_prec, d_rec, d_f1, d_dist = evaluate_points(p_doors, g_doors)
    win_prec, win_rec, win_f1, win_dist = evaluate_points(p_wins, g_wins)
    
    # Return structured data for Excel
    results = [
        {
            "File ID": file_id, "Category": "Walls", 
            "Precision": w_prec, "Recall": w_rec, "F1-Score": w_f1, 
            "IoU": w_iou, "Geo Error (px)": w_epe
        },
        {
            "File ID": file_id, "Category": "Doors", 
            "Precision": d_prec, "Recall": d_rec, "F1-Score": d_f1, 
            "IoU": None, "Geo Error (px)": d_dist
        },
        {
            "File ID": file_id, "Category": "Windows", 
            "Precision": win_prec, "Recall": win_rec, "F1-Score": win_f1, 
            "IoU": None, "Geo Error (px)": win_dist
        }
    ]
    return results

# ================= ENTRY POINT =================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_ROOT = os.path.join(BASE_DIR, "floorplan_understanding")
GT_ROOT = os.path.join(BASE_DIR, "GT")
FLOORS = ["1floor", "2floor", "3floor"]
NUM_CASES = 15

if __name__ == "__main__":

    all_evaluation_results = []

    for floor in FLOORS:
        for i in range(1, NUM_CASES + 1):
            case = f"cubicasa{i}"
            file_identifier = f"{floor}/{case}"

            PRED_FILE = os.path.join(PRED_ROOT, floor, case, "working_process_data.json")
            GT_FILE = os.path.join(GT_ROOT, floor, f"ground_truth_{case}_aligned_updated.json")

            if os.path.exists(PRED_FILE) and os.path.exists(GT_FILE):
                file_results = run_evaluation(PRED_FILE, GT_FILE, file_identifier)
                all_evaluation_results.extend(file_results)
            else:
                print(f"Skipping {file_identifier}: prediction or GT JSON not found.")

# ================= STATS & EXCEL SAVING =================
if all_evaluation_results:
    output_excel = os.path.join(BASE_DIR, "evaluation_results.xlsx")
    
    # 1. Create Initial DataFrame
    df = pd.DataFrame(all_evaluation_results)
    
    # 2. Calculate Averages by Category
    numeric_cols = ["Precision", "Recall", "F1-Score", "IoU", "Geo Error (px)"]
    
    # Group by Category and get the mean
    averages = df.groupby("Category")[numeric_cols].mean().reset_index()
    averages["File ID"] = "AVERAGE" # Mark these rows as AVERAGE
    
    # 3. Print Averages to Console
    print("\n" + "="*40)
    print("FINAL AVERAGE SCORES (Aligned)")
    print("="*40)
    print(tabulate(averages, headers="keys", tablefmt="grid", floatfmt=".3f"))
    
    # 4. Combine Raw Data and Averages for Excel
    df_final = pd.concat([df, averages], ignore_index=True)
    
    # Reorder columns
    cols = ["File ID", "Category", "Precision", "Recall", "F1-Score", "IoU", "Geo Error (px)"]
    df_final = df_final[cols]
    
    # 5. Save to Excel
    print(f"\nSaving results to {output_excel}...")
    df_final.to_excel(output_excel, index=False)
    print("Success! Open the Excel file to see raw data and averages at the bottom.")
    
else:
    print("No results generated.")