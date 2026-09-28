import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from skimage import morphology
from skimage.transform import probabilistic_hough_line
from PIL import Image
from skimage.measure import label, regionprops
import cv2
from conf.config import Config
from BIMgent.memory.local_memory import LocalMemory
from BIMgent.provider.Deep_fp_provider.utils.floorplan_postprocessing import clean_floor_plan_single
from BIMgent.utils.dict_utils import kget

# Disable eager execution for TF1.x compatibility (and silence its deprecation notice)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()

config = Config()
memory = LocalMemory()

model_path = kget(config.env_config, "models_path", default={}).get('deep_floorplan', '')


class DeepFloorplanProvider:
    def __init__(self, model_path = model_path, image_size=(512, 512), use_gpu=True, 
                 min_line_length=30, angle_tolerance=5, merge_distance=10):
        self.model_dir = model_path
        self.image_size = image_size
        self.use_gpu = use_gpu
        self.min_line_length = min_line_length
        self.angle_tolerance = angle_tolerance  # degrees
        self.merge_distance = merge_distance  # pixels
        
        self.floorplan_map = {
            0: [255, 255, 255],  # background
            1: [192, 192, 224],  # closet
            2: [192, 255, 255],  # bathroom/washroom
            3: [224, 255, 192],  # livingroom/kitchen/dining
            4: [255, 224, 128],  # bedroom
            5: [255, 160, 96],   # hall
            6: [255, 224, 224],  # balcony
            7: [255, 255, 255],  # unused
            8: [255, 255, 255],  # unused
            9: [255, 60, 128],   # door & window
            10: [0, 0, 0],       # wall
        }
        
        self.sess = None
        self.input_tensor = None
        self.room_type_logit = None
        self.room_boundary_logit = None
        self._init_session()

    def _init_session(self):
        """Initializes a TensorFlow session and loads the pretrained model."""
        if self.use_gpu:
            try:
                print("Trying to use GPU for inference...")
                self.sess = tf.compat.v1.Session()
                self._load_model_from_dir()
                print("Successfully used GPU for inference.")
            except Exception as e:
                print(f"GPU inference failed with error: {e}")
                print("Falling back to CPU...")
                tf.compat.v1.reset_default_graph()
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                config_tf = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
                self.sess = tf.compat.v1.Session(config=config_tf)
                self._load_model_from_dir()
                print("Successfully used CPU for inference.")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            config_tf = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.compat.v1.Session(config=config_tf)
            self._load_model_from_dir()

    def _load_model_from_dir(self):
        """Loads the pretrained model from the specified directory."""
        meta_path = os.path.join(self.model_dir, 'pretrained_r3d.meta')
        index_path = os.path.join(self.model_dir, 'pretrained_r3d.index')
        if not (os.path.exists(meta_path) and os.path.exists(index_path)):
            raise FileNotFoundError(f"Pretrained model not found in '{self.model_dir}'.")
        
        saver = tf.compat.v1.train.import_meta_graph(meta_path, clear_devices=True)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())
        saver.restore(self.sess, os.path.join(self.model_dir, 'pretrained_r3d'))
        
        graph = tf.compat.v1.get_default_graph()
        self.input_tensor = graph.get_tensor_by_name('inputs:0')
        self.room_type_logit = graph.get_tensor_by_name('Cast:0')
        self.room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

    def imresize(self, image):
        """Resize an image using PIL."""
        return np.array(Image.fromarray(image.astype(np.uint8)).resize(
            (self.image_size[1], self.image_size[0])))

    def ind2rgb(self, ind_im):
        """Convert an indexed image to RGB using the color map."""
        rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3), dtype=np.uint8)
        for i, rgb in self.floorplan_map.items():
            rgb_im[ind_im == i] = rgb
        return rgb_im

    def process_image(self, im_path, save_output=True, output_dir=None):
        """
        Enhanced floorplan processing with noise reduction and geometric refinement.
        """
        # === 1. Load and preprocess input image ===
        im = imageio.imread(im_path)
        
        # Convert to RGB if needed
        if len(im.shape) == 2:  # Grayscale
            im = np.stack([im, im, im], axis=-1)
        elif im.shape[2] == 4:  # RGBA
            im = im[:, :, :3]  # Drop alpha channel
        elif im.shape[2] != 3:  # Unexpected format
            raise ValueError(f"Unexpected image shape: {im.shape}")
        
        im = im.astype(np.float32)
        im = self.imresize(im) / 255.0
        
        # Debug: check shape before reshape
        print(f"Image shape after resize: {im.shape}")
        print(f"Expected shape: {self.image_size}")

        # === 2. Run inference ===
        feed_dict = {self.input_tensor: im.reshape(1, self.image_size[0], self.image_size[1], 3)}
        room_type, room_boundary = self.sess.run(
            [self.room_type_logit, self.room_boundary_logit], 
            feed_dict=feed_dict
        )
        room_type = np.squeeze(room_type)
        room_boundary = np.squeeze(room_boundary)
        # === 3. Merge segmentation results ===
        floorplan = room_type.copy()
        floorplan[room_boundary == 1] = 9
        floorplan[room_boundary == 2] = 10

        # === 4. Save segmentation RGB ===
        floorplan_rgb = self.ind2rgb(floorplan)
        if save_output:
            if output_dir is None:
                output_dir = os.path.dirname(im_path)
            os.makedirs(output_dir, exist_ok=True)
            output_name = 'segmented_floorplan.png'
            image_path = os.path.join(config.work_dir, output_name)
            plt.imsave(image_path, floorplan_rgb)
            print(f"[Saved] Segmentation image: {image_path}")

        # === 5. ENHANCED Wall Extraction ===
        print("[Processing] Extracting walls with refinement...")
        walls = self.extract_walls_refined(floorplan, im)

        # === 6. ENHANCED Opening Extraction ===
        print("[Processing] Extracting openings with classification...")
        openings = self.extract_openings_refined(floorplan, walls)

        # === 7. Format output for compatibility ===
        walls_formatted = []
        for i, wall in enumerate(walls):
            wall_str = f"Wall{i+1}: {wall['start']} to {wall['end']}"
            walls_formatted.append(wall_str)

        openings_formatted = []
        for opening in openings:
            openings_formatted.append(list(opening['position']))

        param = {
            'walls': walls_formatted,
            'openings': openings_formatted
        }

        # === 8. Post-processing ===
        cleaned, seg_image_path = clean_floor_plan_single(param)

        process_para = {
            'cleaned_floorplan_path': seg_image_path,
        }

        memory.update_info_history(process_para)

        memory.update_info_history(cleaned)
        
        print(f"[Result] Extracted {len(walls)} walls and {len(openings)} openings")
        print(cleaned)

        walls = cleaned['walls']
        openings = cleaned['openings']
        
        del param

        return walls, openings

    def extract_walls_refined(self, floorplan, original_image):
        """
        Enhanced wall extraction with multiple noise reduction techniques.
        Creates continuous walls even when openings are present.
        """
        # === Step 1: Create wall mask INCLUDING openings for continuity ===
        # Use both walls (10) AND openings (9) to detect continuous wall lines
        wall_mask = (floorplan == 10) | (floorplan == 9)
        
        # Progressive morphological cleaning
        kernel = morphology.square(5)
        cleaned = morphology.binary_closing(wall_mask, kernel)
        cleaned = morphology.binary_opening(cleaned, morphology.square(3))
        cleaned = morphology.remove_small_objects(cleaned, min_size=50)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=100)

        # === Step 2: Distance transform for better skeletonization ===
        dist_transform = cv2.distanceTransform(
            cleaned.astype(np.uint8), cv2.DIST_L2, 5
        )
        
        # Adaptive threshold based on median distance
        median_dist = np.median(dist_transform[dist_transform > 0])
        thick_skeleton = dist_transform > (median_dist * 0.4)
        skeleton = morphology.skeletonize(thick_skeleton)

        # === Step 3: Optimized Hough line detection ===
        lines = probabilistic_hough_line(
            skeleton,
            threshold=12,
            line_length=25,
            line_gap=15
        )

        if not lines:
            print("[Warning] No lines detected")
            return []

        print(f"[Debug] Initial lines detected: {len(lines)}")

        # === Step 4: Merge collinear and nearby lines ===
        merged_lines = self.merge_collinear_lines(lines)
        print(f"[Debug] After merging: {len(merged_lines)}")

        # === Step 5: Snap to architectural angles ===
        snapped_lines = self.snap_to_angles(merged_lines)
        
        # === Step 6: Extend to intersections ===
        extended_lines = self.extend_to_intersections(snapped_lines)
        
        # === Step 7: Connect wall segments across openings ===
        continuous_lines = self.connect_walls_across_openings(extended_lines)
        print(f"[Debug] After connecting across openings: {len(continuous_lines)}")

        # === Step 8: Filter and format ===
        walls = []
        for i, (pt0, pt1) in enumerate(continuous_lines):
            length = np.linalg.norm(np.array(pt0) - np.array(pt1))
            if length >= self.min_line_length:
                walls.append({
                    'id': f'Wall{i+1}',
                    'start': tuple(map(int, pt0)),
                    'end': tuple(map(int, pt1)),
                    'length': float(length)
                })

        # Optional: Visualize
        self.visualize_walls(original_image, skeleton, walls)
        
        return walls

    def merge_collinear_lines(self, lines):
        """
        Merge lines that are collinear or nearly parallel.
        """
        if len(lines) < 2:
            return lines

        # Calculate angles for all lines
        angles = []
        for p0, p1 in lines:
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            angle = np.arctan2(dy, dx) * 180 / np.pi
            angles.append(angle % 180)  # Normalize to 0-180
        angles = np.array(angles)

        merged = []
        used = set()

        for i in range(len(lines)):
            if i in used:
                continue

            # Find lines with similar angles
            angle_diffs = np.abs(angles - angles[i])
            angle_diffs = np.minimum(angle_diffs, 180 - angle_diffs)
            similar_idx = np.where(angle_diffs < 5)[0]

            # Collect lines that are close together
            group = [i]
            for j in similar_idx:
                if j <= i or j in used:
                    continue
                if self.line_distance(lines[i], lines[j]) < self.merge_distance:
                    group.append(j)
                    used.add(j)

            # Fit a line through all points in the group
            all_points = []
            for idx in group:
                all_points.extend([lines[idx][0], lines[idx][1]])
            
            merged_line = self.fit_line_to_points(all_points)
            merged.append(merged_line)
            used.add(i)

        return merged

    def fit_line_to_points(self, points):
        """
        Fit a line through multiple points using PCA.
        """
        points = np.array(points)
        
        # PCA to find principal direction
        centroid = points.mean(axis=0)
        centered = points - centroid
        
        if len(points) < 2:
            return (tuple(points[0]), tuple(points[0]))
        
        _, _, vt = np.linalg.svd(centered)
        direction = vt[0]
        
        # Project points onto the principal axis
        projections = centered @ direction
        
        # Get extreme points
        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)
        
        return (tuple(points[min_idx]), tuple(points[max_idx]))

    def snap_to_angles(self, lines, snap_angles=[0, 45, 90, 135]):
        """
        Snap lines to dominant architectural angles.
        """
        snapped = []
        
        for p0, p1 in lines:
            p0, p1 = np.array(p0), np.array(p1)
            
            # Current angle
            dx, dy = p1 - p0
            current_angle = np.arctan2(dy, dx) * 180 / np.pi
            current_angle = current_angle % 180
            
            # Find nearest snap angle
            angle_diffs = [abs(current_angle - a) for a in snap_angles]
            angle_diffs = [min(d, 180 - d) for d in angle_diffs]
            
            if min(angle_diffs) < self.angle_tolerance:
                target_angle = snap_angles[np.argmin(angle_diffs)]
                
                # Rotate to snap angle
                length = np.linalg.norm(p1 - p0)
                center = (p0 + p1) / 2
                
                rad = target_angle * np.pi / 180
                direction = np.array([np.cos(rad), np.sin(rad)])
                
                new_p0 = center - direction * length / 2
                new_p1 = center + direction * length / 2
                
                snapped.append((tuple(new_p0), tuple(new_p1)))
            else:
                snapped.append((tuple(p0), tuple(p1)))
        
        return snapped

    def extend_to_intersections(self, lines, extension_length=15):
        """
        Extend line endpoints to meet at intersections.
        """
        if len(lines) < 2:
            return lines
        
        extended = []
        
        for i, (p0, p1) in enumerate(lines):
            p0, p1 = np.array(p0), np.array(p1)
            new_p0, new_p1 = p0.copy(), p1.copy()
            
            for j, (q0, q1) in enumerate(lines):
                if i == j:
                    continue
                
                q0, q1 = np.array(q0), np.array(q1)
                
                # Find intersection point
                intersection = self.line_intersection_extended(p0, p1, q0, q1)
                
                if intersection is not None:
                    # Extend if intersection is close to endpoint
                    dist_to_p0 = np.linalg.norm(intersection - p0)
                    dist_to_p1 = np.linalg.norm(intersection - p1)
                    
                    if dist_to_p0 < extension_length and dist_to_p0 < dist_to_p1:
                        new_p0 = intersection
                    elif dist_to_p1 < extension_length:
                        new_p1 = intersection
            
            extended.append((tuple(new_p0), tuple(new_p1)))
        
        return extended

    def connect_walls_across_openings(self, lines, max_gap=50, angle_tolerance=10):
        """
        Connect wall segments that are collinear and separated by small gaps (openings).
        This ensures walls remain continuous even when doors/windows interrupt them.
        """
        if len(lines) < 2:
            return lines
        
        # Build a graph of potentially connectable lines
        connectable = []
        
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                p0_i, p1_i = np.array(lines[i][0]), np.array(lines[i][1])
                p0_j, p1_j = np.array(lines[j][0]), np.array(lines[j][1])
                
                # Calculate angles
                angle_i = np.arctan2(p1_i[1] - p0_i[1], p1_i[0] - p0_i[0]) * 180 / np.pi
                angle_j = np.arctan2(p1_j[1] - p0_j[1], p1_j[0] - p0_j[0]) * 180 / np.pi
                
                # Normalize angles to 0-180
                angle_i = angle_i % 180
                angle_j = angle_j % 180
                
                # Check if angles are similar (collinear)
                angle_diff = abs(angle_i - angle_j)
                angle_diff = min(angle_diff, 180 - angle_diff)
                
                if angle_diff > angle_tolerance:
                    continue
                
                # Find minimum distance between the two segments
                distances = [
                    np.linalg.norm(p0_i - p0_j),
                    np.linalg.norm(p0_i - p1_j),
                    np.linalg.norm(p1_i - p0_j),
                    np.linalg.norm(p1_i - p1_j)
                ]
                
                min_dist = min(distances)
                min_idx = distances.index(min_dist)
                
                # If segments are close and collinear, check if they're actually aligned
                if min_dist < max_gap:
                    # Check if the gap is approximately along the line direction
                    if min_idx == 0:  # p0_i closest to p0_j
                        gap_vec = p0_j - p0_i
                    elif min_idx == 1:  # p0_i closest to p1_j
                        gap_vec = p1_j - p0_i
                    elif min_idx == 2:  # p1_i closest to p0_j
                        gap_vec = p0_j - p1_i
                    else:  # p1_i closest to p1_j
                        gap_vec = p1_j - p1_i
                    
                    # Check if gap is roughly aligned with line direction
                    line_dir = (p1_i - p0_i) / (np.linalg.norm(p1_i - p0_i) + 1e-6)
                    gap_angle = np.arctan2(gap_vec[1], gap_vec[0]) * 180 / np.pi % 180
                    gap_angle_diff = abs(angle_i - gap_angle)
                    gap_angle_diff = min(gap_angle_diff, 180 - gap_angle_diff)
                    
                    if gap_angle_diff < angle_tolerance * 2:  # More lenient for gap
                        connectable.append((i, j, min_dist, min_idx))
        
        # Build connected components using Union-Find
        parent = list(range(len(lines)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Connect collinear segments
        for i, j, dist, _ in sorted(connectable, key=lambda x: x[2]):
            union(i, j)
        
        # Group lines by connected component
        groups = {}
        for i in range(len(lines)):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # Merge each group into a single continuous line
        connected_lines = []
        
        for group_indices in groups.values():
            if len(group_indices) == 1:
                # Single line, keep as is
                connected_lines.append(lines[group_indices[0]])
            else:
                # Multiple lines to merge - find extreme points
                all_points = []
                for idx in group_indices:
                    all_points.extend([lines[idx][0], lines[idx][1]])
                
                # Fit a line and find extreme projections
                merged_line = self.fit_line_to_points(all_points)
                connected_lines.append(merged_line)
        
        return connected_lines

    def line_intersection_extended(self, p0, p1, q0, q1):
        """
        Find intersection of two infinite lines.
        """
        d1 = p1 - p0
        d2 = q1 - q0
        
        cross = np.cross(d1, d2)
        
        if abs(cross) < 1e-6:  # Parallel
            return None
        
        t = np.cross(q0 - p0, d2) / cross
        intersection = p0 + t * d1
        
        return intersection

    def extract_openings_refined(self, floorplan, walls):
        """
        Enhanced opening extraction with classification and wall association.
        """
        # Openings are label 9
        opening_mask = (floorplan == 9)
        
        # Clean the mask
        cleaned = morphology.binary_opening(opening_mask, morphology.square(3))
        cleaned = morphology.remove_small_objects(cleaned, min_size=25)
        
        # Label connected components
        labeled = label(cleaned)
        regions = regionprops(labeled)
        
        openings = []
        
        for region in regions:
            y, x = region.centroid
            
            # Geometric properties
            bbox = region.bbox
            height = bbox[2] - bbox[0]
            width = bbox[3] - bbox[1]
            aspect_ratio = max(height, width) / (min(height, width) + 1e-6)
            
            # Classify opening type
            opening_type = 'door' if aspect_ratio > 2.5 else 'window'
            
            # Find closest wall
            closest_wall = self.find_closest_wall(x, y, walls)
            
            openings.append({
                'position': (int(round(x)), int(round(y))),
                'type': opening_type,
                'area': int(region.area),
                'closest_wall': closest_wall,
                'aspect_ratio': float(aspect_ratio)
            })
        
        return openings

    def find_closest_wall(self, x, y, walls):
        """
        Find which wall an opening belongs to.
        """
        point = np.array([x, y])
        min_dist = float('inf')
        closest_wall_id = None
        
        for wall in walls:
            p0 = np.array(wall['start'])
            p1 = np.array(wall['end'])
            
            dist = self.point_to_segment_distance(point, p0, p1)
            
            if dist < min_dist:
                min_dist = dist
                closest_wall_id = wall['id']
        
        return closest_wall_id

    def point_to_segment_distance(self, point, seg_start, seg_end):
        """
        Calculate distance from point to line segment.
        """
        line_vec = seg_end - seg_start
        point_vec = point - seg_start
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-6:
            return np.linalg.norm(point_vec)
        
        line_unitvec = line_vec / line_len
        projection = np.dot(point_vec, line_unitvec)
        
        if projection < 0:
            return np.linalg.norm(point_vec)
        elif projection > line_len:
            return np.linalg.norm(point - seg_end)
        else:
            return np.linalg.norm(point_vec - projection * line_unitvec)

    def visualize_walls(self, original_image, skeleton, walls):
        """
        Visualize the extracted walls for debugging.
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        # Show skeleton
        axes[0].imshow(skeleton, cmap='gray')
        axes[0].set_title("Wall Skeleton")
        axes[0].axis('off')
        
        # Show skeleton with detected lines
        axes[1].imshow(skeleton, cmap='gray')
        for wall in walls:
            p0, p1 = wall['start'], wall['end']
            axes[1].plot([p0[0], p1[0]], [p0[1], p1[1]], 
                        'r-', linewidth=2, alpha=0.7)
            # Draw endpoints
            axes[1].plot(p0[0], p0[1], 'go', markersize=5)
            axes[1].plot(p1[0], p1[1], 'bo', markersize=5)
        axes[1].set_title(f"Continuous Walls ({len(walls)} total)")
        axes[1].axis('off')
        
        # Show on original image
        axes[2].imshow(original_image)
        for wall in walls:
            p0, p1 = wall['start'], wall['end']
            axes[2].plot([p0[0], p1[0]], [p0[1], p1[1]], 
                        'g-', linewidth=3, alpha=0.8)
        axes[2].set_title("Walls on Original Image")
        axes[2].axis('off')
        
        plt.tight_layout()

        print("showing the result_______________________")
        plt.show(block=False)  # Show plot without blocking
        plt.pause(3)  # Display for 3 seconds
        plt.close()



    @staticmethod
    def line_distance(line1, line2):
        """
        Calculate minimum distance between two line segments.
        """
        p1, p2 = np.array(line1[0]), np.array(line1[1])
        p3, p4 = np.array(line2[0]), np.array(line2[1])
        
        distances = [
            np.linalg.norm(p1 - p3),
            np.linalg.norm(p1 - p4),
            np.linalg.norm(p2 - p3),
            np.linalg.norm(p2 - p4)
        ]
        
        return min(distances)

    @staticmethod
    def distance(p, q):
        """Compute Euclidean distance between two points."""
        return np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

    @staticmethod
    def to_int(point):
        """Convert a coordinate point to integer values."""
        return (int(round(point[0])), int(round(point[1])))
