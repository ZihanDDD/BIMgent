import os
from copy import deepcopy
import base64
from openai import OpenAI
from dotenv import load_dotenv
import re
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# Use imageio.v2 to avoid deprecation warnings
import imageio.v2 as imageio
from skimage import morphology
from skimage.transform import probabilistic_hough_line
from PIL import Image
from skimage.measure import label, regionprops
import json
from conf.config import Config
from BIMgent.memory.local_memory import LocalMemory
from BIMgent.provider.Deep_fp_provider.utils.floorplan_postprocessing import clean_floor_plan_single
from BIMgent.utils.dict_utils import kget


memory = LocalMemory()
config = Config()

model_path = kget(config.env_config, "models_path", default='')['deep_floorplan']


# Disable eager execution for TF1.x compatibility mode in TF2.x
tf.compat.v1.disable_eager_execution()
# Only show TensorFlow errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

load_dotenv()
config = Config()
api_key = os.getenv('OA_OPENAI_KEY')
config = Config()
memory = LocalMemory()

client = OpenAI(api_key=api_key)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class DeepFloorplanProvider:
    def __init__(self, model_dir=model_path, image_size=(512, 512), use_gpu=True, min_line_length=20):
        self.model_dir = model_dir
        self.image_size = image_size  # (height, width)
        self.use_gpu = use_gpu
        self.min_line_length = min_line_length
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
                config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
                self.sess = tf.compat.v1.Session(config=config)
                self._load_model_from_dir()
                print("Successfully used CPU for inference.")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.compat.v1.Session(config=config)
            self._load_model_from_dir()

    def _load_model_from_dir(self):
        """Loads the pretrained model from the specified directory."""
        meta_path = os.path.join(self.model_dir, 'pretrained_r3d.meta')
        index_path = os.path.join(self.model_dir, 'pretrained_r3d.index')
        if not (os.path.exists(meta_path) and os.path.exists(index_path)):
            raise FileNotFoundError("Pretrained model not found in '{}'. "
                                    "Please ensure 'pretrained_r3d.meta' and related files are available.".format(self.model_dir))
        saver = tf.compat.v1.train.import_meta_graph(meta_path, clear_devices=True)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())
        saver.restore(self.sess, os.path.join(self.model_dir, 'pretrained_r3d'))
        graph = tf.compat.v1.get_default_graph()
        self.input_tensor = graph.get_tensor_by_name('inputs:0')
        self.room_type_logit = graph.get_tensor_by_name('Cast:0')
        self.room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

    def imresize(self, image):
        """Resize an image using PIL. Expects the image in numpy array format."""
        # PIL expects size as (width, height)
        return np.array(Image.fromarray(image.astype(np.uint8)).resize((self.image_size[1], self.image_size[0])))

    def ind2rgb(self, ind_im):
        """Convert an indexed image to an RGB image using the given color map."""
        rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3), dtype=np.uint8)
        for i, rgb in self.floorplan_map.items():
            rgb_im[ind_im == i] = rgb
        return rgb_im
        
    def process_image(self, im_path, save_output=True, output_dir=None):
        """
        Processes the floorplan image with step-by-step visualization.
        """
        # === 1. Load and preprocess input image ===
        im = imageio.imread(im_path)
        im = im.astype(np.float32)
        im = self.imresize(im) / 255.0

        # Debug: Show preprocessed image
        #plt.figure(); plt.imshow(im); plt.title("1. Preprocessed Input"); plt.axis('off'); plt.show()

        # === 2. Run inference ===
        feed_dict = {self.input_tensor: im.reshape(1, self.image_size[0], self.image_size[1], 3)}
        room_type, room_boundary = self.sess.run([self.room_type_logit, self.room_boundary_logit],
                                                feed_dict=feed_dict)
        room_type = np.squeeze(room_type)
        room_boundary = np.squeeze(room_boundary)

        # Debug: Show raw model outputs
        # plt.figure(); plt.imshow(room_type); plt.title("2. Room Type Output"); plt.axis('off'); plt.show()
        #plt.figure(); plt.imshow(room_boundary); plt.title("3. Room Boundary Output"); plt.axis('off'); plt.show()

        # === 3. Merge segmentation results ===
        floorplan = room_type.copy()
        floorplan[room_boundary == 1] = 9
        floorplan[room_boundary == 2] = 10

        # === 4. Convert to RGB for visualization ===
        floorplan_rgb = self.ind2rgb(floorplan)
        # plt.figure(); plt.imshow(floorplan_rgb); plt.title("4. Segmentation RGB"); plt.axis('off'); plt.show()

        # === 5. Save the segmentation result ===
        if save_output:
            if output_dir is None:
                output_dir = os.path.dirname(im_path)
            os.makedirs(output_dir, exist_ok=True)

            output_name = 'segmented_floorplan.png'
            image_path = os.path.join(config.work_dir, output_name)
            plt.imsave(image_path, floorplan_rgb)
            print(f"[Saved] Segmentation image: {image_path}")

        # === 6. Extract walls mask (label==9 or 10) ===
        wall_mask = (floorplan == 9) | (floorplan == 10)
        # plt.figure(); plt.imshow(wall_mask, cmap='gray'); plt.title("5. Wall Mask"); plt.axis('off'); plt.show()

        # === 7. Clean and skeletonize ===
        int_matrix = np.where(wall_mask, 1, 0)
        cleaned = morphology.binary_opening(int_matrix, morphology.square(3))
        cleaned = morphology.remove_small_objects(cleaned, min_size=10)
        skeleton = morphology.skeletonize(cleaned)
        #plt.figure(); plt.imshow(skeleton, cmap='gray'); plt.title(" Skeletonized Walls"); plt.axis('off'); plt.show()

        # === 8. Hough Line Detection ===
        lines = probabilistic_hough_line(skeleton, threshold=10, line_length=20, line_gap=50)
        # plt.figure(); plt.imshow(skeleton, cmap='gray')
        # for line in lines:
        #     p0, p1 = line
        #     plt.plot((p0[0], p1[0]), (p0[1], p1[1]), 'r')
        #plt.title("7. Detected Lines (Hough)"); plt.axis('off'); plt.show()

        # === 9. Line post-processing ===
        # final_lines = self.split_lines_at_intersections(lines)
        final_lines = [line for line in lines if self.distance(line[0], line[1]) >= self.min_line_length]

        plt.figure(); plt.imshow(skeleton, cmap='gray')
        for line in final_lines:
            p0, p1 = line
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]), 'g')
        plt.title("Lines"); plt.axis('off'); plt.show()
        
        walls = []

        # Print wall segments
        for i, (pt0, pt1) in enumerate(final_lines):
            walls.append(f"Wall{i+1}: {pt0} to {pt1}")

        # Processing openings from floorplan (assumed where floorplan == 9)
        opening_mask = (floorplan == 9)
        opening_int_matrix = np.where(opening_mask, 1, 0)
        cleaned_opening = morphology.binary_opening(opening_int_matrix, morphology.square(3))
        cleaned_opening = morphology.remove_small_objects(cleaned_opening, min_size=20)
        skeleton_opening = morphology.skeletonize(cleaned_opening)

        labeled_skel = label(skeleton_opening)
        openings = []
        for region in regionprops(labeled_skel):
            # region.centroid returns (row, col) coordinates
            y, x = region.centroid
            openings.append([int(round(x)), int(round(y))])
            
            
        param = {
            'walls': walls,
            'openings': openings
        }
        
        
        cleaned, seg_image_path = clean_floor_plan_single(param)
        memory.update_info_history(cleaned)
        
        print(cleaned)
        
        
        del param
        
        return seg_image_path, walls, openings
        
    @staticmethod
    def to_int(point):
        """Convert a coordinate point to integer values."""
        return (int(round(point[0])), int(round(point[1])))

    @staticmethod
    def distance(p, q):
        """Compute Euclidean distance between two points."""
        return np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

    @staticmethod
    def align_endpoints(lines, threshold=5):
        """
        Adjust endpoints of lines if they are close to each other.
        Endpoints within the threshold are snapped to their average.
        """
        new_coords = {}
        endpoints = []
        for i, (p1, p2) in enumerate(lines):
            endpoints.append((i, 0, p1))
            endpoints.append((i, 1, p2))
        for i, (line_i, pos_i, coord_i) in enumerate(endpoints):
            for j, (line_j, pos_j, coord_j) in enumerate(endpoints):
                if i >= j:
                    continue
                if DeepFloorplanProvider.distance(coord_i, coord_j) < threshold:
                    avg = ((coord_i[0] + coord_j[0]) / 2, (coord_i[1] + coord_j[1]) / 2)
                    avg_int = DeepFloorplanProvider.to_int(avg)
                    new_coords[(line_i, pos_i)] = avg_int
                    new_coords[(line_j, pos_j)] = avg_int
        new_lines = []
        for i, (p1, p2) in enumerate(lines):
            new_p1 = new_coords.get((i, 0), DeepFloorplanProvider.to_int(p1))
            new_p2 = new_coords.get((i, 1), DeepFloorplanProvider.to_int(p2))
            new_lines.append((new_p1, new_p2))
        return new_lines

    @staticmethod
    def line_intersection(p1, p2, p3, p4, tol=1e-6):
        """
        Computes the intersection point between two line segments p1-p2 and p3-p4.
        Returns the intersection point as integers if it lies strictly within both segments.
        """
        x1, y1 = p1; x2, y2 = p2
        x3, y3 = p3; x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if np.abs(denom) < tol:
            return None  # Lines are parallel or coincident

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom

        if tol < t < 1 - tol and tol < u < 1 - tol:
            inter_x = x1 + t * (x2 - x1)
            inter_y = y1 + t * (y2 - y1)
            return DeepFloorplanProvider.to_int((inter_x, inter_y))
        return None

    @staticmethod
    def split_line_at_points(line, points):
        """
        Splits a line into segments at the given intersection points.
        """
        p_start, p_end = line
        p_start = DeepFloorplanProvider.to_int(p_start)
        p_end = DeepFloorplanProvider.to_int(p_end)
        all_points = [p_start] + [DeepFloorplanProvider.to_int(pt) for pt in points] + [p_end]
        unique_points = {pt for pt in all_points}
        all_points = list(unique_points)
        all_points.sort(key=lambda pt: DeepFloorplanProvider.distance(p_start, pt))
        segments = []
        for i in range(len(all_points) - 1):
            segments.append((all_points[i], all_points[i + 1]))
        return segments

    @staticmethod
    def split_lines_at_intersections(lines):
        """
        For each line, check for intersections with all other lines and split the line at each intersection.
        """
        new_lines = []
        for i, line in enumerate(lines):
            p1, p2 = line
            intersection_points = []
            for j, other_line in enumerate(lines):
                if i == j:
                    continue
                q1, q2 = other_line
                inter_pt = DeepFloorplanProvider.line_intersection(p1, p2, q1, q2)
                if inter_pt is not None:
                    intersection_points.append(inter_pt)
            if intersection_points:
                segments = DeepFloorplanProvider.split_line_at_points(line, intersection_points)
                new_lines.extend(segments)
            else:
                new_lines.append(line)
        return new_lines
    

class DeepFloorplanPostprocessingProvider():
    """
    The designer provider for the design of the floorplan and the other structures that should be on the design for instance the door, windows etc.
    """
    def __init__(
        self,
        task_description: str,
        **kwargs,
    ):
        self.task_description = task_description

        # Load and parse the template file once during initialization
    # === Helpers ===
    def parse_coordinate_string(self, wall_str):
        """Extract start and end coordinates from a wall string like 'Wall1: (x1, y1) to (x2, y2)'."""
        coords = re.findall(r'\(([\d\.\-]+),\s*([\d\.\-]+)\)', wall_str)
        if len(coords) == 2:
            start = (float(coords[0][0]), float(coords[0][1]))
            end = (float(coords[1][0]), float(coords[1][1]))
            return start, end
        return (0, 0), (0, 0)

    def get_wall_id(self, wall_str):
        """Extract the wall ID (e.g., Wall1) from the wall string."""
        match = re.match(r'(\w+):', wall_str.strip())
        return match.group(1) if match else "Wall"

    def __call__(self, image_path, seg_image_path, *args, **kwargs) -> str:

        # Deep copy memory to avoid unintended side effects
        params = deepcopy(memory.working_area)
        response = {}
        walls_coord = memory.get_recent_history('walls')
        openings_coord = memory.get_recent_history('openings')
        
        ori_image = encode_image(image_path)
        seg_image = encode_image(seg_image_path)
        
        prompt = f""" You will be provided with two images and corresponding floorplan metadata.
        Image 1: The original floorplan.
        Image 2: The processed floorplan, where walls are skeletonized and openings are marked with red points. Each opening is annotated with its associated wall name.
        The first image is the original floorplan
        
        The metadata includes:
        Wall coordinates, a list of wall names and their corresponding coordinates: {walls_coord}
        Opening coordinates, a list of opening coordinates (not yet labeled as doors or windows).: {openings_coord}
        
        Here are some hints to help you for the decision for the response.
        floorplan:
        The processed floorplan image (Image 2) and its associated metadata may contain minor errors such as unconnected walls, misdetected structures, or incorrectly positioned openings. Your task is to refine the metadata based on the following instructions:

        1. Use Image 1 (original floorplan) as the baseline reference. Image 2 is mostly correct — only refine walls and openings where necessary.
        2. Remove noise or structural errors, including:
            - Short isolated walls whose endpoints do not connect to any other wall and are far from other walls.    
        3. Ensure wall connectivity by adjusting coordinates:
            - Extend or trim walls along their original direction to restore intersections observed in Image 1.
            - If a wall is isolated (i.e., disconnected at both endpoints) but is very close to another wall, snap its endpoints to the nearest neighboring wall without changing its orientation (only adjust length, not angle).
        4. Ensure that the final metadata is visually and structurally consistent with Image 1, accurately representing the floorplan. If confirmed as a duplicate detection, delete the extra instance.
        5. No need to change any opening coordinates.
        You should respond strictly in the following format, and you must not output any comments, explanations, or additional information. Don't include anything beside the requested data represented in the following format
        floorplan: 
            [
                {{
                    "walls": [
                        "Wall1: (x1, y1) to (x2, y2)",
                        "Wall2: (x3, y3) to (x4, y4)",
                        ...
                    ],
                    "openings": [[x5, y5], ...]
                }}
            ]
        """
    

        completion = client.chat.completions.create(
            model="o4-mini-2025-04-16",
            messages=[
                {
                    "role": "system",
                    "content": [
                        { "type": "text", "text": "You are an artificial assistant for the process of image coordinates refinement." },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": f"{prompt}" },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{ori_image}",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{seg_image}",
                            },
                        },
                    ],
                }
            ],
        )


        
        message = completion.choices[0].message.content
        # Extract and parse
        match = re.search(r'\[.*', message, re.DOTALL)
        floorplan_content = match.group(0).strip() if match else ""
        
        
        floorplan_str = floorplan_content.replace('{{', '{').replace('}}', '}')  # Handle double braces if present
        floorplan_data = json.loads(floorplan_str)
        
        # === Plot ===
        plt.figure(figsize=(6, 6))
        for item in floorplan_data:
            walls = item.get("walls", [])
            openings = item.get("openings", [])

            # Plot each wall
            for wall_str in walls:
                start, end = self.parse_coordinate_string(wall_str)
                plt.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=2)
                
                # Label the wall by its ID at its midpoint
                mid_x = (start[0] + end[0]) / 2.0
                mid_y = (start[1] + end[1]) / 2.0
                plt.text(mid_x, mid_y, self.get_wall_id(wall_str), color='blue', fontsize=10, 
                        ha='center', va='center')
            
            # Plot openings if provided
            if openings:
                for op in openings:
                    plt.plot(op[0], op[1], 'ro', markersize=6)

        plt.axis('equal')
        plt.grid(False)
        plt.gca().invert_yaxis()

        # === Save ===
        
        screenshot_name = "postprocessed_floorplan_visualization.png"
        screenshot_path = os.path.join(config.work_dir, screenshot_name)
        plt.savefig(screenshot_path, dpi=300, bbox_inches='tight')
        print(f"The processed floorplan image is saved in {screenshot_path}")

        plt.show()
        
        print(floorplan_content)
        params = {
            'floorplan': floorplan_content
            }
        memory.update_info_history(params)
        

        del params
    

        return screenshot_path



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_path', type=str, default='./demo/45765448.jpg',
                        help='Path to the input floorplan image.')
    args, _ = parser.parse_known_args()

    provider = DeepFloorplanProvider()
    provider.process_image(args.im_path)
