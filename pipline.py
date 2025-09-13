import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")


import torch
import cv2
import numpy as np
import time
from collections import defaultdict, deque
from ultralytics import YOLO

TARGET_FPS = 500
FRAME_INTERVAL = 1.0 / TARGET_FPS

# Load Model
model = YOLO("./pipelineWeights.pt")

model.eval()
# model.to("cuda").half()

input_path = "./pipeline1.mp4"
filename = input_path.split("/")[-1].split(".")[0]
output_path = f"./outputs/{filename}_out.mp4"

# Setup Input
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print(f"Error: Could not open video {input_path}")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare VideoWriter
out = None
if output_path:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


# Object counting and tracking variables
tracked_objects = {}  # {object_id: (class_name, dominant_color, centroid, bbox, last_seen)}
next_object_id = 0
MAX_DISAPPEARED = 1000  # Frames to keep an object before considering it gone
MIN_CONFIDENCE = 0.5  # Minimum confidence for detection
class_counts = defaultdict(int)  # Count of objects by class
color_palette = None
last_move_cmds = {}  # {object_id: last_command}
last_print_time = {}  # {object_id: timestamp}
DEBOUNCE_TIME = 0.5  # seconds
logs = []


TAGET_SHAPE = "pipeline"
# TAGET_COLOR = "red"


def preprocess(frame):
    # 1. White balance (compensate color shift)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Balance channels (keep types consistent)
    avg_a = np.mean(a)
    avg_b = np.mean(b)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    l_float = l.astype(np.float32) / 255.0

    a = a - ((avg_a - 128) * l_float * 1.1)
    b = b - ((avg_b - 128) * l_float * 1.1)

    # Clip back to [0,255] and convert to uint8
    a = np.clip(a, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)

    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Dehaze with CLAHE on LAB L-channel
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge([cl, a, b])
    result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 3. Sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    result = cv2.filter2D(result, -1, kernel)

    return result


def add_log(frame, logs):
    for i, msg in enumerate(logs):
        cv2.putText(frame, msg, (10, 30+(i*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def get_move_command(centroid_pt, frame_width, tolerance=50):
    """
    Generate a simple move command based on the centroid's x-position.
    - centroid_pt: (x, y) of the object
    - frame_width: width of the frame
    - tolerance: pixels around center considered "forward"
    Returns: "left", "right", "forward"
    """
    frame_center_x = frame_width // 2
    dx = centroid_pt[0] - frame_center_x

    if abs(dx) <= tolerance:
        return "forward"
    elif dx > 0:
        return "right"
    else:
        return "left"


def centroid(bbox):
    # Calculate centroid from bounding box [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

def get_color_palette():
    """Return a color palette for visualization"""
    return {
        'red': (0, 0, 255),
        'orange': (0, 165, 255),
        'yellow': (0, 255, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'purple': (255, 0, 127),
        'pink': (203, 192, 255),
        'brown': (42, 42, 165),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'gray': (128, 128, 128),
        'unknown': (255, 255, 255)
    }

def detect_dominant_color(roi):
    """
    Detect the dominant color in a region of interest (ROI).
    Returns the color name and average BGR values.
    """
    if roi.size == 0:
        return "unknown", (255, 255, 255)
    
    # Convert ROI to HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV
    color_ranges = {
        'red': [(0, 50, 20), (10, 255, 255), (170, 50, 20), (180, 255, 255)],
        'orange': [(11, 50, 20), (25, 255, 255)],
        'yellow': [(26, 50, 20), (35, 255, 255)],
        'green': [(36, 50, 20), (85, 255, 255)],
        'blue': [(86, 50, 20), (125, 255, 255)],
        'purple': [(126, 50, 20), (150, 255, 255)],
        'pink': [(160, 50, 80), (180, 150, 255)],
        'brown': [(10, 100, 20), (20, 255, 200)],
        'black': [(0, 0, 0), (180, 255, 50)],
        'white': [(0, 0, 200), (180, 30, 255)],
        'gray':  [(0, 0, 50), (180, 30, 200)]
    }

    # Count pixels in each range
    max_pixels = 0
    dominant_color = "unknown"

    for color, ranges in color_ranges.items():
        mask = None
        if len(ranges) == 2:  # Single range
            lower, upper = ranges
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv_roi, lower, upper)
        elif len(ranges) == 4:  # Two ranges (for red)
            lower1, upper1, lower2, upper2 = ranges
            lower1, upper1 = np.array(lower1), np.array(upper1)
            lower2, upper2 = np.array(lower2), np.array(upper2)
            mask1 = cv2.inRange(hsv_roi, lower1, upper1)
            mask2 = cv2.inRange(hsv_roi, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)

        count = cv2.countNonZero(mask)
        if count > max_pixels:
            max_pixels = count
            dominant_color = color

    # Average BGR for visualization (not used in classification)
    avg_bgr = np.mean(roi, axis=(0, 1)).astype(int)

    return dominant_color, tuple(avg_bgr)


def update_object_count(detections, frame):
    global next_object_id, tracked_objects, class_counts, color_palette
    
    # Initialize color palette if not done
    if color_palette is None:
        color_palette = get_color_palette()
    
    # If no detections, mark all objects as disappeared
    if len(detections) == 0:
        for object_id in list(tracked_objects.keys()):
            tracked_objects[object_id] = (
                tracked_objects[object_id][0],  # class_name
                tracked_objects[object_id][1],  # centroid
                tracked_objects[object_id][2],  # bbox
                tracked_objects[object_id][3] + 1,  # increment disappeared count
                tracked_objects[object_id][4]  # dominant_color
            )
            # Remove object if it hasn't been seen for too long
            if tracked_objects[object_id][3] > MAX_DISAPPEARED:
                del tracked_objects[object_id]
        return
    
    # Current frame centroids
    current_centroids = np.zeros((len(detections), 2), dtype="int")
    current_bboxes = []
    current_classes = []
    current_colors = []
    
    # Extract centroids and classes from detections
    for i, (*box, conf, cls) in enumerate(detections):
        x1, y1, x2, y2 = map(int, box)
        current_centroids[i] = centroid((x1, y1, x2, y2))
        current_bboxes.append((x1, y1, x2, y2))
        current_classes.append(model.names[int(cls)])
        
        # Detect color for this detection
        roi = frame[y1:y2, x1:x2]
        color_name, _ = detect_dominant_color(roi)
        current_colors.append(color_name)
    
    # If no objects being tracked, register all detections
    if len(tracked_objects) == 0:
        for i in range(len(detections)):
            class_name = current_classes[i]
            color_name = current_colors[i]
            tracked_objects[next_object_id] = (
                class_name, 
                current_centroids[i], 
                current_bboxes[i], 
                0,  # disappeared count
                color_name  # dominant color
            )
            class_counts[class_name] += 1
            next_object_id += 1
    else:
        # Track existing objects and register new ones
        object_ids = list(tracked_objects.keys())
        object_centroids = [tracked_objects[obj_id][1] for obj_id in object_ids]
        
        # Calculate distances between existing objects and new detections
        D = np.zeros((len(object_ids), len(current_centroids)))
        for i in range(len(object_ids)):
            for j in range(len(current_centroids)):
                D[i, j] = np.linalg.norm(object_centroids[i] - current_centroids[j])
        
        # Match existing objects to new detections
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        
        used_rows = set()
        used_cols = set()
        
        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
                
            object_id = object_ids[row]
            tracked_objects[object_id] = (
                current_classes[col],  # update class name (in case of misclassification)
                current_centroids[col], 
                current_bboxes[col], 
                0,  # reset disappeared count
                current_colors[col]  # update color
            )
            
            used_rows.add(row)
            used_cols.add(col)
        
        # Check for unmatched rows (objects that disappeared)
        unmatched_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unmatched_rows:
            object_id = object_ids[row]
            class_name, centroid_pt, bbox, disappeared, color_name = tracked_objects[object_id]
            tracked_objects[object_id] = (class_name, centroid_pt, bbox, disappeared + 1, color_name)
            
            # Remove object if it hasn't been seen for too long
            if tracked_objects[object_id][3] > MAX_DISAPPEARED:
                del tracked_objects[object_id]
        
        # Check for unmatched columns (new objects)
        unmatched_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unmatched_cols:
            class_name = current_classes[col]
            color_name = current_colors[col]
            tracked_objects[next_object_id] = (
                class_name, 
                current_centroids[col], 
                current_bboxes[col], 
                0,  # disappeared count
                color_name  # dominant color
            )
            class_counts[class_name] += 1
            next_object_id += 1

def draw_object_info(frame):
    # Pre-cache font + scale (avoids recomputing)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    frame_width = frame.shape[1]

    # Draw bounding boxes & labels
    for object_id, (class_name, centroid_pt, bbox, disappeared, color_name) in tracked_objects.items():
        if disappeared > 0:
            continue  # Skip objects that disappeared

        x1, y1, x2, y2 = bbox
        color_bgr = color_palette.get(color_name, (0, 255, 0))

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 1)

        # Label
        label = f"ID:{object_id} {class_name} ({color_name})"
        cv2.putText(frame, label, (x1, y1 - 5),
                    font, font_scale, color_bgr, thickness, cv2.LINE_AA)

        # Centroid
        cv2.circle(frame, centroid_pt, 3, color_bgr, -1)

        # ---- Generate move command ----
        move_cmd = get_move_command(centroid_pt, frame_width)
        cv2.putText(frame, f"Move: {move_cmd}", (x1, y2 + 15),
                    font, font_scale, (0,0,255), 1, cv2.LINE_AA)

        # Debounce: print only if command changes
        global last_move_cmds
        now = time.time()
        prev_time = last_print_time.get(object_id, 0)
        prev_cmd = last_move_cmds.get(object_id)
        
        if move_cmd != prev_cmd and (now - prev_time > DEBOUNCE_TIME):
            if class_name == TAGET_SHAPE:
                msg = f"Object {object_id} ({class_name}, {color_name}): {move_cmd}"
                print(msg)
                logs.append(msg)
                
            last_move_cmds[object_id] = move_cmd
            last_print_time[object_id] = now



    # Draw class counts only once
    # y_offset = 25
    # for class_name, count in class_counts.items():
    #     text = f"{class_name}: {count}"
    #     cv2.putText(frame, text, (10, y_offset),
    #                 font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
    #     y_offset += 20





while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = preprocess(frame)

    # Run YOLO inference
    results = model(frame, verbose=False)  # returns a Results object
    detections = results[0].boxes.data.cpu().numpy()  # numpy array: [x1, y1, x2, y2, conf, cls]

    # Filter detections
    filtered_detections = []
    for x1, y1, x2, y2, conf, cls in detections:
        if conf > MIN_CONFIDENCE:
            filtered_detections.append((x1, y1, x2, y2, conf, cls))

    # Update tracking and counting
    update_object_count(filtered_detections, frame)

    # Draw
    draw_object_info(frame)


    for *box, conf, cls in filtered_detections:
        label = model.names[int(cls)]
        x1, y1, x2, y2 = map(int, box)
        
        # Detect color for this detection
        roi = frame[y1:y2, x1:x2]
        color_name, _ = detect_dominant_color(roi)
        
        # msg = f"{label}:{color_name}:{x1},{y1},{x2},{y2}\n"
        # print(f"Detected: {msg.strip()}")
        # ser.write(msg.encode())  # Uncomment if Arduino is connected
        # if label == 'circle' and color_name == 'yellow':
        #     print("GOTCHA!!!")
        

    # Show total count
    # count_text = f"Total objects: {len(tracked_objects)}"
    # cv2.putText(frame, count_text, (10, frame.shape[0] - 10),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    add_log(frame, logs[-5:])
    
    cv2.imshow("YOLO Detection with Counting", frame)
    
    if out:
        out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed = time.time() - start_time
    sleep_time = max(0, FRAME_INTERVAL - elapsed)
    time.sleep(sleep_time)

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()

# Print counts
# print("\nFinal object counts:")
# for class_name, count in class_counts.items():
#     print(f"{class_name}: {count}")