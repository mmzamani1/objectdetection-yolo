import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")


import torch
import cv2
import numpy as np
import time
from collections import defaultdict, deque
from ultralytics import YOLO

# ------------------- Config -------------------
MIN_CONFIDENCE = 0.4
DEBOUNCE_TIME = 9 # seconds
FRAME_SKIP = 1
MAX_DISAPPEARED = 1000
OBJ_TO_FRAME_RATIO = 0.03
MAKE_OUTPUT = False
UN_TARGETED = True
TAGET_SHAPE = "circle"
TAGET_COLOR = "blue"
MISSION = "track" # count track

# ------------------- Load YOLO Model -------------------
model = YOLO("weights/shapeWeights.pt")

# ------------------- I/O -------------------
input_path = "../data/qt/arrow.mp4"
filename = input_path.split("/")[-1].split(".")[0]
output_path = f"../data/outputs/{filename}_out_{time.time()}.mp4"

# ------------------- Load Input -------------------
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# ------------------- Get video properties -------------------
vid_fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ------------------- Prepare VideoWriter -------------------
out = None
if output_path and MAKE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, vid_fps, (width, height))

# ------------------- Variables -------------------
tracked_objects = {}  # {object_id: (class_name, dominant_color, centroid, bbox, last_seen)}
next_object_id = 0
class_counts = defaultdict(int)  # Count of objects by class
color_palette = None
last_move_cmds = {}  # {object_id: last_command}
last_print_time = {}  # {object_id: timestamp}
logs = []
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = 0
logs = deque(maxlen=2) 
prev_time = time.time()

# ------------------- Functions -------------------
def preprocess(frame):
    """
    Compensate for blue dominance in clear water.
    """
    # Convert to float for manipulation
    b, g, r = cv2.split(frame.astype(np.float32))

    # 1. Reduce blue slightly, boost red slightly
    b = b * 0.85
    r = r * 1.25

    # Clip back to [0,255]
    b = np.clip(b, 0, 255)
    r = np.clip(r, 0, 255)

    # Merge channels back
    result = cv2.merge([b, g, r]).astype(np.uint8)

    # Optional: mild sharpening
    kernel = np.array([[0, -0.25, 0],
                       [-0.25, 2, -0.25],
                       [0, -0.25, 0]])
    result = cv2.filter2D(result, -1, kernel)

    return result

def add_log(frame, logs):
    if MISSION == 'track':
        for i, msg in enumerate(logs):
            cv2.putText(frame, msg, (50, 30+(i*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    else:
        # Show total count
        count_text = f"Total objects: {len(tracked_objects)}"
        cv2.putText(frame, count_text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def print_progress(current, total, bar_length=30):
    fraction = current / total
    filled_length = int(bar_length * fraction)
    bar = "#" * filled_length + '-' * (bar_length - filled_length)
    print(f"\rProcessing: |{bar}| {current}/{total} frames", end='')
    
def get_move_command(centroid_pt, frame_width, bbox, frame_shape, tolerance=50):
    """
    Generate a move command based on centroid (left/right/forward)
    AND bounding box area (closeness).
    """
    # Direction
    frame_center_x = frame_width // 2
    dx = centroid_pt[0] - frame_center_x
    if abs(dx) <= tolerance:
        direction = "forward"
    elif dx > 0:
        direction = "right"
    else:
        direction = "left"
        
    # Closeness
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    frame_area = frame_shape[0] * frame_shape[1]
    area_ratio = area / frame_area

    if area_ratio > OBJ_TO_FRAME_RATIO:
        distance = "very close"
    elif area_ratio > OBJ_TO_FRAME_RATIO * 0.75:
        distance = "close"
    elif area_ratio > OBJ_TO_FRAME_RATIO * 0.5:
        distance = "medium"
    else:
        distance = "far"

    return direction, f"{distance} ({area_ratio:.3f})"

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
    Detect dominant color in clear, blue-dominated water.
    """
    if roi.size == 0:
        return "unknown", (255, 255, 255)

    # Convert to HSV for hue-based detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Compute mean hue
    mean_h = np.mean(h)
    mean_s = np.mean(s)
    mean_v = np.mean(v)

    # Simple thresholds for dominant color
    if mean_v < 50:
        color = "black"
    elif mean_s < 30:
        color = "gray"
    elif mean_h < 10 or mean_h > 160:
        color = "red"
    elif 10 <= mean_h < 25:
        color = "orange"
    elif 25 <= mean_h < 35:
        color = "yellow"
    elif 35 <= mean_h < 85:
        color = "green"
    elif 85 <= mean_h < 130:
        color = "blue"
    elif 130 <= mean_h < 160:
        color = "purple"
    else:
        color = "unknown"

    avg_bgr = tuple(np.mean(roi, axis=(0, 1)).astype(int))
    return color, avg_bgr

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
        move_cmd, distance = get_move_command(centroid_pt, frame_width, bbox, frame.shape)
        cv2.putText(frame, f"Move: {move_cmd}", (x1, y2 + 15),
                    font, font_scale, (0,0,255), 1, cv2.LINE_AA)

        # Debounce: print only if command changes
        global last_move_cmds
        now = time.time()
        prev_time = last_print_time.get(object_id, 0)
        prev_cmd = last_move_cmds.get(object_id)
        
        if move_cmd != prev_cmd or (now - prev_time > DEBOUNCE_TIME):
            if (class_name == TAGET_SHAPE and color_name == TAGET_COLOR) or UN_TARGETED:
                msg = f"Object {object_id} ({class_name}, {color_name}): {move_cmd}, {distance}"
                # print(msg)
                logs.append(msg)
                
            last_move_cmds[object_id] = move_cmd
            last_print_time[object_id] = now

    if MISSION == 'count':
        # Draw class counts
        y_offset = 25
        for class_name, count in class_counts.items():
            text = f"{class_name}: {count}"
            cv2.putText(frame, text, (10, y_offset),
                        font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
            y_offset += 20

# ------------------- Main Loop -------------------
while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # calculate fps
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # Draw FPS on frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (width - 180, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print_progress(frame_idx, total_frames)

    
    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        if out:
            add_log(frame, logs)
            out.write(frame)
        continue
        
    frame = preprocess(frame)

    # Run YOLO inference
    results = model(frame, verbose=False)  # returns a Results object
    detections = results[0].boxes.data.cpu().numpy()  # numpy array: [x1, y1, x2, y2, conf, cls]

    # Filter detections
    filtered_detections = [(x1, y1, x2, y2, conf, cls)
                           for x1, y1, x2, y2, conf, cls in detections
                           if conf > MIN_CONFIDENCE]

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
    
    add_log(frame, logs)
    
    cv2.imshow("YOLO Detection with Counting", frame)
    
    if out:
        out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()

# Print counts
if MISSION == 'count':
    print("\nFinal object counts:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")