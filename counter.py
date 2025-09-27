import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import time

# ------------------- Config -------------------
INPUT_VIDEO = "E:/0CODING/MyProjects/SUB-IP/data/IP/2024-08-31_11.18.34.mp4"
OUTPUT_VIDEO = "output_video.mp4"
YOLO_WEIGHTS = "E:/0CODING/MyProjects/SUB-IP/prj-files/weights/shapeWeights.pt"  # Your trained YOLO model
MIN_CONFIDENCE = 0.5
FRAME_SKIP = 1  # Skip frames to speed up processing
DEBOUNCE_TIME = 0.5  # seconds between repeated logs

# ------------------- Load YOLO -------------------
model = YOLO(YOLO_WEIGHTS)

# ------------------- Video I/O -------------------
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise Exception(f"Could not open video: {INPUT_VIDEO}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = None
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# ------------------- Tracking & Counting -------------------
tracked_objects = {}  # {id: (class_name, bbox, last_seen_frame)}
next_id = 0
MAX_DISAPPEARED = 50  # frames to remove lost objects
class_counts = defaultdict(int)
logs = deque(maxlen=10)
last_print_time = {}

def centroid(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2)/2), int((y1 + y2)/2))

def get_color(class_name):
    # Return BGR color for visualization
    color_palette = {
        'pipeline': (0,255,0),
        'leak': (0,0,255),
        'other': (255,0,0)
    }
    return color_palette.get(class_name, (255,255,255))

def add_log(frame, logs):
    for i, msg in enumerate(logs):
        cv2.putText(frame, msg, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

def update_tracking(detections, frame_idx):
    global next_id

    if len(detections) == 0:
        # No detections, mark existing objects as disappeared
        for obj_id in list(tracked_objects.keys()):
            class_name, bbox, last_seen = tracked_objects[obj_id]
            if frame_idx - last_seen > MAX_DISAPPEARED:
                del tracked_objects[obj_id]
        return

    # Convert detections to centroids
    current_centroids = []
    current_bboxes = []
    current_classes = []
    for x1, y1, x2, y2, conf, cls in detections:
        if conf < MIN_CONFIDENCE:
            continue
        current_centroids.append(centroid((x1, y1, x2, y2)))
        current_bboxes.append((x1, y1, x2, y2))
        current_classes.append(cls)

    if len(tracked_objects) == 0:
        # Register all detections as new objects
        for i in range(len(current_centroids)):
            tracked_objects[next_id] = (current_classes[i], current_bboxes[i], frame_idx)
            class_counts[current_classes[i]] += 1
            next_id += 1
        return

    # Compute distances
    object_ids = list(tracked_objects.keys())
    object_centroids = [centroid(tracked_objects[obj_id][1]) for obj_id in object_ids]

    # Check if there are no current centroids (all below confidence)
    if len(current_centroids) == 0:
        for obj_id in list(tracked_objects.keys()):
            if frame_idx - tracked_objects[obj_id][2] > MAX_DISAPPEARED:
                del tracked_objects[obj_id]
        return

    D = np.zeros((len(object_ids), len(current_centroids)))
    for i, c1 in enumerate(object_centroids):
        for j, c2 in enumerate(current_centroids):
            D[i,j] = np.linalg.norm(np.array(c1)-np.array(c2))

    # Match objects to detections
    used_rows, used_cols = set(), set()
    for i in range(D.shape[0]):
        if D.shape[1] == 0:
            continue
        col = D[i].argmin()
        if i in used_rows or col in used_cols:
            continue
        obj_id = object_ids[i]
        tracked_objects[obj_id] = (current_classes[col], current_bboxes[col], frame_idx)
        used_rows.add(i)
        used_cols.add(col)

    # Register unmatched detections as new objects
    unmatched_cols = set(range(len(current_centroids))) - used_cols
    for col in unmatched_cols:
        tracked_objects[next_id] = (current_classes[col], current_bboxes[col], frame_idx)
        class_counts[current_classes[col]] += 1
        next_id += 1

    # Remove lost objects
    for obj_id in list(tracked_objects.keys()):
        if frame_idx - tracked_objects[obj_id][2] > MAX_DISAPPEARED:
            del tracked_objects[obj_id]


# ------------------- Main Loop -------------------
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    if frame_idx % FRAME_SKIP != 0:
        out.write(frame)
        continue

    # YOLO detection
    results = model(frame, verbose=False)[0]
    detections = results.boxes.data.cpu().numpy()  # [[x1,y1,x2,y2,conf,cls], ...]

    # Convert detections to list of (x1,y1,x2,y2,conf,class_name)
    detection_list = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        class_name = model.names[int(cls)]
        detection_list.append((x1, y1, x2, y2, conf, class_name))

    # Update tracking & counting
    update_tracking(detection_list, frame_idx)

    # Draw tracked objects
    for obj_id, (class_name, bbox, _) in tracked_objects.items():
        x1, y1, x2, y2 = map(int, bbox)
        color = get_color(class_name)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"ID:{obj_id} {class_name}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Show total counts
    y = 30
    for cls, count in class_counts.items():
        cv2.putText(frame, f"{cls}: {count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y += 30

    out.write(frame)
    cv2.imshow("YOLO Object Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
