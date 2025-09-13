import cv2
import numpy as np
import time
# import torch
from ultralytics import YOLO
import math

# ------------------- Config -------------------
TARGET_FPS = 5000
FRAME_INTERVAL = 1.0 / TARGET_FPS
MIN_CONFIDENCE = 0.4

# Load YOLO Model
model = YOLO("./arrowWeights.pt")
model.eval()
# model.to("cuda").half()


input_path = "./arrow.mp4"
filename = input_path.split("/")[-1].split(".")[0]
output_path = f"../data/outputs/{filename}_out_4.mp4"

# Open camera (0 = webcam)
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("❌ Error: Could not open camera")
    exit()




# Get video properties
# fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare VideoWriter
# out = None
# if output_path:
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Configs
last_steer_cmd = None
last_print_time = 0
DEBOUNCE_TIME = 9 # seconds
logs = []

def add_log(frame, logs):
    for i, msg in enumerate(logs):
        cv2.putText(frame, msg, (10, 30+(i*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def get_direction(cx, cy, tipx, tipy):
    dx = tipx - cx
    dy = -(tipy - cy)
    angle = math.degrees(math.atan2(dy, dx))
    if -90 > angle > -180 or 180 > angle > 90:
        return "Left"
    else:
        return "Right"

def find_tip(points):
    min_angle = 360
    tip = None
    if len(points) < 3:
        return None
    for i in range(len(points)):
        p_prev = points[(i - 1) % len(points)][0]
        p_curr = points[i][0]
        p_next = points[(i + 1) % len(points)][0]
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        dot_product = np.dot(v1, v2)
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        if mag1 * mag2 == 0:
            continue
        cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
        angle = math.acos(cos_angle)
        angle_deg = math.degrees(angle)
        if angle_deg < min_angle:
            min_angle = angle_deg
            tip = tuple(p_curr)
    return tip

def detect_arrows(frame):
    global last_steer_cmd, last_print_time  
    h, w = frame.shape[:2]
    detected_dir = None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame, None

    # Pick largest contour in ROI
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 300:  # lower threshold for ROI
        return frame, None

    approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

    # Centroid
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return frame, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Arrow tip (sharpest corner)
    tip = find_tip(approx)
    if tip:
        tipx, tipy = tip
        cv2.circle(frame, (tipx, tipy), 6, (0, 0, 255), -1)

        direction = get_direction(cx, cy, tipx, tipy)
        detected_dir = direction

        # Debounce
        now = time.time()
        print(f"Steer {direction} Area: {area}")
        if (last_steer_cmd != direction or (now - last_print_time > DEBOUNCE_TIME)) and area > 13000:
            print(f"Steer {direction} Area: {area}")
            logs.append(f"Steer {direction}")
            last_steer_cmd = direction
            last_print_time = now

        # Draw direction text
        cv2.putText(frame, direction, (cx - 30, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame, detected_dir


# ------------------- Preprocess -------------------
def preprocess(frame):
    # --- Convert to LAB ---
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # --- Gentle white balance ---
    avg_a, avg_b = np.mean(a), np.mean(b)
    shift_a = int(np.clip(128 - avg_a, -10, 10))
    shift_b = int(np.clip(128 - avg_b, -10, 10))
    a = cv2.add(a, shift_a)
    b = cv2.add(b, shift_b)

    # --- CLAHE on L channel ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- Lightweight sharpening ---
    sharpen_kernel = np.array([[0, -0.5, 0],
                               [-0.5, 3, -0.5],
                               [0, -0.5, 0]], dtype=np.float32)
    result = cv2.filter2D(result, -1, sharpen_kernel)

    return frame



# ------------------- Move Command -------------------
def get_move_command(filtered_detections):
    if not filtered_detections:
        return "null", None

    # Pick highest confidence detection
    x1, y1, x2, y2, conf, cls = max(filtered_detections, key=lambda x: x[4])
    class_name = model.names[int(cls)]

    if class_name == "arrow_left":
        return "left", (x1, y1, x2, y2, conf, cls)
    elif class_name == "arrow_right":
        return "right", (x1, y1, x2, y2, conf, cls)
    elif class_name == "arrow_up":
        return "up", (x1, y1, x2, y2, conf, cls)
    elif class_name == "arrow_down":
        return "down", (x1, y1, x2, y2, conf, cls)
    else:
        return "null", None


# ------------------- Main Loop -------------------
while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = preprocess(frame)
    
    # Run YOLO
    results = model(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

    # Filter by confidence
    filtered_detections = [
        (x1, y1, x2, y2, conf, cls)
        for x1, y1, x2, y2, conf, cls in detections
        if conf > MIN_CONFIDENCE
    ]

    # Get movement command
    move_command, best_det = get_move_command(filtered_detections)

    # ---------------- Annotate frame ----------------
    annotated_frame = frame.copy()

    for x1, y1, x2, y2, conf, cls in filtered_detections:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # convert YOLO coords to int

        # Crop ROI safely
        roi = frame[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]
        if roi.size == 0:
            continue

        # roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            roi_vis, direction = detect_arrows(roi.copy())
            annotated_frame[y1:y2, x1:x2] = roi_vis
        # Draw YOLO bounding box (just for visualization)
        label = f"arrow {conf:.2f}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ---------------- Debounced Print ----------------
    now = time.time()
    if move_command != "null":
        logs.append(move_command)
        if move_command != last_move_cmd and (now - last_print_time > DEBOUNCE_TIME):
            print(f"{move_command}")
            last_move_cmd = move_command
            last_print_time = now

    add_log(annotated_frame, logs[-5:])

    # Show result
    cv2.imshow("Filtered YOLO Detections", annotated_frame)
    # if out:
    #     out.write(annotated_frame)
    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # FPS control
    elapsed = time.time() - start_time
    sleep_time = max(0, FRAME_INTERVAL - elapsed)
    time.sleep(sleep_time)

cap.release()
# if out:
#     out.release()
cv2.destroyAllWindows()
