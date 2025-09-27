import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque

# ------------------- Config -------------------
MIN_CONFIDENCE = 0.4
DEBOUNCE_TIME = 9  # seconds
FRAME_SKIP = 1
MAKE_OUTPUT = False
POS_THRESHOLD = 40

# ------------------- Load YOLO Model -------------------
model = YOLO("E:/0CODING/MyProjects/SUB-IP/trainModel/runs/detect/progressive_training_1_mypipeline2.02/weights/best.pt")

# ------------------- I/O -------------------
input_path = "../data/ip/hobab.mkv"
filename = input_path.split("/")[-1].split(".")[0]
output_path = f"../data/outputs/{filename}_out_{time.time()}.mp4"

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"Error: Could not open video {input_path}")
    exit()

# ------------------- Get video properties -------------------
vid_fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ------------------- Prepare VideoWriter -------------------
out = None
if output_path and MAKE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, vid_fps, (width, height))

# ------------------- Variables -------------------
last_steer_cmd = None
last_print_time = 0
last_line_coords = None
logs = deque(maxlen=1)  # keep last 5 logs
frame_count = 0
prev_time = time.time()


# ------------------- Utility -------------------
def add_log(frame, logs):
    """Overlay log messages on frame"""
    for i, msg in enumerate(logs):
        cv2.putText(frame, msg, (10, 30 + (i * 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def print_progress(current, total, bar_length=30):
    """Progress bar in console"""
    fraction = current / total
    filled_length = int(bar_length * fraction)
    bar = "#" * filled_length + '-' * (bar_length - filled_length)
    print(f"\rProcessing: |{bar}| {current}/{total} frames", end='')

# ------------------- Main Loop -------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calc
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # Draw FPS
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

    # ------------------- YOLO detection -------------------
    results = model(frame, verbose=False)
    steer_command = None

    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            conf = confs[i]
            cls = classes[i]

            if conf < MIN_CONFIDENCE:
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw detection
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # green = pipeline, red = bubble
            label = "pipeline" if cls == 0 else "bubble"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)

            # If pipeline → decide steering
            if cls == 0:
                frame_center_x = width // 2
                error = cx - frame_center_x
                if abs(error) < POS_THRESHOLD:
                    steer_command = "STRAIGHT"
                elif error < 0:
                    steer_command = "LEFT"
                else:
                    steer_command = "RIGHT"

            # If leak → report
            elif cls == 1:
                cv2.putText(frame, f"Bubbles detected at ({cx},{cy})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # logs.append(f"⚠️ ALERT: Leak detected at ({cx},{cy})!")

    # Add steering log (debounced)
    now = time.time()
    if steer_command is not None and ((steer_command != last_steer_cmd) or (now - last_print_time > DEBOUNCE_TIME)):
        logs.append(f"Steer: {steer_command}")
        last_steer_cmd = steer_command
        last_print_time = now

    # Overlay logs
    add_log(frame, logs)

    # Show
    cv2.imshow("Pipeline & Leak Detection", frame)
    if out:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
