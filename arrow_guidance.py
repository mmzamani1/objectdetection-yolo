import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO

# ------------------- Config -------------------
TARGET_FPS = 50
FRAME_INTERVAL = 1.0 / TARGET_FPS
MIN_CONFIDENCE = 0.4
last_move_cmd = None
last_print_time = 0
DEBOUNCE_TIME = 0.5  # seconds

# Load YOLO Model
model = YOLO("E:/0CODING/MyProjects/SUB-IP/prj-files/arrowWeights.pt")
model.eval()
model.to("cuda").half()

# Open camera (0 = webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open camera")
    exit()

# ------------------- Preprocess -------------------
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
        label = f"{model.names[int(cls)]} {conf:.2f}"
        color = (0, 255, 0) if (best_det and (x1, y1, x2, y2, conf, cls) == best_det) else (255, 0, 0)

        # Draw bounding box
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Label text
        cv2.putText(annotated_frame, label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ---------------- Debounced Print ----------------
    now = time.time()
    if move_command != "null":
        cv2.putText(annotated_frame, f"Move: {move_command}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if move_command != last_move_cmd and (now - last_print_time > DEBOUNCE_TIME):
            print(f"{move_command}")
            last_move_cmd = move_command
            last_print_time = now

    # Show result
    cv2.imshow("Filtered YOLO Detections", annotated_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # FPS control
    elapsed = time.time() - start_time
    sleep_time = max(0, FRAME_INTERVAL - elapsed)
    time.sleep(sleep_time)

cap.release()
cv2.destroyAllWindows()
