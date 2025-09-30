import cv2
import numpy as np
import time
from ultralytics import YOLO
import math
from collections import deque
import pytesseract
import re

# ------------------- Config -------------------
MIN_CONFIDENCE = 0.4
DEBOUNCE_TIME = 9 # seconds
FRAME_SKIP = 1
MAKE_OUTPUT = False
OBJ_TO_FRAME_RATIO = 0.03
ARROW_COUNT = 3
OCR_ENABLED = False

# ------------------- Load YOLO Model -------------------
model = YOLO("weights/myarrow.pt")

# ------------------- I/O -------------------
input_path = "../data/qt/arrow.mp4"
filename = input_path.split("/")[-1].split(".")[0]
output_path = f"../data/outputs/{filename}_out_{time.time()}.mp4"

# ------------------- Load Input -------------------
cap = cv2.VideoCapture(0)
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
last_steer_cmd = None
last_print_time = 0
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = 0
prev_time = time.time()
logs = []
logs = deque(maxlen=1)  # auto-trims old logs
# text = ''

# ------------------- Functions -------------------
def add_log(frame, logs):
    for i, msg in enumerate(logs):
        cv2.putText(frame, msg, (50, 30+(i*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def print_progress(current, total, bar_length=30):
    fraction = current / total
    filled_length = int(bar_length * fraction)
    bar = "#" * filled_length + '-' * (bar_length - filled_length)
    print(f"\rProcessing: |{bar}| {current}/{total} frames", end='')

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
    frame_area = height * width
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

        if area / frame_area > OBJ_TO_FRAME_RATIO:
            detected_dir = direction
            
        # Draw direction text
        cv2.putText(frame, direction, (cx - 30, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame, detected_dir

def read_text_from_roi(roi):
    """
    Performs OCR on a cropped ROI.
    roi: BGR image (numpy array)
    Returns detected text
    """
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Optional: thresholding to improve OCR
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR
    text = pytesseract.image_to_string(thresh, config='--psm 6')  # PSM 6 = Assume a single uniform block of text
    text = text.strip()
    return text

def process_ocr_text(text):
    """
    Check if OCR text is a math expression.
    If yes -> evaluate and return result.
    If not -> return text itself.
    """
    # Clean text (remove spaces, newlines)
    expr = text.replace(" ", "").replace("\n", "")

    # Regex: only allow digits, + - * / ^ ( )
    if re.fullmatch(r"[0-9\+\-\*/\^\(\)\.]+", expr):
        try:
            # Replace ^ with ** for Python power
            expr = expr.replace("^", "**")
            result = eval(expr, {"__builtins__": {}})
            return f"{text} = {result}"
        except Exception:
            return text  # If eval fails, return as plain text
    else:
        return text

# ------------------- Main Loop -------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if len(logs) == ARROW_COUNT:
        OCR_ENABLED = True
    
    if OCR_ENABLED:
        text = read_text_from_roi(frame)
        processed_text = process_ocr_text(text)

        cv2.putText(annotated_frame, processed_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (159, 255, 255), 2)
        print(processed_text)  # also log in console
    
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

    # YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

    filtered_detections = [(x1, y1, x2, y2, conf, cls)
                           for x1, y1, x2, y2, conf, cls in detections
                           if conf > MIN_CONFIDENCE]


    # Annotate
    annotated_frame = frame.copy()
    for x1, y1, x2, y2, conf, cls in filtered_detections:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # convert YOLO coords to int

        # Crop ROI safely
        roi = frame[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]

        if roi.size > 0:
            roi_vis, direction = detect_arrows(roi.copy())
            annotated_frame[y1:y2, x1:x2] = roi_vis
            
            
        # Debounce
        now = time.time()
        if direction != None and (last_steer_cmd != direction or (now - last_print_time > DEBOUNCE_TIME)):
            msg = f"Steer {direction}"
            logs.append(msg)
            last_steer_cmd = direction
            last_print_time = now

            
        # Draw YOLO bounding box 
        label = f"arrow {conf:.2f}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    add_log(annotated_frame, logs)
    
    
    

    # Optional display for debugging
    cv2.imshow("Filtered YOLO Detections", annotated_frame)
    

    if out:
        out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
if out:
    out.release()
cv2.destroyAllWindows()

