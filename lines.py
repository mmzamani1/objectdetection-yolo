import cv2
import numpy as np
import time

input_path = "E:/0CODING/MyProjects/SUB-IP/data/QT/IMG_0798.MOV"
filename = input_path.split("/")[-1].split(".")[0]
output_path = f"./outputs/{filename}_out.mp4"

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

# Configs
last_steer_cmd = None
last_print_time = 0
DEBOUNCE_TIME = 3  # seconds
logs = []

def add_log(frame, logs):
    for i, msg in enumerate(logs):
        cv2.putText(frame, msg, (10, 30+(i*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_center = (w // 2, h)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                            minLineLength=80, maxLineGap=20)

    steering_angle = None
    steer_command = None

    if lines is not None:
        # Pick the line closest to the bottom of the frame
        chosen_line = max(lines, key=lambda l: max(l[0][1], l[0][3]))
        x1, y1, x2, y2 = chosen_line[0]

        # Draw chosen line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Compute angle (in degrees relative to horizontal axis)
        dx, dy = x2 - x1, y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))

        # Convert to steering relative to vertical (0 = straight)
        steering_angle = -angle + 90  # negative = left, positive = right

        if steering_angle <= 5 or steering_angle >= 175:
            steer_command = "Forward"
        elif 90 <= steering_angle <= 180:
            steer_command = "Right"
        elif 0 <= steering_angle <= 90:
            steer_command = "Left"

        # Debounce logging
        now = time.time()
        if steer_command and (now - last_print_time) > DEBOUNCE_TIME:
            logs.append(f"Steer: {steer_command}")
            last_steer_cmd = steer_command
            last_print_time = now

    add_log(frame, logs[-5:])

    cv2.imshow("Line Following", frame)

    if out:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
