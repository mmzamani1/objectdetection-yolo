import cv2
import numpy as np
import math
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
DEBOUNCE_TIME = 2 # seconds
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
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)

        if len(approx) >= 5:
            area = cv2.contourArea(cnt)
            if area > 5000:  # filter noise
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    tip = find_tip(approx)

                    if tip:
                        tipx, tipy = tip
                        # cv2.circle(frame, tip, 7, (0, 0, 255), -1)

                        direction = get_direction(cx, cy, tipx, tipy)

                        # Debounce printing
                        now = time.time()
                        if area/(w*h) > 0.03:
                            if last_steer_cmd != direction or (last_steer_cmd == direction and (now - last_print_time) > DEBOUNCE_TIME):
                                # print(f"Direction: {direction}, Area: {area:.2f}")
                                logs.append(f"Steer {direction}")
                                last_steer_cmd = direction
                                last_print_time = now

                        # Draw text on frame
                        cv2.putText(frame, f"{direction}", (cx - 50, cy - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        # cv2.putText(frame, f"Area: {int(area)}", (cx - 50, cy + 20),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detect_arrows(frame)
    add_log(result, logs[-5:])

    cv2.imshow("Arrow", result)

    if out:
        out.write(result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()