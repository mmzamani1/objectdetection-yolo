import cv2
import numpy as np
import math
import time 

cap = cv2.VideoCapture("E:/0CODING/MyProjects/SUB-IP/data/QT/IMG_0798.MOV")  

# Debounce storage
last_steer_cmd = None
last_print_time = 0
DEBOUNCE_TIME = 3  # seconds


def get_direction(cx, cy, tipx, tipy):
    dx = tipx - cx
    dy = tipy - cy

    angle = math.degrees(math.atan2(dy, dx))

    if -45 <= angle <= 45:
        return "Left"
    elif 45 < angle <= 135:
        return "Up"
    elif angle > 135 or angle < -135:
        return "Right"
    else:
        return "Down"


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

                    # Arrow tip = farthest vertex
                    max_dist = 0
                    tip = None
                    for point in approx:
                        px, py = point[0]
                        dist = np.sqrt((px - cx)**2 + (py - cy)**2)
                        if dist > max_dist:
                            max_dist = dist
                            tip = (px, py)

                    if tip:
                        tipx, tipy = tip
                        cv2.circle(frame, tip, 7, (0, 0, 255), -1)

                        direction = get_direction(cx, cy, tipx, tipy)

                        # Debounce printing
                        now = time.time()
                        if (now - last_print_time > DEBOUNCE_TIME) and (area/(w*h) > 0.03):
                            print(f"Direction: {direction}, Area: {area:.2f}")
                            last_steer_cmd = direction
                            last_print_time = now

                        # Draw text on frame
                        cv2.putText(frame, f"{direction}", (cx - 50, cy - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(frame, f"Area: {int(area)}", (cx - 50, cy + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detect_arrows(frame)

    cv2.imshow("Arrow", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
