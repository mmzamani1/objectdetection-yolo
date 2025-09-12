import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

# Debounce storage
last_steer_cmd = None
last_print_time = 0
DEBOUNCE_TIME = 0.5  # seconds

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
        
        # Draw steering info
        cv2.putText(frame, f"{steer_command}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Debounce print
        now = time.time()
        if steer_command != last_steer_cmd and (now - last_print_time > DEBOUNCE_TIME):
            print(f"Steering {steer_command} Angle: {steering_angle:.2f} degrees")
            last_steer_cmd = steer_command
            last_print_time = now

    cv2.imshow("Line Following", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
