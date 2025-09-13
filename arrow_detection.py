import cv2
import numpy as np
import math
import time

# --- Configuration ---
VIDEO_PATH = "E:/0CODING/MyProjects/SUB-IP/data/QT/IMG_0798.MOV"
OUTPUT_PATH = "E:/0CODING/MyProjects/SUB-IP/data/QT/IMG_0798_output.mp4"
DEBOUNCE_TIME = 3  # seconds

# --- Debounce Storage ---
last_print_time = 0

# --- Functions ---
def get_direction(cx, cy, tipx, tipy):
    dx = tipx - cx
    dy = -(tipy - cy)
    angle = math.degrees(math.atan2(dy, dx))
    if -90 > angle > -180 or 180 > angle > 90:
        return "Left", angle
    else:
        return "Right", angle

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
    global last_print_time
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if 4 <= len(approx) <= 10:
            area = cv2.contourArea(cnt)
            relative_area = area / (w * h)
            if 0.005 < relative_area < 0.5:
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    tip = find_tip(approx)
                    if tip:
                        tipx, tipy = tip
                        cv2.circle(frame, tip, 7, (0, 0, 255), -1)
                        direction, angle = get_direction(cx, cy, tipx, tipy)
                        now = time.time()
                        if (now - last_print_time > DEBOUNCE_TIME):
                            print(f"Direction: {direction}, Relative Area: {relative_area:.3f} Angle: {angle}")
                            last_print_time = now
                        cv2.putText(frame, f"{direction} Angle: {angle:.1f}", (cx - 50, cy - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(frame, f"Area: {int(area)}", (cx - 50, cy + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame

# --- Main Loop ---
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}")
        return

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        result_frame = detect_arrows(frame)
        out.write(result_frame)  # Save frame to output video
        cv2.imshow("Arrow Detection", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
