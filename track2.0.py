import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

import time
import math
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ------------------- CONFIG -------------------
INPUT_VIDEO = "E:/0CODING/MyProjects/SUB-IP/data/OBJ-final.mp4"
MODEL_PATH = "E:/0CODING/MyProjects/SUB-IP/trainModel/runs/detect/progressive_training_1_shapes2/weights/best.pt"

TARGET_FPS = 10
FRAME_INTERVAL = 1.0 / TARGET_FPS

DETECTION_INTERVAL_FRAMES = 1   # run YOLO every N frames (tune)
TRACKER_TIMEOUT_FRAMES = 30      # frames before removing a tracker
DETECTION_CONF_THR = 0.35
IOU_MATCH_THR = 0.3

USE_CUDA = torch.cuda.is_available()
USE_HALF = USE_CUDA  # enable fp16 on GPU for speed
SHOW_WINDOW = True     # set False to disable cv2.imshow and speed up
PREPROCESS = True      # basic enhancement (CLAHE)
TRACKER_TYPE = "KCF"   # "CSRT" (accurate) or "KCF" (faster)

# ------------------- INIT MODEL -------------------
device = "cuda" if USE_CUDA else "cpu"
model = YOLO(MODEL_PATH)
if USE_CUDA:
    try:
        model.to(device)
        if USE_HALF:
            model.model.half()  # only if model supports it; speeds up inference on GPU
    except Exception:
        pass
model.eval()

# ------------------- VIDEO -------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit(f"Error: could not open {INPUT_VIDEO}")

orig_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Optionally write output (disabled here by default)
WRITE_OUTPUT = False
OUT_PATH = "./out_opt.mp4"
if WRITE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, orig_fps, (video_width, video_height))
else:
    writer = None

# ------------------- TRACKER UTIL -------------------
if TRACKER_TYPE == "CSRT":
    def create_tracker(): return cv2.legacy.TrackerCSRT_create()
else:
    def create_tracker(): return cv2.legacy.TrackerKCF_create()

tracker_dict = {}   # id -> {tracker, class, conf, inactive_count, color}
next_tracker_id = 0

def init_tracker(frame, bbox_xywh, class_name, conf, tracker_id=None):
    """Initialize and store a tracker. bbox_xywh is (x, y, w, h)."""
    global next_tracker_id
    tr = create_tracker()
    tr.init(frame, bbox_xywh)
    if tracker_id is None:
        tracker_id = next_tracker_id
        next_tracker_id += 1
    tracker_dict[tracker_id] = {
        "tracker": tr,
        "class": class_name,
        "conf": conf,
        "inactive_count": 0,
        "color": tuple(np.random.randint(50, 230, 3).tolist())
    }
    return tracker_id

def update_trackers(frame):
    """Update all trackers. Returns list of (id, bbox_xywh, p1, p2)."""
    remove_ids = []
    results = []
    for tid, meta in list(tracker_dict.items()):
        success, bbox = meta["tracker"].update(frame)
        if not success:
            meta["inactive_count"] += 1
            if meta["inactive_count"] > TRACKER_TIMEOUT_FRAMES:
                remove_ids.append(tid)
            continue
        else:
            meta["inactive_count"] = 0
            x, y, w, h = map(int, bbox)
            p1 = (x, y)
            p2 = (x + w, y + h)
            results.append((tid, (x, y, w, h), p1, p2))
    for tid in remove_ids:
        del tracker_dict[tid]
    return results

# ------------------- IOU MATCH (small optimizations) -------------------
def iou_xyxy(d, t):
    """IoU for two boxes in x1,y1,x2,y2"""
    d_x1, d_y1, d_x2, d_y2 = d
    t_x1, t_y1, t_x2, t_y2 = t
    inter_x1 = max(d_x1, t_x1)
    inter_y1 = max(d_y1, t_y1)
    inter_x2 = min(d_x2, t_x2)
    inter_y2 = min(d_y2, t_y2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_d = (d_x2 - d_x1) * (d_y2 - d_y1)
    area_t = (t_x2 - t_x1) * (t_y2 - t_y1)
    return inter / (area_d + area_t - inter + 1e-9)

def match_detections_to_trackers(detections, tracked_boxes, iou_threshold=IOU_MATCH_THR):
    """
    detections: list of (cls, conf, x1, y1, x2, y2, class_name, area)
    tracked_boxes: list returned from update_trackers -> (tid, (x,y,w,h), p1, p2)
    returns: matched_pairs (d_idx, t_idx), unmatched_detection_indices, unmatched_tracked_indices
    """
    if len(detections) == 0:
        return [], [], list(range(len(tracked_boxes)))
    if len(tracked_boxes) == 0:
        return [], list(range(len(detections))), []

    # build arrays
    det_boxes = np.array([[d[2], d[3], d[4], d[5]] for d in detections], dtype=float)
    tr_boxes = np.array([[t[1][0], t[1][1], t[1][0] + t[1][2], t[1][1] + t[1][3]] for t in tracked_boxes], dtype=float)

    iou_mat = np.zeros((len(detections), len(tracked_boxes)), dtype=float)
    for i in range(len(detections)):
        for j in range(len(tracked_boxes)):
            iou_mat[i, j] = iou_xyxy(det_boxes[i], tr_boxes[j])

    matched_pairs = []
    unmatched_dets = set(range(len(detections)))
    unmatched_trks = set(range(len(tracked_boxes)))

    # greedy matching by highest IoU
    while True:
        idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        best_iou = iou_mat[idx]
        if best_iou < iou_threshold:
            break
        d_i, t_i = idx
        matched_pairs.append((int(d_i), int(t_i)))
        iou_mat[d_i, :] = -1
        iou_mat[:, t_i] = -1
        unmatched_dets.discard(d_i)
        unmatched_trks.discard(t_i)

    return matched_pairs, list(unmatched_dets), list(unmatched_trks)

# ------------------- PREPROCESS -------------------
def preprocess(frame):
    if not PREPROCESS:
        return frame
    # Resize if needed for speed (keep ratio)
    # small enhancement using CLAHE on L channel
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ------------------- MAIN LOOP -------------------
frame_idx = 0
last_detection_frame = -999
start_time = time.time()

while cap.isOpened():
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    proc_frame = preprocess(frame)  # lightweight enhancement
    display_frame = proc_frame.copy()

    # update trackers (fast)
    tracked = update_trackers(proc_frame)

    # Draw tracked boxes
    for tid, bbox, p1, p2 in tracked:
        meta = tracker_dict[tid]
        color = meta["color"]
        label = f"ID:{tid} {meta['class']} {meta['conf']:.2f}"
        cv2.rectangle(display_frame, p1, p2, color, 2)
        cv2.putText(display_frame, label, (p1[0], max(15, p1[1]-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Decide whether to run detection this frame
    run_detection = (frame_idx - last_detection_frame) >= DETECTION_INTERVAL_FRAMES or len(tracker_dict) == 0

    if run_detection:
        last_detection_frame = frame_idx
        # Ultralytics call - minimize return processing
        # We use model.predict to configure batch and device internally, but model(frame) works too.
        try:
            # Using model(frame, imgsz=...) can speed up but rely on model defaults
            results = model.predict(proc_frame, verbose=False, conf=DETECTION_CONF_THR, device=device)
        except Exception:
            # fallback
            results = model(proc_frame, verbose=False)

        # Extract detections (if any)
        detection_list = []
        if len(results) > 0:
            res = results[0]
            if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()     # [N,4]
                confs = res.boxes.conf.cpu().numpy()    # [N]
                clss = res.boxes.cls.cpu().numpy().astype(int)  # [N]
                for x1, y1, x2, y2, conf, cls in zip(xyxy[:,0], xyxy[:,1], xyxy[:,2], xyxy[:,3], confs, clss):
                    if conf < DETECTION_CONF_THR:
                        continue
                    area = (x2-x1)*(y2-y1)
                    detection_list.append((int(cls), float(conf), int(x1), int(y1), int(x2), int(y2), model.names[int(cls)], area))

        # Match detections -> trackers
        matched_pairs, unmatched_dets, unmatched_trks = match_detections_to_trackers(detection_list, tracked)

        # For matched pairs: reinit tracker with detection bbox (correct drift)
        for d_idx, t_idx in matched_pairs:
            tid = tracked[t_idx][0]
            cls, conf, x1, y1, x2, y2, class_name, _ = detection_list[d_idx]
            bbox_xywh = (x1, y1, x2 - x1, y2 - y1)
            # Re-initialize the existing tracker with the new bbox (helps correct drift)
            init_tracker(proc_frame, bbox_xywh, class_name, conf, tracker_id=tid)

        # For unmatched detections: create new trackers
        for d_idx in unmatched_dets:
            cls, conf, x1, y1, x2, y2, class_name, _ = detection_list[d_idx]
            if conf >= DETECTION_CONF_THR:
                bbox_xywh = (x1, y1, x2 - x1, y2 - y1)
                init_tracker(proc_frame, bbox_xywh, class_name, conf)

    # OPTIONAL: send tracker updates to hardware or log
    # Example: print tracker centers
    for tid, bbox, p1, p2 in tracked:
        cx = int(bbox[0] + bbox[2] / 2)
        cy = int(bbox[1] + bbox[3] / 2)
        # print(f"Tracker {tid}: center=({cx},{cy}) class={tracker_dict[tid]['class']}")
        # If you want to send via serial: ser.write(f"...".encode())

    # display / write
    if SHOW_WINDOW:
        cv2.imshow("Optimized Detection+Tracking", display_frame)
    if writer is not None:
        writer.write(display_frame)

    # timing control to approximate TARGET_FPS
    elapsed = time.time() - t0
    sleep_time = max(0.0, FRAME_INTERVAL - elapsed)
    if sleep_time > 0:
        time.sleep(sleep_time)

    if SHOW_WINDOW and (cv2.waitKey(1) & 0xFF == ord("q")):
        break

# cleanup
cap.release()
if writer:
    writer.release()
if SHOW_WINDOW:
    cv2.destroyAllWindows()
print("Finished.")
