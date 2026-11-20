import cv2
import numpy as np
import torch
import os
import pickle
import time
import argparse

from client.cosine_similarity import calculate_similarity
from self_supervised.model import PalmprintEncoder

ENCODER_PATH = 'output/model/palmprint_encoder.pth'
COLLEAGUES_DB_PATH = 'output/colleagues_db.pkl'
MATCH_THRESHOLD = 0.75
STABILITY_FRAMES = 8

parser = argparse.ArgumentParser(description="Live Palm Recognition and Auto-Enrollment")
parser.add_argument("--enroll", type=str, default="", help="Name of colleague to auto-enroll (e.g. --enroll 'Rahul')")
args = parser.parse_args()

# 1. Load Neural Network
print("[*] Loading neural network model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = PalmprintEncoder().encoder
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
encoder.eval()
encoder.to(device)
print(f"[*] Model loaded on {device}.")

# 2. Database Management
def load_db():
    if os.path.exists(COLLEAGUES_DB_PATH):
        try:
            with open(COLLEAGUES_DB_PATH, "rb") as f:
                raw_db = pickle.load(f)
                migrated = {}
                for k, v in raw_db.items():
                    if "samples" in v:
                        migrated[k] = v
                    elif "embeddings" in v:
                        samples = [{"hand": "Palm", "embedding": emb, "roi": roi} 
                                   for emb, roi in zip(v["embeddings"], v["rois"])]
                        migrated[k] = {"role": v.get("role", ""), "samples": samples}
                    else:
                        migrated[k] = {
                            "role": v.get("role", ""),
                            "samples": [{"hand": "Palm", "embedding": v["embedding"], "roi": v["roi"]}]
                        }
                return migrated
        except Exception:
            return {}
    return {}

def save_db(db):
    os.makedirs(os.path.dirname(COLLEAGUES_DB_PATH), exist_ok=True)
    with open(COLLEAGUES_DB_PATH, "wb") as f:
        pickle.dump(db, f)

db = load_db()

# 3. Strict Human Palm Validation
def validate_human_palm(crop_bgr):
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    skin_ratio = np.count_nonzero(skin_mask) / (crop_bgr.shape[0] * crop_bgr.shape[1])
    if skin_ratio < 0.20:
        return False, "Not Skin (Show Palm)"

    blurred = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, "No Hand Detected"

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 7500:
        return False, "Bring Hand Closer"

    hull_pts = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull_pts)
    if hull_area == 0:
        return False, "Invalid Shape"
    solidity = float(area) / hull_area

    if solidity > 0.88:
        return False, "Not a Palm (Too Solid)"

    hull_idx = cv2.convexHull(largest, returnPoints=False)
    defect_count = 0
    if hull_idx is not None and len(hull_idx) >= 4:
        try:
            defects = cv2.convexityDefects(largest, hull_idx)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(largest[s][0])
                    end = tuple(largest[e][0])
                    far = tuple(largest[f][0])
                    depth = d / 256.0
                    if depth > 12:
                        a = np.linalg.norm(np.array(start) - np.array(far))
                        b = np.linalg.norm(np.array(end) - np.array(far))
                        c = np.linalg.norm(np.array(start) - np.array(end))
                        if a * b > 0:
                            cos_val = np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0)
                            angle = np.degrees(np.arccos(cos_val))
                            if 20 <= angle <= 95:
                                defect_count += 1
        except Exception:
            pass

    if defect_count >= 1 or (0.45 <= solidity <= 0.85):
        return True, "Palm Confirmed"

    return False, "Show Open Palm with Fingers"

# 4. Background Removal & Feature Extraction
def remove_background_and_extract_features(palm_bgr):
    gray = cv2.cvtColor(palm_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 5000:
        return None, None

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

    clean_bgr = cv2.bitwise_and(palm_bgr, palm_bgr, mask=mask)
    clean_gray = cv2.bitwise_and(gray, gray, mask=mask)

    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None, None
    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])

    half = 138
    h, w = gray.shape
    y1, y2 = max(0, cy - half), min(h, cy + half)
    x1, x2 = max(0, cx - half), min(w, cx + half)

    roi_gray = clean_gray[y1:y2, x1:x2]
    roi_bgr = clean_bgr[y1:y2, x1:x2]
    if roi_gray.size == 0:
        return None, None

    roi_std = cv2.resize(roi_gray, (138, 138))
    roi_bgr_std = cv2.resize(roi_bgr, (138, 138))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    roi_enhanced = clahe.apply(roi_std)
    roi_norm = roi_enhanced.astype(np.float32) / 255.0

    tensor = torch.from_numpy(np.expand_dims(roi_norm, axis=0)).float().repeat(1, 3, 1, 1).to(device)
    with torch.no_grad():
        embedding = encoder(tensor).cpu().numpy().flatten()

    roi_rgb = cv2.cvtColor(roi_bgr_std, cv2.COLOR_BGR2RGB)
    return embedding, roi_rgb

def match_colleague(query_emb):
    if not db:
        return None, 0.0, ""
    best_score = -1.0
    best_name, best_hand = None, ""

    for name, data in db.items():
        samples = data.get("samples", [])
        if not samples:
            continue
        embs = np.array([s["embedding"] for s in samples])
        sims = calculate_similarity(query_emb, embs)
        idx = int(np.argmax(sims))
        score = float(sims[idx])

        if score > best_score:
            best_score = score
            best_name = name
            best_hand = samples[idx].get("hand", "Palm")

    return best_name, best_score, best_hand

# 5. Start Camera Stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[!] Error: Could not open camera.")
    exit()

is_enroll_mode = bool(args.enroll.strip())
enroll_target_name = args.enroll.strip()

stages = [
    {"hand": "Left Hand", "label": "LEFT PALM"},
    {"hand": "Right Hand", "label": "RIGHT PALM"}
]
current_stage = 0
collected_samples = []
shot_cooldown = 0.0

print("=" * 60)
if is_enroll_mode:
    print(f"[*] AUTO-ENROLLMENT MODE for '{enroll_target_name}'")
    print("[*] 2-Shot Workflow: 1x Left Palm + 1x Right Palm.")
else:
    print("[*] AUTOMATIC BIOMETRIC SCANNER RUNNING")
    print("[*] Show either LEFT or RIGHT palm to be identified.")
print("[*] Press 'q' to quit.")
print("=" * 60)

steady_count = 0
last_result_text = "Waiting for palm..."
last_result_color = (200, 200, 200)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    box_size = 280
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2, y2 = x1 + box_size, y1 + box_size

    palm_crop = frame[y1:y2, x1:x2]
    is_palm, palm_msg = validate_human_palm(palm_crop)
    now = time.time()

    if is_enroll_mode:
        if current_stage < len(stages):
            stage_info = stages[current_stage]
            box_color = (0, 255, 255) if is_palm else (0, 0, 255)

            if now < shot_cooldown:
                status_msg = "Saved! Hold steady for next shot..."
                box_color = (0, 255, 0)
            elif is_palm:
                steady_count += 1
                pct = int(min(1.0, steady_count / STABILITY_FRAMES) * 100)
                status_msg = f"✋ {stage_info['label']}: Hold Steady... {pct}%"

                if steady_count >= STABILITY_FRAMES:
                    emb, clean_roi = remove_background_and_extract_features(palm_crop)
                    if emb is not None:
                        sample = {"hand": stage_info["hand"], "embedding": emb, "roi": clean_roi}
                        collected_samples.append(sample)
                        current_stage += 1
                        shot_cooldown = now + 1.8
                        print(f"[✓] Captured {stage_info['label']} (Clean, Background Removed)")
                    steady_count = 0
            else:
                steady_count = max(0, steady_count - 1)
                status_msg = f"Show {stage_info['label']} ({palm_msg})"

            if current_stage >= 2:
                # Save to database
                if enroll_target_name in db:
                    db[enroll_target_name]["samples"].extend(collected_samples)
                else:
                    db[enroll_target_name] = {"role": "", "samples": collected_samples}
                save_db(db)
                last_result_text = f"SUCCESS: BOTH PALMS ENROLLED FOR {enroll_target_name}!"
                last_result_color = (0, 255, 0)
                print(f"[✓] ENROLLMENT COMPLETE! Both palms stored for {enroll_target_name}.")
        else:
            status_msg = "ENROLLMENT COMPLETED! Press 'q' to exit."
            box_color = (0, 255, 0)
    else:
        # Identification Mode
        if is_palm and now > shot_cooldown:
            steady_count += 1
            box_color = (0, 255, 255)
            pct = int(min(1.0, steady_count / STABILITY_FRAMES) * 100)
            status_msg = f"Palm Confirmed! Hold steady... [{pct}%]"

            if steady_count >= STABILITY_FRAMES:
                emb, clean_roi = remove_background_and_extract_features(palm_crop)
                if emb is not None:
                    matched_name, score, hand_side = match_colleague(emb)
                    if score >= MATCH_THRESHOLD:
                        last_result_text = f"ACCESS GRANTED: {matched_name} ({hand_side} - {score*100:.1f}%)"
                        last_result_color = (0, 255, 0)
                    else:
                        last_result_text = f"UNKNOWN PERSON ({score*100:.1f}%)"
                        last_result_color = (0, 0, 255)
                    shot_cooldown = now + 3.0
                steady_count = 0
        else:
            if now > shot_cooldown:
                steady_count = max(0, steady_count - 1)
                box_color = (0, 0, 255) if not is_palm else (0, 255, 0)
                status_msg = f"{palm_msg}"
            else:
                box_color = last_result_color
                status_msg = "Scan Processed"

    # Draw Green Target Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    c_len = 25
    for pt1, pt2 in [((x1, y1), (x1+c_len, y1)), ((x1, y1), (x1, y1+c_len)),
                     ((x2, y1), (x2-c_len, y1)), ((x2, y1), (x2, y1+c_len)),
                     ((x1, y2), (x1+c_len, y2)), ((x1, y2), (x1, y2-c_len)),
                     ((x2, y2), (x2-c_len, y2)), ((x2, y2), (x2, y2-c_len))]:
        cv2.line(frame, pt1, pt2, box_color, 4)

    # Top Header
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
    mode_title = f"4-SHOT ENROLLMENT: {enroll_target_name}" if is_enroll_mode else "DUAL-HAND PALM BIOMETRIC SCANNER"
    cv2.putText(frame, mode_title, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, status_msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

    # Bottom Result
    cv2.rectangle(frame, (0, h - 55), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, last_result_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, last_result_color, 2)

    cv2.imshow("Palm Biometric System (Press Q to Exit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
