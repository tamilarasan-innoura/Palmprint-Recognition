import cv2
import numpy as np
import torch
import os
import pickle
import time
import json
import uuid
import winsound
from datetime import datetime

from self_supervised.model import PalmprintEncoder

# ================= CONFIGURATION =================
ENCODER_PATH = 'output/model/palmprint_encoder.pth'
DB_PATH = 'output/colleagues_db.pkl'
AUDIT_LOG_PATH = 'output/audit_logs.json'
MATCH_THRESHOLD = 0.75
STABILITY_FRAMES = 4
BOX_SIZE = 280

# ================= 1. LOAD MODEL & TEMPLATES =================
print("\n" + "=" * 65)
print("  🛡️  PALMGUARD: DEDICATED BIOMETRIC AUTHENTICATION TERMINAL")
print("=" * 65)
print("[*] Initializing neural network and biometric matcher...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = PalmprintEncoder().encoder
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
encoder.eval()
encoder.to(device)
print(f"[*] Deep neural network loaded on: {device}")

def load_db():
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "rb") as f:
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

db = load_db()
print(f"[*] Authorized Employee Templates Loaded: {len(db)}")
for u in db.keys():
    print(f"    - User: {u} ({len(db[u].get('samples', []))} palmprints registered)")
print("=" * 65 + "\n")

def log_event(user_name, hand, score, status):
    event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user_name,
        "hand": hand,
        "confidence": f"{score*100:.1f}%",
        "status": status,
        "session_token": str(uuid.uuid4())[:8].upper() if status == "GRANTED" else "N/A"
    }
    logs = []
    if os.path.exists(AUDIT_LOG_PATH):
        try:
            with open(AUDIT_LOG_PATH, "r") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.insert(0, event)
    logs = logs[:200]
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    with open(AUDIT_LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)
    return event

# ================= 2. BIOMETRIC PIPELINE =================
def classify_hand_side(contour, palm_cx, palm_cy):
    pts = contour[:, 0, :]
    left_pts = pts[pts[:, 0] < palm_cx]
    right_pts = pts[pts[:, 0] > palm_cx]
    
    left_lower = left_pts[left_pts[:, 1] > (palm_cy - 20)] if len(left_pts) > 0 else []
    right_lower = right_pts[right_pts[:, 1] > (palm_cy - 20)] if len(right_pts) > 0 else []
    
    left_spread = (palm_cx - np.min(left_lower[:, 0])) if len(left_lower) > 0 else 0
    right_spread = (np.max(right_lower[:, 0]) - palm_cx) if len(right_lower) > 0 else 0
    
    if left_spread > right_spread + 14:
        return "Left Hand"
    elif right_spread > left_spread + 14:
        return "Right Hand"
    return "Palm"

def fast_palm_check(crop_bgr):
    h_crop, w_crop, _ = crop_bgr.shape
    gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Lighting check
    brightness = float(np.mean(gray_crop))
    if brightness < 20:
        return False, "Room Too Dark", None, None, "Palm"
    if brightness > 250:
        return False, "Severe Glare", None, None, "Palm"

    # Dual-color space skin filter (HSV + YCrCb)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    mask_hsv = cv2.inRange(hsv, np.array([0, 18, 30]), np.array([35, 255, 255]))
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 120, 60]), np.array([255, 190, 145]))
    skin_mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    skin_ratio = np.count_nonzero(skin_mask) / (h_crop * w_crop)
    if skin_ratio < 0.10:
        return False, "Show Open Palm in Box", None, None, "Palm"

    # Contour
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, "Align Palm in Box", None, None, "Palm"

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 5000:
        return False, "Bring Hand Closer", None, largest, "Palm"

    # Extract centered palmprint ROI
    m = cv2.moments(largest)
    if m["m00"] > 0:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
    else:
        cx, cy = w_crop // 2, h_crop // 2

    cx = int(np.clip(cx, 70, w_crop - 70))
    cy = int(np.clip(cy, 70, h_crop - 70))

    half = 70
    y1, y2 = max(0, cy - half), min(h_crop, cy + half)
    x1, x2 = max(0, cx - half), min(w_crop, cx + half)

    roi_bgr = crop_bgr[y1:y2, x1:x2]
    if roi_bgr.shape[0] < 30 or roi_bgr.shape[1] < 30:
        roi_bgr = crop_bgr[30:250, 30:250]

    roi_std_bgr = cv2.resize(roi_bgr, (138, 138))
    clean_roi_rgb = cv2.cvtColor(roi_std_bgr, cv2.COLOR_BGR2RGB)
    side = classify_hand_side(largest, cx, cy)

    return True, f"{side} Locked!", clean_roi_rgb, largest, side

def compute_embedding(clean_roi_rgb):
    gray = cv2.cvtColor(clean_roi_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    roi_enhanced = clahe.apply(gray)
    roi_norm = roi_enhanced.astype(np.float32) / 255.0

    tensor = torch.from_numpy(np.expand_dims(roi_norm, axis=0)).float().repeat(1, 3, 1, 1).to(device)
    with torch.no_grad():
        embedding = encoder(tensor).cpu().numpy().flatten()
    
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding

def match_colleague(query_emb):
    if not db:
        return None
    best_score = -1.0
    best_match = None

    for name, data in db.items():
        samples = data.get("samples", [])
        if not samples:
            continue
        hand_scores = []
        user_best = -1.0
        user_hand = "Palm"

        for s in samples:
            emb = s["embedding"]
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            sim = float(np.dot(query_emb, emb_norm))
            h_label = s.get("hand", "Palm")
            hand_scores.append((h_label, sim))
            if sim > user_best:
                user_best = sim
                user_hand = h_label

        if user_best > best_score:
            best_score = user_best
            best_match = {
                "name": name,
                "role": data.get("role", ""),
                "hand": user_hand,
                "score": best_score,
                "all_scores": hand_scores
            }
    return best_match

# ================= 3. TERMINAL SCANNER LOOP =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

steady_count = 0
cooldown_until = 0.0
last_result_time = 0.0
last_result_type = None
last_result_msg = ""
last_result_detail = ""

window_name = "PalmGuard | Biometric Authentication Terminal"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960, 720)

scan_beam_y = 0
scan_beam_dir = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    x1 = (w - BOX_SIZE) // 2
    y1 = (h - BOX_SIZE) // 2
    x2, y2 = x1 + BOX_SIZE, y1 + BOX_SIZE
    palm_crop = frame[y1:y2, x1:x2]

    is_palm, palm_msg, clean_roi, contour, hand_side = fast_palm_check(palm_crop)
    now = time.time()

    # Color Scheme
    # Idle/Searching: Cyan (255, 255, 0)
    # Locked/Verifying: Yellow (0, 255, 255)
    # Granted: Bright Neon Green (0, 255, 0)
    # Denied: Bright Red (0, 0, 255)

    if now <= cooldown_until:
        if last_result_type == "GRANTED":
            box_color = (0, 255, 0)
            status_text = "VERIFIED: ACCESS GRANTED"
        else:
            box_color = (0, 0, 255)
            status_text = "SECURITY ALERT: ACCESS DENIED"
    elif is_palm:
        steady_count += 1
        pct = int(min(1.0, steady_count / STABILITY_FRAMES) * 100)
        box_color = (0, 255, 255)
        status_text = f"✋ {hand_side.upper()} LOCKED: SCANNING {pct}%"

        # Moving laser scanner beam animation
        scan_beam_y += (6 * scan_beam_dir)
        if scan_beam_y > BOX_SIZE - 10:
            scan_beam_dir = -1
        elif scan_beam_y < 10:
            scan_beam_dir = 1
        cv2.line(frame, (x1 + 10, y1 + scan_beam_y), (x2 - 10, y1 + scan_beam_y), (0, 255, 255), 2)

        if steady_count >= STABILITY_FRAMES:
            emb = compute_embedding(clean_roi)
            match = match_colleague(emb)

            if match and match["score"] >= MATCH_THRESHOLD:
                log_event(match["name"], match["hand"], match["score"], "GRANTED")
                last_result_type = "GRANTED"
                last_result_msg = f"WELCOME: {match['name'].upper()}"
                last_result_detail = f"Verified: {match['hand']} | Confidence: {match['score']*100:.1f}%"
                print(f"[✅ GRANTED] {match['name']} ({match['hand']}) - Match Score: {match['score']*100:.1f}%")
                try:
                    winsound.Beep(1800, 150)
                except Exception:
                    pass
            else:
                score_str = f"{match['score']*100:.1f}%" if match else "0.0%"
                log_event("Unknown", "Unknown", match["score"] if match else 0.0, "DENIED")
                last_result_type = "DENIED"
                last_result_msg = "ACCESS DENIED - UNRECOGNIZED PALM"
                last_result_detail = f"Score: {score_str} (Required: {int(MATCH_THRESHOLD*100)}%)"
                print(f"[❌ DENIED] Unrecognized Palm - Match Score: {score_str}")
                try:
                    winsound.Beep(450, 250)
                except Exception:
                    pass

            cooldown_until = now + 3.5
            steady_count = 0
    else:
        steady_count = max(0, steady_count - 1)
        box_color = (255, 200, 0)
        status_text = f"ALIGN PALM IN TARGET: {palm_msg}"

    # Draw Futuristic Biometric Target Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    c_len = 30
    for pt1, pt2 in [((x1, y1), (x1+c_len, y1)), ((x1, y1), (x1, y1+c_len)),
                     ((x2, y1), (x2-c_len, y1)), ((x2, y1), (x2, y1+c_len)),
                     ((x1, y2), (x1+c_len, y2)), ((x1, y2), (x1, y2-c_len)),
                     ((x2, y2), (x2-c_len, y2)), ((x2, y2), (x2, y2-c_len))]:
        cv2.line(frame, pt1, pt2, box_color, 5)

    # Top Status Banner
    cv2.rectangle(frame, (0, 0), (w, 55), (20, 25, 35), -1)
    cv2.putText(frame, "PALMGUARD BIOMETRIC ACCESS TERMINAL", (20, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 220, 255), 2)
    cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (w - 130, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1)

    # Bottom Target Status
    cv2.rectangle(frame, (0, h - 55), (w, h), (20, 25, 35), -1)
    cv2.putText(frame, status_text, (20, h - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, box_color, 2)

    # Result Splash Card
    if now <= cooldown_until:
        card_w, card_h = 600, 100
        cx1, cy1 = (w - card_w) // 2, y1 - 120
        cx2, cy2 = cx1 + card_w, cy1 + card_h
        bg_col = (10, 60, 15) if last_result_type == "GRANTED" else (10, 15, 70)
        border_col = (0, 255, 0) if last_result_type == "GRANTED" else (0, 0, 255)

        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), bg_col, -1)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), border_col, 3)
        cv2.putText(frame, last_result_msg, (cx1 + 25, cy1 + 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, border_col, 2)
        cv2.putText(frame, last_result_detail, (cx1 + 25, cy1 + 78),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (240, 240, 240), 1)

    # Inset Biometric Telemetry (Top-Right Corner)
    if clean_roi is not None:
        gray_roi = cv2.cvtColor(clean_roi, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(5, 5))
        ridge = clahe.apply(gray_roi)
        ridge_bgr = cv2.cvtColor(ridge, cv2.COLOR_GRAY2BGR)
        ridge_small = cv2.resize(ridge_bgr, (110, 110))
        frame[70:180, w - 130:w - 20] = ridge_small
        cv2.rectangle(frame, (w - 130, 70), (w - 20, 180), box_color, 2)
        cv2.putText(frame, "PALM TELEMETRY", (w - 140, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[*] Biometric terminal closed gracefully.")
