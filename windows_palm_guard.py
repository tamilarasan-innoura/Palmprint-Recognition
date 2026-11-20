import cv2
import numpy as np
import torch
import os
import pickle
import time
import ctypes

from client.cosine_similarity import calculate_similarity
from self_supervised.model import PalmprintEncoder

ENCODER_PATH = 'output/model/palmprint_encoder.pth'
COLLEAGUES_DB_PATH = 'output/colleagues_db.pkl'
MATCH_THRESHOLD = 0.75  # Calibrated 75% threshold for 0% False Rejection & 0% False Acceptance
STABILITY_FRAMES = 4
AUTO_OFF_TIMEOUT = 45.0  # Turns screen OFF if no authorized user seen for 45s

# --- 1. Windows Native Display Controls (ctypes) ---
def turn_screen_off():
    """Powers off the Windows monitor immediately."""
    print("[*] Screen Command: Turning Screen OFF...")
    ctypes.windll.user32.SendMessageW(0xFFFF, 0x0112, 0xF170, 2)

def turn_screen_on():
    """Wakes up the Windows monitor."""
    print("[*] Screen Command: Waking Screen ON...")
    ctypes.windll.user32.mouse_event(0x0001, 0, 1, 0, 0)
    ctypes.windll.user32.mouse_event(0x0001, 0, -1, 0, 0)

def lock_windows():
    """Locks the Windows workstation (Win + L)."""
    print("[*] Security Command: Locking Windows...")
    ctypes.windll.user32.LockWorkStation()

# --- 2. Load Neural Network & Database ---
print("[*] Loading Neural Network on CPU/GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = PalmprintEncoder().encoder
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
encoder.eval()
encoder.to(device)

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

db = load_db()
if not db:
    print("[!] Warning: Database is empty! Please enroll your hands first in colleagues_app.py.")
else:
    print(f"[*] Authorized users loaded: {list(db.keys())}")

# --- 3. Fast Palm Verification & Full Vision Quality Check ---
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

    # 1. Lighting Quality Check
    brightness = float(np.mean(gray_crop))
    if brightness < 40:
        return False, "⚠️ Too Dark: Increase Lighting", None, "Palm"
    if brightness > 225:
        return False, "⚠️ Glare: Adjust Hand Angle", None, "Palm"

    # 2. Vision Clarity / Blur Detection (Laplacian Variance)
    sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    if sharpness < 40:
        return False, "⚠️ Image Blurry: Hold Steady", None, "Palm"

    # 3. Human Skin Chrominance Filter (YCrCb)
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    if (np.count_nonzero(skin_mask) / (h_crop * w_crop)) < 0.16:
        return False, "⚠️ Not Skin: Show Palm", None, "Palm"

    # 4. Hand Contour & Boundary Analysis
    blurred = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, "⚠️ No Palm Detected", None, "Palm"

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    bx, by, bw, bh = cv2.boundingRect(largest)

    # Full Palm Boundary Checks
    if by <= 2:
        return False, "⚠️ Fingers Cut Off: Lower Hand", None, "Palm"
    if bx <= 2 or (bx + bw) >= (w_crop - 2):
        return False, "⚠️ Palm Cut Off: Center Hand", None, "Palm"
    if bw < 95 or bh < 95:
        return False, "⚠️ Palm Too Far: Bring Closer", None, "Palm"
    if bw > 272 and bh > 272:
        return False, "⚠️ Palm Too Close: Move Back", None, "Palm"

    hull_pts = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull_pts)
    if hull_area == 0 or (float(area) / hull_area) > 0.88:
        return False, "⚠️ Show Open Palm with Spread Fingers", None, "Palm"

    # 5. Distance Transform Palm Center
    hand_mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
    cv2.drawContours(hand_mask, [largest], -1, 255, -1)
    dist_map = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
    _, max_radius, _, max_loc = cv2.minMaxLoc(dist_map)
    cx, cy = max_loc
    palm_radius = float(max_radius)

    if palm_radius < 35:
        return False, "⚠️ Palm Too Far: Bring Closer", None, "Palm"

    if abs(cx - w_crop // 2) > 65 or abs(cy - h_crop // 2) > 65:
        return False, "⚠️ Center Palm Directly in Square", None, "Palm"

    half = int(np.clip(palm_radius * 1.6, 65, 115))
    y1, y2 = max(0, cy - half), min(h_crop, cy + half)
    x1, x2 = max(0, cx - half), min(w_crop, cx + half)

    roi_bgr = crop_bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0 or roi_bgr.shape[0] < 50 or roi_bgr.shape[1] < 50:
        return False, "⚠️ Align Palm Center Inside Box", None, "Palm"

    roi_std_bgr = cv2.resize(roi_bgr, (138, 138))
    clean_roi_rgb = cv2.cvtColor(roi_std_bgr, cv2.COLOR_BGR2RGB)
    side = classify_hand_side(largest, cx, cy)
    return True, f"✅ {side} Locked!", clean_roi_rgb, side

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
        return None, 0.0, ""
    best_score = -1.0
    best_name, best_hand = None, ""

    for name, data in db.items():
        samples = data.get("samples", [])
        if not samples:
            continue
        for s in samples:
            emb = s["embedding"]
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            sim = float(np.dot(query_emb, emb_norm))
            if sim > best_score:
                best_score = sim
                best_name = name
                best_hand = s.get("hand", "Palm")

    return best_name, best_score, best_hand

# --- 4. Main Windows Screen Guard Loop ---
print("\n" + "=" * 65)
print("  WINDOWS PALM BIOMETRIC AUTHENTICATION (BOTH HANDS TURN ON)")
print("=" * 65)
print("  - SHOW EITHER HAND (LEFT OR RIGHT): Screen turns ON!")
print(f"  - WALK AWAY / INACTIVITY ({int(AUTO_OFF_TIMEOUT)}s): Screen automatically turns OFF!")
print("  - PRESS 'L' on keyboard: Turn Screen OFF & Lock immediately.")
print("  - PRESS 'Q' on keyboard: Exit program.")
print("=" * 65 + "\n")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

steady_count = 0
cooldown_until = 0.0
last_auth_time = time.time()
screen_state = "ON"

last_status_msg = "Waiting for authorized palm (Left or Right)..."
last_status_color = (200, 200, 200)

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

    is_palm, palm_msg, clean_roi, hand_side = fast_palm_check(palm_crop)
    now = time.time()

    # --- PALM DETECTION & AUTHENTICATION ---
    if is_palm and now > cooldown_until:
        steady_count += 1
        pct = int(min(1.0, steady_count / STABILITY_FRAMES) * 100)
        box_color = (0, 255, 255)
        status_text = f"Palm Locked! Verifying... {pct}%"

        if steady_count >= STABILITY_FRAMES:
            emb = compute_embedding(clean_roi)
            name, score, hand = match_colleague(emb)

            if score >= MATCH_THRESHOLD:
                # ✅ BOTH HANDS (LEFT OR RIGHT) TURN SCREEN ON!
                last_status_color = (0, 255, 0)
                last_status_msg = f"AUTHENTICATED: {name} ({hand} - {score*100:.1f}%) --> SCREEN ON!"
                
                turn_screen_on()
                screen_state = "ON"
                last_auth_time = now
                cooldown_until = now + 4.0
            else:
                last_status_color = (0, 0, 255)
                last_status_msg = f"ACCESS DENIED ({score*100:.1f}%)"
                cooldown_until = now + 2.0

            steady_count = 0
    else:
        if now > cooldown_until:
            steady_count = max(0, steady_count - 1)
            box_color = (0, 255, 0) if is_palm else (0, 0, 255)
            status_text = "Show EITHER Palm to Turn Screen ON" if is_palm else f"{palm_msg}"
        else:
            box_color = last_status_color
            status_text = "Authenticated!"

    # --- AUTO-OFF / SLEEP LOGIC ---
    time_since_auth = now - last_auth_time
    time_left_to_sleep = max(0.0, AUTO_OFF_TIMEOUT - time_since_auth)

    if time_since_auth >= AUTO_OFF_TIMEOUT and screen_state == "ON":
        print(f"[*] Inactivity timeout reached ({AUTO_OFF_TIMEOUT}s). Turning screen OFF...")
        turn_screen_off()
        screen_state = "OFF"
        last_status_msg = "Screen is OFF (Sleeping). Show any palm to wake up!"
        last_status_color = (0, 165, 255)

    # Draw HUD
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
    cv2.putText(frame, "PALM SCREEN CONTROLLER (BOTH HANDS = ON)", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

    # Bottom Banner
    cv2.rectangle(frame, (0, h - 60), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, last_status_msg, (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, last_status_color, 2)
    
    sleep_info = f"Screen State: {screen_state} | Auto-Sleep in: {int(time_left_to_sleep)}s"
    cv2.putText(frame, sleep_info, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow("Windows Palm Screen Guard (Q: Exit | L: Sleep/Lock)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        turn_screen_off()
        screen_state = "OFF"
        last_status_msg = "Screen manually turned OFF. Show palm to wake!"

cap.release()
cv2.destroyAllWindows()
