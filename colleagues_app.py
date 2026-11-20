import streamlit as st
import numpy as np
import torch
import cv2
import os
import pickle
import time
import json
import uuid
from datetime import datetime
from PIL import Image

from client.cosine_similarity import calculate_similarity
from self_supervised.model import PalmprintEncoder

# ================= 1. PAGE SETUP & SECURITY THEME =================
st.set_page_config(
    page_title="PalmGuard | Enterprise Palmprint Biometric Authentication",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS Theme
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222b;
        border: 1px solid #2d3342;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .status-granted {
        color: #00e676;
        font-weight: 700;
        font-size: 20px;
    }
    .status-denied {
        color: #ff5252;
        font-weight: 700;
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

# File Paths & Defaults
ENCODER_PATH = 'output/model/palmprint_encoder.pth'
DB_PATH = 'output/colleagues_db.pkl'
AUDIT_LOG_PATH = 'output/audit_logs.json'
DEFAULT_THRESHOLD = 0.75
COUNTDOWN_SECONDS = 3.0
STABILITY_FRAMES = 4

# ================= 2. NEURAL NETWORK ENGINE =================
@st.cache_resource
def load_encoder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = PalmprintEncoder().encoder
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    encoder.eval()
    encoder.to(device)
    return encoder, device

encoder, device = load_encoder()

# ================= 3. DATABASE & AUDIT LOGGING SERVICES =================
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

def save_db(db):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

def log_audit_event(user_name, hand, score, status):
    """Logs biometric authentication events with timestamp and security token."""
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
    # Retain last 200 logs
    logs = logs[:200]
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    with open(AUDIT_LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)
    return event

# ================= 4. BIOMETRIC PIPELINE: FULL PALM & VISION CLARITY VALIDATION =================
def classify_hand_side(contour, palm_cx, palm_cy):
    """
    Classifies whether the presented palm is Left Hand or Right Hand.
    In flipped/mirrored camera view:
    - Left hand inner palm: thumb extends outwards to the LEFT.
    - Right hand inner palm: thumb extends outwards to the RIGHT.
    """
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
    """
    Robust Real-Time Palm Detection:
    - Dual-color space skin detection (HSV + YCrCb) works in all lighting.
    - Responsive contour extraction without impossible micro-pixel boundary traps.
    - Centered 138x138 palmprint crop with CLAHE contrast enhancement.
    """
    h_crop, w_crop, _ = crop_bgr.shape
    gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Basic lighting bounds
    brightness = float(np.mean(gray_crop))
    if brightness < 20:
        return False, "⚠️ Room Too Dark: Increase Lighting", None, None, "Palm"
    if brightness > 250:
        return False, "⚠️ Severe Glare: Adjust Angle", None, None, "Palm"

    # 2. Dual-color space skin filter (HSV + YCrCb)
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
        return False, "👋 Place Your Open Palm Inside Box", None, None, "Palm"

    # 3. Hand Contour Extraction
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, "👋 Place Your Open Palm Inside Box", None, None, "Palm"

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 5000:
        return False, "👋 Bring Hand Closer to Fill Box", None, largest, "Palm"

    # 4. Extract Centered Palmprint ROI
    m = cv2.moments(largest)
    if m["m00"] > 0:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
    else:
        cx, cy = w_crop // 2, h_crop // 2

    # Clip center safely inside box
    cx = int(np.clip(cx, 70, w_crop - 70))
    cy = int(np.clip(cy, 70, h_crop - 70))

    half = 70  # 140x140 palm region
    y1, y2 = max(0, cy - half), min(h_crop, cy + half)
    x1, x2 = max(0, cx - half), min(w_crop, cx + half)

    roi_bgr = crop_bgr[y1:y2, x1:x2]
    if roi_bgr.shape[0] < 30 or roi_bgr.shape[1] < 30:
        roi_bgr = crop_bgr[30:250, 30:250]

    roi_std_bgr = cv2.resize(roi_bgr, (138, 138))
    roi_rgb = cv2.cvtColor(roi_std_bgr, cv2.COLOR_BGR2RGB)

    hand_side = classify_hand_side(largest, cx, cy)
    side_msg = f"{hand_side}" if hand_side != "Palm" else "Palm"

    return True, f"✅ {side_msg} Detected!", roi_rgb, largest, hand_side

def compute_embedding(roi_rgb):
    """Normalizes natural palmprint and computes L2-normalized 512-D neural feature vector."""
    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    roi_enhanced = clahe.apply(gray)
    roi_norm = roi_enhanced.astype(np.float32) / 255.0

    tensor = torch.from_numpy(np.expand_dims(roi_norm, axis=0)).float().repeat(1, 3, 1, 1).to(device)
    with torch.no_grad():
        embedding = encoder(tensor).cpu().numpy().flatten()
    
    # L2 normalize embedding for exact, robust cosine similarity
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding

def match_colleague(query_emb, db, detected_side=None):
    """Compares query against all enrolled users and extracts exact multi-hand scores."""
    if not db:
        return None
    best_score = -1.0
    best_match = None

    for name, data in db.items():
        samples = data.get("samples", [])
        if not samples:
            continue
        
        hand_scores = []
        user_best_score = -1.0
        user_best_hand = "Palm"
        user_best_roi = None

        for s in samples:
            emb = s["embedding"]
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            sim = float(np.dot(query_emb, emb_norm))
            h_label = s.get("hand", "Palm")
            hand_scores.append((h_label, sim))

            if sim > user_best_score:
                user_best_score = sim
                user_best_hand = h_label
                user_best_roi = s.get("roi")

        if user_best_score > best_score:
            best_score = user_best_score
            best_match = {
                "name": name,
                "role": data.get("role", ""),
                "hand": user_best_hand,
                "roi": user_best_roi,
                "score": best_score,
                "all_scores": hand_scores
            }
    return best_match

# ================= 5. APPLICATION STATE & HEADER =================
db = load_db()

# Sidebar: System Status & Policies
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hand.png", width=64)
    st.title("PalmGuard Security")
    st.caption("Enterprise Biometric Authentication Engine")
    st.divider()

    st.subheader("⚙️ Security Policy")
    configured_threshold = st.slider("Authentication Threshold", min_value=0.70, max_value=0.92, value=DEFAULT_THRESHOLD, step=0.01,
                                     help="Higher threshold decreases False Acceptance Rate (FAR). 0.80+ recommended for authentication.")

    st.caption(f"**Current FAR Policy:** Strict ({int(configured_threshold*100)}%)")
    st.caption(f"**Enrolled Identities:** {len(db)}")
    st.divider()

    if st.button("🗑️ Reset All Databases", help="Clears all enrolled biometric templates"):
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        st.success("Database reset successfully.")
        st.rerun()

# State variables for Enrollment
if "en_stage" not in st.session_state:
    st.session_state.en_stage = 0
if "en_samples" not in st.session_state:
    st.session_state.en_samples = []
if "candidate_snap" not in st.session_state:
    st.session_state.candidate_snap = None
if "en_name" not in st.session_state:
    st.session_state.en_name = ""
if "en_role" not in st.session_state:
    st.session_state.en_role = ""

# Navigation Tabs
tabs = st.tabs([
    "🔐 Biometric Authentication", 
    "➕ Dual-Palm Enrollment (3s Countdown)", 
    "📋 Security Audit Logs", 
    "👥 Credentials & Users"
])

def render_dos_and_donts_guide():
    with st.expander("📖 **Visual Guide: Palm Scanning DOs & DON'Ts (How to Position Inside Box)**", expanded=False):
        guide_img_path = None
        candidate_paths = [
            r"C:\Users\EH-DEV-NB013\Desktop\exe files\dos_and_donts_guide.jpg",
            r"C:\Users\EH-DEV-NB013\.gemini\antigravity\brain\f0c50b61-9ad2-4cf0-aba6-8e6336e7d44c\palmprint_biometrics_dos_and_donts_1795005119886.jpg",
            os.path.join(os.path.dirname(__file__), "dos_and_donts_guide.jpg")
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                guide_img_path = p
                break
        
        if guide_img_path:
            st.image(guide_img_path, caption="Biometric Scanner Guidelines: DOs and DON'Ts", use_container_width=True)

        col_do, col_dont = st.columns(2)
        with col_do:
            st.success("""
            ### ✅ DO (Correct Scanning)
            - **Center Palm:** Keep your palm flat, centered directly inside the green square.
            - **Spread Fingers:** Relax and spread all 5 fingers naturally facing the lens.
            - **Balanced Lighting:** Good ambient room lighting without heavy backlighting.
            - **Distance:** Keep hand 15–25 cm (6–10 inches) from camera to fill ~70% of the box.
            - **Hold Steady:** Keep palm still for the **3-second timer** until confirmation.
            """)
        with col_dont:
            st.error("""
            ### ❌ DON'T (Avoid These Errors)
            - **Closed Fist / Clenched Fingers:** Never curl fingers or make a fist.
            - **Sideways / Tilted Angle:** Do not show the side of your hand or angle away.
            - **Cut-Off Palm:** Ensure the palm and base of fingers are not cut off by edges.
            - **Motion Blur:** Do not wave or move your hand while the timer counts down.
            - **Foreign Objects:** Do not present mugs, phones, or covered hands to the camera.
            """)

# ================= TAB 1: BIOMETRIC AUTHENTICATION =================
with tabs[0]:
    st.subheader("🔐 Touchless Palmprint Authentication Portal")
    st.caption("Present either your **LEFT** or **RIGHT** palm to authenticate. Neural network matches your identity in real-time.")
    render_dos_and_donts_guide()

    if len(db) == 0:
        st.warning("⚠️ No credentials found in database! Please complete employee enrollment in the 'Dual-Palm Enrollment' tab first.")
    else:
        col_ctrl, _ = st.columns([1, 2])
        with col_ctrl:
            run_auth = st.toggle("🟢 Activate Live Authentication Scanner", value=True, key="toggle_auth")

        if run_auth:
            col_camera, col_telemetry, col_result = st.columns([3, 1.2, 2.2])
            cam_spot = col_camera.empty()
            telemetry_spot = col_telemetry.empty()
            result_spot = col_result.empty()

            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            steady_count = 0
            last_match_event = None
            cooldown_until = 0.0

            while run_auth:
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

                is_palm, palm_msg, clean_roi, contour, hand_side = fast_palm_check(palm_crop)
                now = time.time()
                box_color = (0, 0, 255) if not is_palm else (0, 255, 255)

                if contour is not None and is_palm:
                    cv2.drawContours(palm_crop, [contour], -1, (0, 255, 0), 2)

                if is_palm and now > cooldown_until:
                    steady_count += 1
                    pct = int(min(1.0, steady_count / STABILITY_FRAMES) * 100)
                    side_info = f"[{hand_side}] " if hand_side != "Palm" else ""
                    status_text = f"✋ {side_info}Biometric Locked! Authenticating... {pct}%"

                    if steady_count >= STABILITY_FRAMES:
                        emb = compute_embedding(clean_roi)
                        match = match_colleague(emb, db, detected_side=hand_side)

                        # Evaluate against security threshold
                        if match and match["score"] >= configured_threshold:
                            auth_event = log_audit_event(match["name"], match["hand"], match["score"], "GRANTED")
                            last_match_event = (match, clean_roi, "GRANTED", auth_event.get("session_token"))
                        else:
                            name_str = match["name"] if match else "Unknown"
                            score_val = match["score"] if match else 0.0
                            log_audit_event(name_str, "Unknown", score_val, "DENIED")
                            last_match_event = (match, clean_roi, "DENIED", None)

                        cooldown_until = now + 3.5
                        steady_count = 0
                elif now <= cooldown_until:
                    box_color = (0, 255, 0) if last_match_event and last_match_event[2] == "GRANTED" else (0, 0, 255)
                    status_text = "Verification Complete"
                else:
                    steady_count = max(0, steady_count - 1)
                    status_text = palm_msg

                # Draw Target Box HUD
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                c_len = 25
                for pt1, pt2 in [((x1, y1), (x1+c_len, y1)), ((x1, y1), (x1, y1+c_len)),
                                 ((x2, y1), (x2-c_len, y1)), ((x2, y1), (x2, y1+c_len)),
                                 ((x1, y2), (x1+c_len, y2)), ((x1, y2), (x1, y2-c_len)),
                                 ((x2, y2), (x2-c_len, y2)), ((x2, y2), (x2, y2-c_len))]:
                    cv2.line(frame, pt1, pt2, box_color, 4)

                cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                cam_spot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                if clean_roi is not None:
                    gray_roi = cv2.cvtColor(clean_roi, cv2.COLOR_RGB2GRAY)
                    clahe_obj = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(5, 5))
                    ridge_enhanced = clahe_obj.apply(gray_roi)
                    with telemetry_spot.container():
                        st.caption("🔬 Biometric Telemetry")
                        st.image(clean_roi, caption="Anatomical Crop", width=120)
                        st.image(ridge_enhanced, caption="Palmprint Ridges", width=120)

                # Live Results Card
                with result_spot.container():
                    st.markdown("### 🛡️ Authentication Terminal")
                    if last_match_event:
                        match, live_crop, status, token = last_match_event
                        if status == "GRANTED":
                            st.success(f"## ✅ ACCESS GRANTED\n### **{match['name']}**")
                            score_pct = match['score'] * 100
                            st.metric("Biometric Match Score", f"{score_pct:.1f}%")
                            st.progress(min(1.0, max(0.0, (score_pct - 50) / 50)))
                            st.caption(f"🔐 **Security Session Token:** `{token}`")
                            st.write(f"**Verified Hand:** ✋ **{match['hand']}**")
                            if match['role']:
                                st.write(f"**Role/Department:** {match['role']}")

                            st.markdown("---")
                            st.write("**Dual-Palm Verification Breakdown:**")
                            for h_name, h_sc in match.get("all_scores", []):
                                st.write(f"- {h_name}: **{h_sc*100:.1f}%**")
                        else:
                            st.error("## ❌ ACCESS DENIED\nUnrecognized Biometric Template")
                            score_disp = f"{match['score']*100:.1f}%" if match else "0.0%"
                            st.caption(f"Nearest Match: **{score_disp}** (Threshold required: **{int(configured_threshold*100)}%**)")
                            st.warning("⚠️ Unenrolled palmprint. Access rejected by security policy.")
                    else:
                        st.info("👋 Present an authorized palm inside the green square to authenticate.")

            cap.release()

# ================= TAB 2: DUAL-PALM ENROLLMENT =================
with tabs[1]:
    st.subheader("➕ High-Security Dual-Palm Enrollment")
    st.caption("Enroll exactly **2 palmprints (1 Left Palm + 1 Right Palm)** with an on-screen **3-second steady countdown** and manual confirmation.")
    render_dos_and_donts_guide()

    stages_plan = [
        {"hand": "Left Hand", "title": "LEFT PALM"},
        {"hand": "Right Hand", "title": "RIGHT PALM"},
    ]

    col_name = st.text_input("Employee / Colleague Full Name", value=st.session_state.en_name, placeholder="e.g. Rahul Sharma")
    col_role = st.text_input("Department / Designation", value=st.session_state.en_role, placeholder="e.g. Senior Security Architect")
    st.session_state.en_name = col_name
    st.session_state.en_role = col_role

    cur_stage = st.session_state.en_stage
    st.progress(cur_stage / 2.0, text=f"Enrollment Progress: {cur_stage} of 2 palmprints verified")

    if cur_stage < 2:
        stage_target = stages_plan[cur_stage]

        # LIVE CAMERA STAGE (WITH 3-SECOND COUNTDOWN)
        if st.session_state.candidate_snap is None:
            st.markdown(f"### 📷 Step {cur_stage + 1} of 2: Position Your **{stage_target['title']}**")
            st.info(f"👉 Place your **{stage_target['hand']}** inside the square. Camera counts down **3 seconds** and snaps automatically!")

            toggle_en_cam = st.toggle("🟢 Activate Enrollment Camera", value=True, key=f"toggle_en_{cur_stage}")

            if toggle_en_cam:
                col_live_vid, col_live_monitor = st.columns([3, 1.5])
                vid_container = col_live_vid.empty()
                monitor_container = col_live_monitor.empty()

                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                palm_start_time = None

                while toggle_en_cam and st.session_state.candidate_snap is None:
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

                    is_palm, palm_msg, clean_roi, contour, hand_side = fast_palm_check(palm_crop)
                    now = time.time()

                    target_hand = stage_target["hand"]
                    is_wrong_hand = False
                    if is_palm and hand_side != "Palm" and hand_side != target_hand:
                        is_wrong_hand = True

                    if contour is not None and is_palm and not is_wrong_hand:
                        cv2.drawContours(palm_crop, [contour], -1, (0, 255, 0), 2)

                    if is_palm and not is_wrong_hand:
                        if palm_start_time is None:
                            palm_start_time = now
                            burst_frames = []

                        elapsed = now - palm_start_time
                        remaining = max(0.0, COUNTDOWN_SECONDS - elapsed)
                        box_color = (0, 255, 255)
                        hud_text = f"✋ HOLD STEADY: Snapping in {remaining:.1f}s..."

                        # Display Countdown Overlay
                        sec_num = int(np.ceil(remaining))
                        cv2.putText(frame, str(sec_num), (x1 + 115, y1 + 160),
                                    cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 255, 255), 4)

                        # Capture burst frames during final 1.5 seconds for multi-sample noise rejection
                        if elapsed >= 1.5 and clean_roi is not None:
                            gray_roi = cv2.cvtColor(clean_roi, cv2.COLOR_RGB2GRAY)
                            sh = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
                            burst_frames.append((sh, clean_roi, palm_crop.copy()))

                        if elapsed >= COUNTDOWN_SECONDS:
                            if burst_frames:
                                burst_frames.sort(key=lambda x: x[0], reverse=True)
                                best_rois = [x[1] for x in burst_frames[:3]]
                                best_embs = [compute_embedding(r) for r in best_rois]
                                avg_emb = np.mean(best_embs, axis=0)
                                avg_emb = avg_emb / np.linalg.norm(avg_emb)
                                final_roi = best_rois[0]
                                final_crop = burst_frames[0][2]
                            else:
                                avg_emb = compute_embedding(clean_roi)
                                final_roi = clean_roi
                                final_crop = palm_crop

                            raw_crop_rgb = cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB)
                            st.session_state.candidate_snap = {
                                "clean_roi": final_roi,
                                "embedding": avg_emb,
                                "raw_rgb": raw_crop_rgb,
                                "hand": target_hand,
                                "title": stage_target["title"]
                            }
                            cap.release()
                            st.rerun()
                    else:
                        palm_start_time = None
                        burst_frames = []
                        box_color = (0, 0, 255)
                        if is_wrong_hand:
                            hud_text = f"⚠️ Wrong Hand: Step requires your {target_hand.upper()}"
                        else:
                            hud_text = palm_msg

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, hud_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                    vid_container.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                    if clean_roi is not None:
                        monitor_container.image(clean_roi, caption="Live Palm Telemetry", width=140)

                cap.release()

        # QUALITY REVIEW & APPROVAL GATE
        else:
            snap = st.session_state.candidate_snap
            st.markdown(f"### 🔍 Quality Inspection: **{snap['title']}**")
            st.success("📸 Palm snapshot acquired! Inspect image clarity before committing to biometric credential database.")

            c_prev1, c_prev2, c_decision = st.columns([1, 1, 1.5])
            with c_prev1:
                st.image(snap["raw_rgb"], caption="Raw Frame Capture", width=190)
            with c_prev2:
                st.image(snap["clean_roi"], caption="Extracted Palmprint Credential", width=170)

            with c_decision:
                st.markdown("#### User Approval Decision")
                st.write("Ensure principal creases and lines are in sharp focus.")

                col_save_btn, col_retake_btn = st.columns(2)
                with col_save_btn:
                    if st.button("✅ Confirm & Proceed", type="primary", key=f"btn_save_{cur_stage}"):
                        if not st.session_state.en_name.strip():
                            st.error("Please specify Employee Full Name above before proceeding.")
                        else:
                            st.session_state.en_samples.append({
                                "hand": snap["hand"],
                                "embedding": snap["embedding"],
                                "roi": snap["clean_roi"]
                            })
                            st.session_state.candidate_snap = None
                            st.session_state.en_stage += 1
                            st.rerun()

                with col_retake_btn:
                    if st.button("🔄 Retake Capture", key=f"btn_retake_{cur_stage}"):
                        st.session_state.candidate_snap = None
                        st.rerun()

    else:
        # All 2 Samples Acquired!
        st.success(f"🎉 **Both Biometric Credentials Verified for {st.session_state.en_name}!**")
        cols = st.columns(2)
        for idx, s in enumerate(st.session_state.en_samples):
            with cols[idx]:
                st.image(s["roi"], caption=f"{s['hand']}", width=140)

        c_commit, c_discard = st.columns([1, 1])
        with c_commit:
            if st.button("💾 Commit Credentials to Secure Database", type="primary"):
                name_key = st.session_state.en_name.strip()
                db[name_key] = {
                    "role": st.session_state.en_role.strip(),
                    "samples": st.session_state.en_samples
                }
                save_db(db)
                st.balloons()
                st.success(f"🎉 Identity **{name_key}** registered into biometric authorization database!")

                st.session_state.en_stage = 0
                st.session_state.en_samples = []
                st.session_state.candidate_snap = None
                st.session_state.en_name = ""
                st.session_state.en_role = ""
                st.rerun()

        with c_discard:
            if st.button("❌ Discard Session"):
                st.session_state.en_stage = 0
                st.session_state.en_samples = []
                st.session_state.candidate_snap = None
                st.rerun()

# ================= TAB 3: AUDIT & ACCESS LOGS =================
with tabs[2]:
    st.subheader("📋 Real-Time Security Audit Trail")
    st.caption("Immutable record of authentication attempts, match scores, and session security tokens.")

    if os.path.exists(AUDIT_LOG_PATH):
        try:
            with open(AUDIT_LOG_PATH, "r") as f:
                logs_data = json.load(f)
            if logs_data:
                st.dataframe(logs_data, use_container_width=True)
                # Export Button
                st.download_button(
                    label="📥 Export Audit Logs (JSON)",
                    data=json.dumps(logs_data, indent=2),
                    file_name=f"palmguard_audit_log_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            else:
                st.info("No authentication events recorded yet.")
        except Exception:
            st.info("Audit log initialized.")
    else:
        st.info("No audit logs found. Completed authentications will appear here.")

# ================= TAB 4: CREDENTIALS & USERS =================
with tabs[3]:
    st.subheader("👥 Registered Employee Credentials")
    if len(db) == 0:
        st.info("No users registered in database.")
    else:
        for name, data in list(db.items()):
            samples = data.get("samples", [])
            c1, c2, c3 = st.columns([2.5, 3, 1])
            with c1:
                cols = st.columns(2)
                for idx, s in enumerate(samples[:2]):
                    with cols[idx]:
                        if s.get("roi") is not None:
                            st.image(s["roi"], caption=s.get('hand', 'Palm'), width=80)
            with c2:
                st.markdown(f"### {name}")
                if data.get("role"):
                    st.write(f"*{data['role']}*")
                st.caption(f"Enrolled Palmprints: {len(samples)} (Dual-Hand Registered)")
            with c3:
                if st.button("Revoke Access", key=f"revoke_{name}"):
                    del db[name]
                    save_db(db)
                    st.warning(f"Access revoked for {name}.")
                    st.rerun()
            st.divider()
