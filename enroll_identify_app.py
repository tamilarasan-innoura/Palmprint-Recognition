"""
Palmprint Enrollment & Identification

Two features on one shared engine (palm_core.py):

  Enrollment      Register a person with their details, from an uploaded palm
                  image or a webcam snapshot. BOTH palms are
                  required -- a person becomes login-ready only once their left
                  and right palms are on file.
  Identification  Submit either palm, compare it against every login-ready
                  person, and return a ranked result with similarity scores.

Camera orientation (upside-down / mirrored laptop webcams) is a saved setting
applied to every frame before detection, configurable on the Camera tab.
"""
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import palm_core as pc

st.set_page_config(page_title="Palmprint Enrollment & Identification",
                   page_icon="🖐️", layout="wide")

st.markdown("""
<style>
    .verdict-match {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border: 2px solid #10b981; border-radius: 12px;
        padding: 20px; color: #ecfdf5; margin-bottom: 14px;
    }
    .verdict-nomatch {
        background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
        border: 2px solid #ef4444; border-radius: 12px;
        padding: 20px; color: #fef2f2; margin-bottom: 14px;
    }
    .verdict-title { font-size: 26px; font-weight: 700; margin-bottom: 6px; }
    .score-big { font-size: 34px; font-weight: 800; }
</style>
""", unsafe_allow_html=True)


# ===================== ENGINE =====================
@st.cache_resource
def get_engine():
    encoder, device = pc.load_encoder()
    notes = pc.init_db(encoder, device)
    return encoder, device, notes


encoder, device, migration_notes = get_engine()

if "camera" not in st.session_state:
    st.session_state.camera = pc.load_camera_settings()
camera = st.session_state.camera


def make_embedding(bgr, mirrored=None):
    """BGR frame -> (embedding, roi_norm, roi_rgb, hand, source, message)."""
    if mirrored is None:
        mirrored = camera["mirror"]
    roi_norm, roi_rgb, hand, source, msg = pc.extract_roi(bgr, mirrored=mirrored)
    if roi_norm is None:
        return None, None, None, hand, None, msg
    return pc.embed(roi_norm, encoder, device), roi_norm, roi_rgb, hand, source, msg


def open_camera(index):
    cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
    if not cap.isOpened():  # DirectShow is not available on every machine
        cap.release()
        cap = cv2.VideoCapture(int(index))
    return cap


def read_source(key_prefix, help_upload):
    """Renders the Upload / Webcam snapshot selector.

    Returns a BGR frame, or None when nothing has been supplied yet.
    """
    mode = st.radio("Palm input", ["Upload image", "Webcam snapshot"],
                    horizontal=True, key=key_prefix + "_mode")

    if mode == "Upload image":
        up = st.file_uploader(help_upload, type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                              key=key_prefix + "_upload")
        if up is not None:
            return pc.decode_image(up.read())

    else:
        st.caption("Allow camera access, place your open palm in frame, then take the "
                   "photo. The saved rotation is applied to the snapshot.")
        shot = st.camera_input("Take a palm photo", key=key_prefix + "_cam")
        if shot is not None:
            frame = pc.decode_image(shot.getvalue())
            # Rotation is a sensor/driver property so it applies to the browser
            # capture too; mirroring is the browser's own preview convention and
            # is not baked into the returned image.
            return pc.apply_orientation(frame, {"rotation": camera["rotation"],
                                                "mirror": False})

    return None


def hand_badges(has_left, has_right):
    return "%s Left  ·  %s Right" % ("✅" if has_left else "⬜",
                                     "✅" if has_right else "⬜")


# ===================== SIDEBAR =====================
people = pc.load_templates()
persons = pc.list_persons()
total_samples = sum(len(p["samples"]) for p in people.values())
ready = [p for p in persons if p["complete"]]

with st.sidebar:
    st.title("🖐️ Palmprint System")
    st.caption("Self-supervised ResNet-18 · 512-D embeddings")
    st.divider()
    st.metric("Login-ready people", "%d / %d" % (len(ready), len(persons)),
              help="A person is login-ready only once BOTH palms are enrolled.")
    st.metric("Stored palm samples", total_samples)
    stale_total = sum(p["stale_count"] for p in persons)
    if stale_total:
        st.warning("%d sample(s) use the superseded centre-crop extraction and are "
                   "excluded from matching. Re-enroll those palms." % stale_total)
    st.divider()
    threshold = st.slider("Match threshold", 0.60, 0.99, pc.DEFAULT_THRESHOLD, 0.01,
                          help="Cosine similarity a palm must reach. Swept on your "
                               "enrolled palms plus 1500 unenrolled strangers: 0.88 is "
                               "the lowest value where no stranger is ever accepted. "
                               "Below 0.84 strangers start getting in; above 0.88 you "
                               "only lose genuine accepts.")
    min_margin = st.slider("Required margin over runner-up (1:N only)", 0.00, 0.20,
                           pc.DEFAULT_MARGIN, 0.01,
                           help="Applies only to 1:N identification: the best candidate "
                                "must beat the second-best person by this much, or the "
                                "result is refused as ambiguous. Set 0.00 to always "
                                "return the nearest match. Ignored in 1:1 "
                                "authentication, which never compares two people.")
    if threshold < 0.84:
        st.error("Below 0.84, unenrolled strangers start being accepted (measured "
                 "on 1500 non-enrolled palms). The relative check cannot stop a "
                 "stranger — they have no competing template — so the threshold is "
                 "the only defence there.")
    top_k = st.slider("Results to show", 1, 10, 5)
    st.divider()
    st.caption("Camera: index %s · rotation %d° · mirror %s"
               % (camera["index"], camera["rotation"], "on" if camera["mirror"] else "off"))
    st.caption("Database: `data/palm_users.db`")
    if migration_notes:
        with st.expander("⚠️ Legacy data migration"):
            for note in migration_notes:
                st.caption(note)

tab_enroll, tab_identify, tab_people, tab_camera, tab_logs = st.tabs(
    ["➕ Enrollment", "🔍 Identification", "👥 Enrolled People",
     "🎥 Camera", "📋 Identification Log"])


# ===================== ENROLLMENT =====================
with tab_enroll:
    st.subheader("Enroll a person — both palms required")
    st.caption("Register the left palm and the right palm. Only once both are on file "
               "does the person become login-ready, and from then on either palm "
               "identifies them.")

    flash = st.session_state.pop("en_flash", None)
    if flash:
        st.success(flash["message"])
        st.caption("ROI: `%s`" % flash["roi_path"])
        if flash["gallery_name"]:
            st.caption("Dataset gallery entry: `%s`" % flash["gallery_name"])
        elif flash["wanted_gallery"]:
            st.caption("Gallery append skipped (embedding dimension mismatch with "
                       "the existing gallery).")
        if flash["missing"]:
            st.warning("**%s** still needs their **%s** to become login-ready."
                       % (flash["name"], " and ".join(flash["missing"])))
        else:
            st.success("✅ **%s** now has both palms enrolled and can log in with "
                       "either hand." % flash["name"])

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("**Person details**")
        name = st.text_input("Full name *", key="en_name")

        existing = next((p for p in persons
                         if p["name"].lower() == name.strip().lower()), None) \
            if name.strip() else None
        if existing:
            st.info("Existing person — enrollment progress:  %s  (%d sample(s) on file)"
                    % (hand_badges(existing["has_left"], existing["has_right"]),
                       existing["sample_count"]))
            if existing["missing_hands"]:
                st.caption("Still required: **%s**" % ", ".join(existing["missing_hands"]))

        hand_choice = st.radio(
            "Which palm are you enrolling now? *", list(pc.REQUIRED_HANDS),
            horizontal=True, key="en_hand",
            help="Recorded exactly as selected. Auto-detection is only used as a "
                 "cross-check, since it depends on camera orientation.")

        c1, c2 = st.columns(2)
        person_code = c1.text_input("ID / Employee code", key="en_code")
        department = c2.text_input("Department", key="en_dept")
        c3, c4 = st.columns(2)
        email = c3.text_input("Email", key="en_email")
        phone = c4.text_input("Phone", key="en_phone")
        notes = st.text_area("Notes", key="en_notes", height=70)
        add_gallery = st.checkbox(
            "Also append this palm to the dataset gallery", value=True,
            help="Writes the palm to dataset/preprocessed_images and its embedding to "
                 "output/embeddings, so app.py's dataset-wide search finds it too. "
                 "Note the Kaggle images use the centre-crop pipeline, so enrolled "
                 "full palms score high against each other but low against dataset "
                 "images — that is expected, not a fault.")

        st.divider()
        bgr = read_source("en", "Upload a palm image (JPG / PNG / TIFF)")

    with right:
        if bgr is None:
            st.info("Supply a palm image to begin — upload a file or take a "
                    "webcam snapshot.")
        else:
            emb, roi_norm, roi_rgb, detected_hand, source, msg = make_embedding(bgr)
            if emb is None:
                st.error("Palm detection failed: %s" % msg)
                st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Submitted image",
                         use_container_width=True)
            else:
                st.success(msg)
                pc1, pc2 = st.columns(2)
                pc1.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Submitted image",
                          use_container_width=True)
                pc2.image(roi_rgb, caption="Full palm stored in the database (%d×%d)"
                                           % pc.PALM_STORE_SIZE,
                          use_container_width=True)
                st.caption("Recording as: **%s** · auto-detected: **%s** · extraction: "
                           "**%s**" % (hand_choice, detected_hand, source))
                if source == pc.METHOD_DATASET:
                    st.info("No hand outline was found, so this fell back to the "
                            "dataset crop pipeline. That is right for Kaggle TIFFs, "
                            "but a webcam or phone photo should normally yield a full "
                            "palm — check lighting and that the whole hand is in frame.")

                if detected_hand in pc.REQUIRED_HANDS and detected_hand != hand_choice:
                    st.warning("Auto-detection read this as a **%s**, but you selected "
                               "**%s**. Check the palm, or the mirror setting on the "
                               "Camera tab." % (detected_hand, hand_choice))

                if people:
                    ranked_w, _, _, _ = pc.identify(emb, people, threshold, top_k=1,
                                                    require_both_palms=False)
                    if ranked_w and ranked_w[0]["score"] >= threshold \
                            and ranked_w[0]["name"].lower() != name.strip().lower():
                        st.warning("This palm already scores %.2f%% against **%s**. "
                                   "Enroll under that name to add a sample to them."
                                   % (ranked_w[0]["score"] * 100, ranked_w[0]["name"]))

                sample_key = "%s|%s|%s" % (name.strip().lower(), hand_choice,
                                           hash(emb.round(5).tobytes()))
                already = sample_key in st.session_state.get("en_stored", set())
                if already:
                    st.info("This exact palm is already enrolled for that hand in this "
                            "session. Supply a new image to add another sample.")

                if st.button("💾 Enroll this %s" % hand_choice.lower(), type="primary",
                             disabled=already, use_container_width=True, key="en_submit"):
                    if not name.strip():
                        st.error("A full name is required to enroll.")
                    else:
                        details = {
                            "name": name.strip(), "person_code": person_code.strip() or None,
                            "email": email.strip() or None, "phone": phone.strip() or None,
                            "department": department.strip() or None,
                            "notes": notes.strip() or None,
                        }
                        pid = pc.upsert_person(details)
                        roi_path, gallery_name = pc.add_sample(
                            pid, details["name"], emb, roi_norm, roi_rgb,
                            hand_choice, source, append_to_gallery=add_gallery)

                        hands_now = [s["hand"] for s in pc.person_samples(details["name"])]
                        status = pc.enrollment_status(hands_now)

                        st.session_state.setdefault("en_stored", set()).add(sample_key)
                        st.session_state["en_flash"] = {
                            "message": "Saved **%s** for **%s**."
                                       % (hand_choice, details["name"]),
                            "name": details["name"], "roi_path": roi_path,
                            "gallery_name": gallery_name, "wanted_gallery": add_gallery,
                            "missing": status["missing_hands"],
                        }
                        st.rerun()

    st.divider()
    st.markdown("**Enrollment progress**")
    if not persons:
        st.caption("Nobody enrolled yet.")
    else:
        st.dataframe(pd.DataFrame([{
            "Person": p["name"],
            "Left palm": "✅ %d" % p["left_count"] if p["has_left"] else "⬜ missing",
            "Right palm": "✅ %d" % p["right_count"] if p["has_right"] else "⬜ missing",
            "Status": "Login-ready" if p["complete"]
                      else "Needs " + ", ".join(p["missing_hands"]),
        } for p in persons]), use_container_width=True, hide_index=True)

    with st.expander("Tips for a good enrollment"):
        st.markdown("""
- Enroll the **left palm and the right palm** — both are mandatory. Adding 2 or 3
  samples per hand (slightly different positions) improves recall further.
- Keep the palm **open, flat and filling the frame**, with even lighting and no glare.
- Re-enrolling an existing name **adds** a sample rather than replacing the person.
- If auto-detection keeps disagreeing with your hand selection, your camera mirror
  setting is probably wrong — fix it on the **Camera** tab.
        """)


# ===================== IDENTIFICATION =====================
with tab_identify:
    st.subheader("Authenticate / identify a palm")

    # 1:N is the default: for attendance the person just presents a palm and the
    # system works out who they are. 1:1 answers a different question ("is this
    # palm the person it claims to be?") and requires the name up front.
    mode_1to1 = st.radio(
        "Mode", ["Identify (1:N) — show a palm, the system works out who it is"
                 "  ·  use this for attendance",
                 "Authenticate (1:1) — verify a palm against ONE named person"],
        key="id_mode_kind").startswith("Authenticate")

    if mode_1to1:
        st.caption("Pick the person first; the palm is then checked against that "
                   "person's templates. The verdict can only be GRANTED as them, or "
                   "DENIED — no one else's name can be returned.")
    else:
        st.caption("Present either palm. It is scored against every login-ready "
                   "person and the best match is returned — which is how the system "
                   "can tell who is clocking in without anyone typing a name.")

    if not persons:
        st.warning("No one is enrolled yet. Add a person on the **Enrollment** tab first.")
    elif not ready:
        st.warning("Nobody is login-ready yet — every enrolled person is still missing "
                   "a palm. Identification requires both palms on file.")
        for p in persons:
            if p["missing_hands"]:
                st.caption("· **%s** still needs their %s"
                           % (p["name"], " and ".join(p["missing_hands"])))

    claimed = None
    if mode_1to1 and ready:
        # No pre-selected name. A default here silently verifies everyone against
        # whoever happens to be first in the list, which is exactly how Tamil's
        # and Bharani's palms ended up being checked against Narasimma.
        PICK = "— select your name —"
        chosen = st.selectbox("Who are you? (claimed identity)",
                              [PICK] + [p["name"] for p in ready], index=0,
                              key="id_claim")
        claimed = None if chosen == PICK else chosen
        if claimed is None:
            st.info("Choose your name above, or switch to **Identify (1:N)** to let "
                    "the system work out who you are.")

    also_dataset = st.checkbox("Also search the unlabeled dataset gallery", value=False,
                               help="Shows nearest neighbours among the dataset images. "
                                    "Those are unlabeled, so they carry no identity.")

    bgr_q = read_source("id", "Upload the palm image to identify")

    if bgr_q is not None:
        emb_q, roi_norm_q, roi_rgb_q, hand_q, source_q, msg_q = make_embedding(bgr_q)

        if emb_q is None:
            st.error("Palm detection failed: %s" % msg_q)
            st.image(cv2.cvtColor(bgr_q, cv2.COLOR_BGR2RGB), caption="Query image",
                     use_container_width=True)
        else:
            qcol, rcol = st.columns([1, 1.6])
            with qcol:
                st.markdown("**Query palm**")
                st.image(cv2.cvtColor(bgr_q, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.image(roi_rgb_q, caption="Extracted full palm",
                         use_container_width=True)
                st.caption("Auto-detected: **%s** · extraction: **%s**"
                           % (hand_q, source_q))

                # Draw the creases the matcher actually keys on, so a result can
                # be inspected rather than taken on trust.
                overlay, creases, lines = pc.detect_palm_lines(roi_rgb_q)
                if overlay is not None:
                    st.markdown("**Palm lines detected**")
                    lc1, lc2 = st.columns(2)
                    lc1.image(overlay, caption="Lines found on your palm",
                              use_container_width=True)
                    lc2.image(creases, caption="Crease map (what the matcher sees)",
                              use_container_width=True)
                    if lines:
                        st.caption("%d crease line(s); the %d longest are coloured "
                                   "red, green, yellow."
                                   % (len(lines), min(3, len(lines))))
                        st.dataframe(pd.DataFrame([{
                            "Line": "#%d" % (i + 1),
                            "Length (px)": "%.0f" % l["length"],
                            "Thickness (px²)": l["area"],
                        } for i, l in enumerate(lines[:5])]),
                            use_container_width=True, hide_index=True)
                    else:
                        st.warning("No clear palm lines were found. This capture "
                                   "carries little crease detail, so any match is "
                                   "unreliable — improve the lighting and hold the "
                                   "palm flat and steady.")

            with rcol:
                # ---------- 1:1 AUTHENTICATION ----------
                if mode_1to1:
                    if not claimed:
                        st.warning("No name selected, so there is nothing to verify "
                                   "against. Pick your name above, or use "
                                   "**Identify (1:N)** instead.")
                        st.stop()
                    result = pc.verify(emb_q, people, claimed, threshold,
                                       min_margin=min_margin)

                    if result["accepted"]:
                        st.markdown(
                            "<div class='verdict-match'>"
                            "<div class='verdict-title'>✅ ACCESS GRANTED &mdash; %s</div>"
                            "<div class='score-big'>%.2f%%</div>"
                            "<div>matched their <b>%s</b> template · threshold %.0f%%"
                            "</div></div>"
                            % (claimed, result["score"] * 100, result["hand"],
                               threshold * 100), unsafe_allow_html=True)
                        meta = {k: v for k, v in {
                            "ID / Code": people[claimed]["person_code"],
                            "Department": people[claimed]["department"],
                            "Email": people[claimed]["email"],
                            "Phone": people[claimed]["phone"],
                        }.items() if v}
                        if meta:
                            st.table(pd.DataFrame({"Field": list(meta.keys()),
                                                   "Value": list(meta.values())}))
                    else:
                        st.markdown(
                            "<div class='verdict-nomatch'>"
                            "<div class='verdict-title'>❌ ACCESS DENIED</div>"
                            "<div class='score-big'>%s</div><div>claimed %s &mdash; %s"
                            "</div></div>"
                            % ("%.2f%%" % (result["score"] * 100)
                               if result["score"] is not None else "n/a",
                               claimed, result["reason"]), unsafe_allow_html=True)

                    st.markdown("**Scored against %s's templates only**" % claimed)
                    st.dataframe(pd.DataFrame([{
                        "Template": hand, "Score": "%.2f%%" % (sc * 100),
                        "Verdict": "pass" if sc >= threshold else "below threshold",
                    } for hand, sc in result["per_sample"]]),
                        use_container_width=True, hide_index=True)

                    if result["margin"] is not None:
                        st.caption("The claim is %s the best-matching other enrolled "
                                   "template by %.3f."
                                   % ("ahead of" if result["margin"] >= 0 else "BEHIND",
                                      abs(result["margin"])))

                    with st.expander("Diagnostics — reference scores "
                                     "(never returned as an identity)"):
                        st.caption("Other enrolled templates are consulted only to "
                                   "judge whether this claim stands out. The verdict "
                                   "can only ever be GRANTED as %s, or DENIED."
                                   % claimed)
                        others = [r for r in pc.nearest_match(emb_q, people, top_k=top_k)
                                  if r["name"] != claimed]
                        if others:
                            st.dataframe(pd.DataFrame([{
                                "Reference template": r["name"],
                                "Scores this palm at": "%.2f%%" % (r["score"] * 100),
                            } for r in others]), use_container_width=True,
                                hide_index=True)
                        else:
                            st.warning("Only one person is enrolled, so the relative "
                                       "check cannot run and the threshold alone "
                                       "decides — which is the weaker mode. Enroll "
                                       "more people to strengthen it.")

                    pc.log_identification(claimed, result["score"], threshold,
                                          "GRANTED" if result["accepted"] else "DENIED",
                                          source_q)
                    ranked, match, pending, info = [], None, [], {"margin": None,
                                                                  "rejected_reason": None}

                # ---------- 1:N IDENTIFICATION ----------
                else:
                    ranked, match, pending, info = pc.identify(
                        emb_q, people, threshold, top_k=top_k, min_margin=min_margin)

                if match:
                    st.markdown(
                        "<div class='verdict-match'>"
                        "<div class='verdict-title'>✅ IDENTIFIED &mdash; %s</div>"
                        "<div class='score-big'>%.2f%%</div>"
                        "<div>similarity · threshold %.0f%% · matched their "
                        "<b>%s</b> sample (%d on file) · clear of the runner-up "
                        "by %s</div></div>"
                        % (match["name"], match["score"] * 100, threshold * 100,
                           match["hand"], match["sample_count"],
                           "no other candidate" if info["margin"] == float("inf")
                           else "%.3f" % info["margin"]),
                        unsafe_allow_html=True)
                    meta = {k: v for k, v in {
                        "ID / Code": match["person_code"], "Department": match["department"],
                        "Email": match["email"], "Phone": match["phone"],
                        "Enrolled at": match["enrolled_at"],
                    }.items() if v}
                    if meta:
                        st.table(pd.DataFrame(
                            {"Field": list(meta.keys()), "Value": list(meta.values())}))
                    if match["roi_path"] and os.path.exists(match["roi_path"]):
                        st.image(match["roi_path"], width=160,
                                 caption="Enrolled %s for %s"
                                         % (match["hand"], match["name"]))
                elif ranked:
                    if info["rejected_reason"]:
                        detail = ("top two candidates are too close to tell apart "
                                  "&mdash; %s vs %s differ by only %.3f (need %.2f)"
                                  % (ranked[0]["name"], ranked[1]["name"],
                                     info["margin"], min_margin))
                    else:
                        detail = ("best score, below the %.0f%% threshold &mdash; "
                                  "closest login-ready person: %s"
                                  % (threshold * 100, ranked[0]["name"]))
                    st.markdown(
                        "<div class='verdict-nomatch'>"
                        "<div class='verdict-title'>❌ NOT IDENTIFIED</div>"
                        "<div class='score-big'>%.2f%%</div><div>%s</div></div>"
                        % (ranked[0]["score"] * 100, detail),
                        unsafe_allow_html=True)
                    if info["rejected_reason"]:
                        st.error("Refused on purpose: accepting the top score here "
                                 "would be a coin flip between two people. Enroll "
                                 "more samples per hand to separate them.")

                # A strong score against a half-enrolled person is withheld, not
                # silently dropped -- otherwise the result looks inexplicable.
                for p in pending:
                    st.warning("This palm scores **%.2f%%** against **%s**, but they "
                               "are not login-ready: still missing their **%s**. "
                               "Finish their enrollment to allow access."
                               % (p["score"] * 100, p["name"],
                                  " and ".join(p["missing_hands"])))

                if ranked:
                    st.info("**Nearest match in the database: %s — %.2f%%** "
                            "(matched their %s template)"
                            % (ranked[0]["name"], ranked[0]["score"] * 100,
                               ranked[0]["hand"]))

                    st.markdown("**Ranked candidates (login-ready only)**")

                    def verdict(i, r):
                        # Only one candidate can ever be the match. Labelling every
                        # above-threshold row "MATCH" implies several winners.
                        if match and r["name"] == match["name"]:
                            return "MATCH"
                        if r["score"] >= threshold:
                            return "above threshold, not selected"
                        return "below threshold"

                    st.dataframe(pd.DataFrame([{
                        "Rank": i + 1,
                        "Person": r["name"],
                        "Best score": "%.2f%%" % (r["score"] * 100),
                        "Mean score": "%.2f%%" % (r["mean_score"] * 100),
                        "Matched hand": r["hand"],
                        "Samples": r["sample_count"],
                        "Decision": verdict(i, r),
                    } for i, r in enumerate(ranked)]), use_container_width=True,
                        hide_index=True)
                    st.bar_chart(
                        pd.DataFrame({"similarity": [r["score"] for r in ranked]},
                                     index=[r["name"] for r in ranked]))

                    pc.log_identification(
                        match["name"] if match else ranked[0]["name"],
                        ranked[0]["score"], threshold,
                        "MATCH" if match else "NO MATCH", source_q)

            if also_dataset:
                st.divider()
                st.markdown("**Nearest dataset gallery images**")
                hits = pc.search_dataset(emb_q, top_k=top_k)
                if not hits:
                    st.info("Dataset gallery unavailable or dimension mismatch.")
                else:
                    cols = st.columns(min(len(hits), 5))
                    for col, hit in zip(cols, hits):
                        npy = os.path.join(pc.GALLERY_DIR, hit["name"] + ".npy")
                        with col:
                            if os.path.exists(npy):
                                col.image(np.load(npy), use_container_width=True)
                            else:
                                col.info("ROI not on disk")
                            col.caption("`%s`" % hit["name"])
                            col.markdown("**%.2f%%**" % (hit["score"] * 100))


# ===================== ENROLLED PEOPLE =====================
with tab_people:
    st.subheader("Enrolled people")
    if not persons:
        st.info("Nobody enrolled yet.")
    else:
        st.dataframe(pd.DataFrame([{
            "Name": p["name"], "ID / Code": p["person_code"] or "",
            "Department": p["department"] or "", "Email": p["email"] or "",
            "Phone": p["phone"] or "",
            "Left": p["left_count"], "Right": p["right_count"],
            "Stale": p["stale_count"],
            "Status": "✅ Login-ready" if p["complete"]
                      else "⚠️ Needs " + ", ".join(p["missing_hands"]),
            "Enrolled at": p["created_at"],
        } for p in persons]), use_container_width=True, hide_index=True)

        outdated = [p for p in persons
                    if any(s["source"] != pc.METHOD_FULL_PALM
                           for s in pc.person_samples(p["name"]))]
        if outdated:
            st.warning("Some templates were built by an older extraction pipeline. "
                       "Rebuild them from their stored palm images — no need to "
                       "enroll again.")
            if st.button("🔄 Re-encode all templates with the current pipeline",
                         type="primary"):
                updated, skipped = pc.reencode_samples(encoder, device)
                st.success("Rebuilt %d template(s)." % updated)
                for sid, why in skipped:
                    st.caption("Sample #%d skipped: %s" % (sid, why))
                if updated:
                    st.rerun()

        st.divider()
        who = st.selectbox("Inspect / manage", [p["name"] for p in persons])
        samples = pc.person_samples(who)
        status = pc.enrollment_status([s["hand"] for s in samples])
        st.caption("%d stored sample(s) for **%s** — %s"
                   % (len(samples), who, hand_badges(status["has_left"],
                                                     status["has_right"])))
        if samples:
            cols = st.columns(min(len(samples), 6))
            for col, s in zip(cols, samples):
                with col:
                    if s["roi_path"] and os.path.exists(s["roi_path"]):
                        col.image(s["roi_path"], use_container_width=True)
                    stale_tag = " · ⚠️ stale" if pc.is_stale_source(s["source"]) else ""
                    col.caption("%s · %s%s" % (s["hand"], s["source"] or "n/a", stale_tag))

        stale = [s for s in samples if pc.is_stale_source(s["source"])]
        if stale:
            st.error("%d sample(s) were captured with the superseded centre-crop "
                     "extraction, which stored only a small square from the middle "
                     "of the palm. Those templates cannot be compared against "
                     "full-palm captures, so they are excluded from matching — "
                     "please re-enroll these palms." % len(stale))

        # Only full-palm samples can satisfy the dual-palm requirement.
        unassigned = [s for s in samples
                      if s["hand"] not in pc.REQUIRED_HANDS
                      and not pc.is_stale_source(s["source"])]
        if unassigned:
            st.warning("%d sample(s) have no left/right label, so they do not count "
                       "towards the dual-palm requirement. Assign them below instead "
                       "of re-enrolling." % len(unassigned))
            with st.form("assign_hands_%s" % who):
                choices = {}
                for s in unassigned:
                    fc1, fc2 = st.columns([1, 2])
                    with fc1:
                        if s["roi_path"] and os.path.exists(s["roi_path"]):
                            st.image(s["roi_path"], width=90)
                    choices[s["id"]] = fc2.radio(
                        "Sample #%d (currently '%s', enrolled %s)"
                        % (s["id"], s["hand"], s["created_at"]),
                        ["Leave unassigned"] + list(pc.REQUIRED_HANDS),
                        horizontal=True, key="assign_%d" % s["id"])
                if st.form_submit_button("Save hand labels", type="primary"):
                    changed = 0
                    for sid, choice in choices.items():
                        if choice in pc.REQUIRED_HANDS:
                            pc.set_sample_hand(sid, choice)
                            changed += 1
                    st.success("Updated %d sample(s)." % changed) if changed \
                        else st.info("Nothing to update.")
                    if changed:
                        st.rerun()

        confirm = st.checkbox("I want to delete **%s** and all their samples" % who)
        if st.button("🗑️ Delete person", disabled=not confirm):
            pc.delete_person(who)
            st.success("Deleted %s." % who)
            st.rerun()


# ===================== CAMERA =====================
with tab_camera:
    st.subheader("Camera orientation")
    st.caption("Laptop webcams differ: some stream upside-down, some mirrored, some "
               "both. Set the correction once here — it is applied to every frame "
               "before palm detection, so the preview and what gets encoded always "
               "agree.")

    c1, c2, c3 = st.columns(3)
    new_index = c1.number_input("Camera index", 0, 5, int(camera["index"]))
    new_rotation = c2.selectbox(
        "Rotation", [0, 90, 180, 270],
        index=[0, 90, 180, 270].index(int(camera["rotation"])),
        format_func=lambda d: "%d°%s" % (d, "  (upside-down fix)" if d == 180 else ""))
    new_mirror = c3.checkbox(
        "Mirror horizontally", value=bool(camera["mirror"]),
        help="On gives the natural selfie view. This also decides how left/right "
             "hand auto-detection is labelled.")

    pending_settings = {"index": int(new_index), "rotation": int(new_rotation),
                        "mirror": bool(new_mirror)}

    b1, b2 = st.columns(2)
    if b1.button("📷 Capture a test frame", use_container_width=True):
        cap = open_camera(pending_settings["index"])
        if not cap.isOpened():
            cap.release()
            st.error("Could not open camera index %d." % pending_settings["index"])
        else:
            for _ in range(8):  # let auto-exposure settle
                cap.read()
            ok, raw = cap.read()
            cap.release()
            if not ok or raw is None:
                st.error("Camera opened but returned no frame.")
            else:
                st.session_state["cam_test"] = raw

    if b2.button("💾 Save camera settings", type="primary", use_container_width=True):
        pc.save_camera_settings(pending_settings)
        st.session_state.camera = pc.load_camera_settings()
        st.success("Saved. Rotation %d°, mirror %s, index %d."
                   % (pending_settings["rotation"],
                      "on" if pending_settings["mirror"] else "off",
                      pending_settings["index"]))
        st.rerun()

    raw = st.session_state.get("cam_test")
    if raw is not None:
        corrected = pc.apply_orientation(raw, pending_settings)
        v1, v2 = st.columns(2)
        v1.image(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB),
                 caption="Raw frame (straight from the camera)",
                 use_container_width=True)
        v2.image(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB),
                 caption="Corrected — rotation %d°, mirror %s"
                         % (pending_settings["rotation"],
                            "on" if pending_settings["mirror"] else "off"),
                 use_container_width=True)
        brightness = float(np.mean(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)))
        if brightness < 10:
            st.error("This frame is almost entirely black (mean brightness %.1f). "
                     "The privacy shutter is probably closed, or another app is "
                     "holding the camera." % brightness)
        st.info("Adjust **Rotation** and **Mirror** until the right-hand preview looks "
                "correct, then press **Save camera settings**.")
    else:
        st.info("Press **Capture a test frame** to see the raw and corrected views "
                "side by side.")

    with st.expander("Which setting do I need?"):
        st.markdown("""
| What you see | Fix |
|---|---|
| Everything is upside-down | Rotation **180°** |
| Text reads backwards / you raise your right hand and the left side moves | toggle **Mirror** |
| Upside-down *and* backwards | Rotation **180°** + **Mirror** |
| Sideways (portrait sensor) | Rotation **90°** or **270°** |

The mirror setting also flips how left/right hand **auto-detection** is labelled —
but enrollment always stores the hand you explicitly select, so a wrong mirror
setting can never put a palm in the database under the wrong hand.
        """)


# ===================== LOG =====================
with tab_logs:
    st.subheader("Identification log")
    logs = pc.recent_identifications(100)
    if not logs:
        st.info("No identification attempts recorded yet.")
    else:
        st.dataframe(pd.DataFrame([{
            "Time": l["ts"], "Closest person": l["matched"] or "-",
            "Score": "%.2f%%" % (l["score"] * 100) if l["score"] is not None else "-",
            "Threshold": "%.0f%%" % (l["threshold"] * 100),
            "Decision": l["decision"], "Input": l["source"] or "-",
        } for l in logs]), use_container_width=True, hide_index=True)
