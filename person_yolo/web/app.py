import io
import os
import sys
import tempfile
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# Support running via `streamlit run person_yolo/web/app.py` (no package context)
try:
    from .utils import (
        analyze_image,
        load_model,
        track_video_collect,
        render_selected_track_video,
    )
except Exception:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from person_yolo.web.utils import (
        analyze_image,
        load_model,
        track_video_collect,
        render_selected_track_video,
    )

@st.cache_resource(show_spinner=False)
def _load_model(weights: Optional[str] = None) -> YOLO:
    return load_model(weights)


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def heatmap_to_rgb(heat: np.ndarray) -> np.ndarray:
    import cv2
    heat_uint8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def inject_css():
    st.markdown(
        """
        <style>
        .stApp { background: #0b1220; color: #e5e7eb; }
        [data-testid="stSidebar"] { background: #0d1325; }
        /* Sidebar nav buttons */
        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            background: #1f2937;
            color: #e5e7eb;
            border: 1px solid #374151;
            border-radius: 10px;
            padding: 8px 12px;
            font-weight: 600;
            height: 44px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        [data-testid="stSidebar"] .stButton > button:hover { background: #273244; border-color: #4b5563; }
        .nav-spacer { height: 8px; }
        .hero {
            margin: 1rem 0 2rem 0; padding: 1.5rem 2rem; border-radius: 20px;
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
            color: #ffffff; font-weight: 800; font-size: 2rem; text-align:center;
            box-shadow: 0 12px 32px rgba(0,0,0,0.30);
        }
        .divider { border-bottom: 1px solid #1f2937; margin: 1rem 0 1.5rem 0; }
        .card-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.5rem; align-items: stretch; }
        .card {
            border-radius: 16px; padding: 1.1rem 1.4rem; color: #0b1220;
            box-shadow: 0 10px 28px rgba(0,0,0,0.28);
        }
        .card.image { background: linear-gradient(135deg,#667eea 0%, #764ba2 100%); }
        .card.video { background: linear-gradient(135deg,#FF6B6B 0%, #FFE66D 100%); }
        .card h3 { margin: 0 0 .5rem 0; font-weight: 800; }
        .card .subtitle { margin:.25rem 0 1rem 0; color:#0b1220; opacity:.8; }
        .card ul { margin: .5rem 0 0 1rem; }
        .card ul li { margin:.35rem 0; }
        .badge { display:inline-flex; align-items:center; gap:.5rem; font-weight:800; color:#0b1220; }
        .badge .icon { font-size:1.35rem; }
        .section { background:#0f172a; border-radius:12px; padding:1rem; margin-top:1rem; }
        .metric-row { display:grid; grid-template-columns: 2fr 1fr; gap:1rem; }
        .thumb-grid { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:.75rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def feature_cards():
        st.markdown('<div class="card-grid">', unsafe_allow_html=True)
        st.markdown(
                """
                <div class="card image">
                    <div class="badge"><span class="icon">🖼️</span><span>IMAGE ANALYSIS</span></div>
                    <div class="subtitle">Upload a single image and get instant analysis:</div>
                    <ul>
                        <li>• 👤 Person Detection</li>
                        <li>• 📊 Crowd Density Estimation</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
        )
        st.markdown(
                """
                <div class="card video">
                    <div class="badge"><span class="icon">🎬</span><span>VIDEO ANALYSIS</span></div>
                    <div class="subtitle">Upload a video for comprehensive analysis:</div>
                    <ul>
                        <li>• ⏱️ Real-time Detection</li>
                        <li>• 📈 Crowd Density Over Time</li>
                        <li>• 🆔 Person Tracking (select-one highlight)</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="SDS Dashboard", layout="wide")
    inject_css()

    def set_page(p):
        st.session_state["page"] = p

    def nav_button(label: str, icon: str = ""):
        if st.button(f"{icon} {label}", key=f"nav_{label}"):
            set_page(label)
        st.markdown('<div class="nav-spacer"></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## SDS Dashboard")
        if "page" not in st.session_state:
            st.session_state["page"] = "Home"
        nav_button("Home", "🏠")
        nav_button("Image Analysis", "🖼️")
        nav_button("Video Analysis", "🎬")
        st.divider()
        st.markdown("### Settings")
        conf = st.slider("Confidence", 0.05, 0.9, 0.25, 0.01)
        repo_root = Path(__file__).resolve().parents[2]
        default_weights = str(repo_root / "yolo26n.pt")
        weights = st.text_input("Weights (optional)", value=default_weights)
        model = _load_model(weights if weights.strip() else None)
        st.success("Model ready")

    st.markdown('<div class="hero">👥 SDS – Smart Detection & Surveillance</div>', unsafe_allow_html=True)
    st.markdown("Welcome to the SDS Crowd Analysis Dashboard.")
    st.caption("This system provides real-time analysis of crowds and individuals in images and videos.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Current page from session
    page_norm = st.session_state.get("page", "Home")
    if page_norm == "Home":
        feature_cards()
        # Key features and getting started
        st.markdown("#### Key Features")
        st.markdown("- Fast person-only detection and density visualization")
        st.markdown("- Video tracking with selectable person highlight")
        st.markdown("- Simple, GPU-accelerated pipeline with YOLOv8 + ByteTrack")
        st.markdown("#### Getting Started")
        st.markdown("- Upload an image or video from the sidebar tabs")
        st.markdown("- Adjust confidence threshold in Settings")
        st.markdown("- Optionally provide a custom weights path")

    elif page_norm == "Image Analysis":
        st.markdown("### Image Analysis")
        with st.container():
            st.markdown("#### Detection Controls")
            show_all = st.checkbox("Show all detections", value=True, help="Bypass strict filters to reveal small/distant people.")
            colc1, colc2, colc3 = st.columns([1, 1, 1])
            with colc1:
                min_h = 0
            with colc2:
                min_ar = st.slider("Min aspect ratio (h/w)", 0.0, 3.0, 0.0 if show_all else 0.8, 0.05)
            with colc3:
                max_det = st.slider("Max detections", 50, 1000, 500, 50)

            up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
            if up is not None:
                data = up.read()
                img_array = np.frombuffer(data, dtype=np.uint8)
                import cv2
                bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                conf_eff = 0.12 if show_all else conf
                res = analyze_image(
                    model, bgr, conf=conf_eff, min_height=int(min_h), min_aspect_ratio=float(min_ar), max_det=int(max_det)
                )

                st.markdown('<div class="section">', unsafe_allow_html=True)
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(to_rgb(res["vis_bgr"]), caption=f"People: {res['count']} | Density/MPx: {res['density_per_mpx']:.2f}")
                with col2:
                    st.metric("People Count", res["count"]) 
                    st.metric("Density per MPx", f"{res['density_per_mpx']:.2f}")
                    st.image(heatmap_to_rgb(res["heatmap"]), caption="Spatial Density Heatmap")
                st.markdown('</div>', unsafe_allow_html=True)

    elif page_norm == "Video Analysis":
        st.markdown("### Video Analysis")
        # End session button to clear current video and selections
        end_col1, end_col2 = st.columns([1, 3])
        with end_col1:
            if st.button("End Session"):
                for k in [
                    "uploaded_signature",
                    "video_tmp_path",
                    "selected_track_id",
                    "final_video_path",
                ]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.success("Session cleared. You can upload the next video.")
                st.stop()
        st.markdown("#### Detection Controls")
        show_all_v = st.checkbox("Show all detections (video)", value=True, help="Reveal small/distant people by lowering confidence.")
        vcol1, vcol2, vcol3, vcol4 = st.columns([1, 1, 1, 1])
        with vcol1:
            min_ar_v = st.slider("Min aspect ratio (h/w)", 0.0, 3.0, 0.0 if show_all_v else 0.8, 0.05)
        with vcol2:
            max_det_v = st.slider("Max detections", 50, 1000, 700, 50)
        with vcol3:
            imgsz_v = st.slider("Inference size (imgsz)", 512, 1280, 1152, 32)
        with vcol4:
            conf_override = st.slider("Video confidence", 0.01, 0.90, 0.08 if show_all_v else conf, 0.01)
        boost_recall = st.checkbox("Boost recall (tuned ByteTrack)", value=True)
        strict_tracking = st.checkbox("Strict tracking (only selected person)", value=True, help="If disabled, shows all detections in frames where the selected ID is temporarily missing.")

        # Quick reset to prepare for a new video upload
        reset_row = st.columns([1,3])
        with reset_row[0]:
            if st.button("New Video"):
                for k in [
                    "uploaded_signature",
                    "video_tmp_path",
                    "selected_track_id",
                    "final_video_path",
                ]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.experimental_rerun()

        vup = st.file_uploader("Upload video (mp4, avi, mov)", type=["mp4", "avi", "mov"])
        tmp_path = None
        if vup is not None:
            # Only process and reset state when a NEW file is uploaded
            upload_sig = f"{getattr(vup, 'name', 'unknown')}:{getattr(vup, 'size', 0)}"
            prev_sig = st.session_state.get("uploaded_signature")
            if upload_sig != prev_sig:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(vup.read())
                    tmp_path = tmp.name
                st.session_state["uploaded_signature"] = upload_sig
                # Persist uploaded video path and reset selection/render state
                st.session_state["video_tmp_path"] = tmp_path
                st.session_state["selected_track_id"] = None
                st.session_state["final_video_path"] = None

        # Use persisted video path if available
        session_path = st.session_state.get("video_tmp_path", None)
        if session_path:
            # Re-analyze and Reset selection actions
            a1, a2 = st.columns(2)
            with a1:
                if st.button("Re-analyze"):
                    st.session_state["selected_track_id"] = None
                    st.session_state["final_video_path"] = None
            with a2:
                if st.button("Reset selection"):
                    st.session_state["selected_track_id"] = None
                    st.session_state["final_video_path"] = None

            st.info("Analyzing video (first pass)...")
            conf_eff_v = float(conf_override)
            tracker_cfg_path = os.path.join(os.path.dirname(__file__), "bytetrack_person.yaml") if boost_recall else None
            summary = track_video_collect(
                model,
                session_path,
                conf=conf_eff_v,
                min_height=0,
                min_aspect_ratio=float(min_ar_v),
                max_det=int(max_det_v),
                imgsz=int(imgsz_v),
                max_frames=200,
                tracker_cfg=tracker_cfg_path,
            )

            df = pd.DataFrame({
                "frame": list(range(1, len(summary["counts"]) + 1)),
                "count": summary["counts"],
                "density": summary["densities"],
            })
            c1, c2 = st.columns(2)
            with c1:
                st.line_chart(df.set_index("frame")["count"], height=220)
            with c2:
                st.line_chart(df.set_index("frame")["density"], height=220)

            st.markdown("#### First frame visualization")
            if summary["first_vis_bgr"] is not None:
                st.image(to_rgb(summary["first_vis_bgr"]))

            st.markdown("#### Select a person to highlight across the video")
            selected = st.session_state.get("selected_track_id", None)
            thumbs = summary["thumbs"]
            keys = list(thumbs.keys())
            if selected is None and len(keys) > 0:
                st.markdown('<div class="thumb-grid">', unsafe_allow_html=True)
                for tid in keys:
                    st.image(to_rgb(thumbs[tid]), caption=f"ID {tid}")
                    if st.button(f"Select ID {tid}", key=f"sel_{tid}"):
                        st.session_state["selected_track_id"] = int(tid)
                        # Force immediate UI update to reflect selection
                        try:
                            st.rerun()
                        except Exception:
                            # Fallback for older Streamlit versions
                            os.environ["_ST_TRIGGER_RERUN"] = "1"
                st.markdown('</div>', unsafe_allow_html=True)
            elif selected is None and len(keys) == 0:
                st.warning("No track IDs visible yet. Try lowering Video confidence, increasing imgsz, or proceed to render all detections.")
                if st.button("Render video (all detections)"):
                    # Persist an explicit None selection to render all
                    st.session_state["selected_track_id"] = None
                    selected = None

            # Render section (available whether an ID is selected or not)
            if selected is not None:
                st.success(f"Selected track ID: {selected}")
            out_dir = os.path.join(os.path.dirname(session_path), "outputs")
            os.makedirs(out_dir, exist_ok=True)
            out_name = f"selected_id_{selected}.mp4" if selected is not None else "all_detections.mp4"
            out_path = os.path.join(out_dir, out_name)
            render_label = (
                "Render video with only selected person highlighted"
                if selected is not None
                else "Render video (all detections)"
            )
            if st.button(render_label):
                with st.spinner("Rendering..."):
                    # Provide first-pass reference box for IoU-based ID remap (stabilizes second pass)
                    ref_xyxy = None
                    try:
                        first_boxes = summary.get("first_boxes", {}) if isinstance(summary, dict) else {}
                        if selected is not None and isinstance(first_boxes, dict):
                            ref_xyxy = first_boxes.get(int(selected))
                    except Exception:
                        ref_xyxy = None
                    final_path = render_selected_track_video(
                        model,
                        session_path,
                        int(selected) if selected is not None else None,
                        out_path,
                        conf=conf_eff_v,
                        min_height=0,
                        min_aspect_ratio=float(min_ar_v),
                        max_det=int(max_det_v),
                        imgsz=int(imgsz_v),
                        tracker_cfg=tracker_cfg_path,
                        strict=bool(strict_tracking),
                        selected_ref_xyxy=ref_xyxy,
                    )
                # Persist final output path for display/download across reruns
                st.session_state["final_video_path"] = final_path

                # If we have a rendered video, always show it with a download option
                final_path = st.session_state.get("final_video_path")
                if final_path and os.path.isfile(final_path):
                    st.video(final_path)
                    with open(final_path, "rb") as f:
                        st.download_button(
                            "Download result",
                            data=f.read(),
                            file_name=os.path.basename(final_path),
                        )


if __name__ == "__main__":
    main()
