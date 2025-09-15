import os
import time
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import streamlit as st
from ultralytics import YOLO

# Device selection (GPU/CPU)
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# App configuration
st.set_page_config(page_title="Fabric Defect Detector", layout="wide")
st.title("Fabric Defect Detector - YOLOv8s")

@st.cache_resource(show_spinner=True)
def load_model(weights_path="best.pt"):
    """
    Load YOLO model with error handling.
    """
    try:
        if not os.path.exists(weights_path):
            st.error(f"Model weights file '{weights_path}' not found.")
            st.info("Please ensure 'best.pt' is in the same directory as this app.")
            st.stop()

        model = YOLO(weights_path)
        if DEVICE == "cuda":
            model.to("cuda")
        st.success(f"Model loaded successfully on {DEVICE.upper()}")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

WEIGHTS = "best.pt"
model = load_model(WEIGHTS)

st.sidebar.header("Inference Settings")
imgsz = st.sidebar.selectbox("Image size", [640, 960, 1280], index=1)
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.80, 0.15, 0.01)
iou = st.sidebar.slider("IoU threshold", 0.30, 0.90, 0.60, 0.01)
agnostic_nms = st.sidebar.checkbox("Agnostic NMS", value=False)
max_det = st.sidebar.slider("Max detections per image", 50, 300, 300, 10)
st.sidebar.caption(f"Device: {DEVICE.upper()}")
if DEVICE == "cuda":
    try:
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.caption(f"GPU: {gpu_name}")
    except Exception:
        pass

if conf >= iou:
    st.sidebar.error("Confidence threshold must be less than IoU threshold.")

if "history" not in st.session_state:
    st.session_state.history = []

MAX_DISPLAY_SIZE = 800  # Adjust as needed

def count_by_class(result):
    try:
        names = result.names
        if result.boxes is None or len(result.boxes) == 0:
            return {}
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        counts = defaultdict(int)
        for cid in cls_ids:
            counts[names[int(cid)]] += 1
        return dict(counts)
    except Exception as e:
        st.error(f"Error counting classes: {str(e)}")
        return {}

def plot_result_image(res):
    try:
        bgr = res.plot()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb
    except Exception as e:
        st.error(f"Error plotting results: {str(e)}")
        return None

def cleanup_temp_file(file_path):
    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except Exception as e:
            st.warning(f"Could not remove temp file: {str(e)}")

# Tabs: Image, Video, Dashboard (webcam removed)
tab_img, tab_vid, tab_dash = st.tabs(["Image", "Video", "Dashboard"])

with tab_img:
    col_left, col_right = st.columns([1, 1])
    with col_left:
        up_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    with col_right:
        st.markdown("Adjust thresholds using the sidebar.")

    if up_img:
        try:
            image = Image.open(up_img).convert("RGB")
            width, height = image.size

            try:
                RESAMPLE = Image.Resampling.LANCZOS
            except AttributeError:
                RESAMPLE = Image.LANCZOS

            # Resize image if it's larger than the max display size
            if width > MAX_DISPLAY_SIZE or height > MAX_DISPLAY_SIZE:
                image_resized = image.copy()
                image_resized.thumbnail((MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE), RESAMPLE)
            else:
                image_resized = image

            # Run detection on resized image
            with st.spinner("Running detection..."):
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                    image_resized.save(tmp.name)
                    t0 = time.time()
                    try:
                        pred = model.predict(
                            source=tmp.name,
                            imgsz=imgsz,
                            conf=conf,
                            iou=iou,
                            agnostic_nms=agnostic_nms,
                            max_det=max_det,
                            verbose=False,
                        )[0]
                        elapsed = time.time() - t0
                    except Exception as e:
                        st.error(f"Detection failed: {str(e)}")
                        st.stop()

                out_img = plot_result_image(pred)

            # Show only original and detection result side by side
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption=f"Original ({width}x{height})", use_container_width=True)
            with col2:
                st.image(out_img, caption="Detection Result", use_container_width=True)

            counts = count_by_class(pred)
            if counts:
                df = pd.DataFrame({
                    "class": list(counts.keys()),
                    "count": list(counts.values())
                }).sort_values("count", ascending=False)
                st.subheader("Per-class counts")
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("class"))
            else:
                st.info("No defects detected.")
            st.caption(f"Time: {elapsed:.3f}s  |  FPS: {1 / max(elapsed, 1e-6):.1f}")
            st.session_state.history.append({
                "source": f"image:{up_img.name}",
                "counts": counts,
                "n_dets": sum(counts.values()) if counts else 0,
                "secs": elapsed,
                "fps": 1 / max(elapsed, 1e-6),
                "when": pd.Timestamp.now()
            })
        except Exception as e:
            st.error(f"Failed to process image: {str(e)}")

with tab_vid:
    st.write("Upload a video for frame-by-frame defect detection. You can download the annotated result after processing.")
    up_vid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv", "webm"])

    if up_vid:
        t_in_path = None
        t_out_path = None
        cap = None
        writer = None
        try:
            file_size = len(up_vid.getvalue()) / (1024 * 1024)
            if file_size > 100:
                st.warning(f"Large video file ({file_size:.1f}MB). Processing may take a long time.")
                if not st.checkbox("Proceed with large file"):
                    st.stop()

            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(up_vid.name)[-1], delete=False) as t_in:
                t_in.write(up_vid.getvalue())
                t_in_path = t_in.name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            cap = cv2.VideoCapture(t_in_path)
            if not cap.isOpened():
                st.error("Could not open the uploaded video. Check file format.")
                st.stop()

            fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

            if w <= 0 or h <= 0:
                st.error("Invalid video dimensions.")
                st.stop()

            st.info(f"Video info: {w}x{h} @ {fps_in:.1f} FPS, {total_frames or 'unknown'} frames")

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t_out:
                t_out_path = t_out.name

            writer = cv2.VideoWriter(t_out_path, fourcc, fps_in, (w, h))
            if not writer.isOpened():
                st.error("Could not initialise video writer.")
                st.stop()

            frame_counts = defaultdict(int)
            t0 = time.time()
            frame_num = 0
            failed_frames = 0
            progress_bar = st.progress(0, text="Starting video processing...")
            status_text = st.empty()

            with st.spinner("Running detection on video..."):
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frame_num += 1
                        try:
                            res = model.predict(
                                source=frame,
                                imgsz=imgsz,
                                conf=conf,
                                iou=iou,
                                agnostic_nms=agnostic_nms,
                                max_det=max_det,
                                verbose=False
                            )[0]
                            drawn = res.plot()
                            cts = count_by_class(res)
                            for k, v in cts.items():
                                frame_counts[k] += v
                        except Exception as e:
                            st.warning(f"Detection failed for frame {frame_num}: {str(e)}")
                            drawn = frame
                            failed_frames += 1

                        writer.write(drawn)
                        if total_frames:
                            progress = min(frame_num / total_frames, 1.0)
                            progress_bar.progress(progress, text=f"Processing frame {frame_num}/{total_frames}")
                        else:
                            status_text.text(f"Processed {frame_num} frames...")

                        if frame_num > 100000:
                            st.warning("Stopping processing - video too long.")
                            break
                except Exception as e:
                    st.error(f"Video processing failed: {str(e)}")
                finally:
                    if cap:
                        cap.release()
                    if writer:
                        writer.release()

            elapsed = time.time() - t0
            if failed_frames > 0:
                st.warning(f"{failed_frames} frames failed to process.")
            st.success("Video processing complete.")

            if os.path.exists(t_out_path) and os.path.getsize(t_out_path) > 0:
                st.video(t_out_path)
                with open(t_out_path, "rb") as f:
                    st.download_button(
                        "Download annotated video",
                        data=f.read(),
                        file_name=f"annotated_{os.path.basename(up_vid.name)}",
                        mime="video/mp4"
                    )

                if frame_counts:
                    dfv = pd.DataFrame({
                        "class": list(frame_counts.keys()),
                        "count": list(frame_counts.values())
                    }).sort_values("count", ascending=False)
                    st.subheader("Total detections (all frames)")
                    st.dataframe(dfv, use_container_width=True)
                    st.bar_chart(dfv.set_index("class"))
                else:
                    st.info("No defects detected in the video.")
                fps_eff = (frame_num / elapsed) if elapsed > 0 else 0
                st.caption(f"Frames: {frame_num}  |  Time: {elapsed:.1f}s  |  Throughput: {fps_eff:.1f} FPS")
                st.session_state.history.append({
                    "source": f"video:{up_vid.name}",
                    "counts": dict(frame_counts),
                    "n_dets": int(sum(frame_counts.values())),
                    "secs": elapsed,
                    "fps": fps_eff,
                    "when": pd.Timestamp.now()
                })
            else:
                st.error("Failed to create output video.")
        except Exception as e:
            st.error(f"Video processing failed: {str(e)}")
        finally:
            if cap:
                cap.release()
            if writer:
                writer.release()
            cleanup_temp_file(t_in_path)

with tab_dash:
    st.subheader("Session Summary")
    if len(st.session_state.history) == 0:
        st.info("No runs in this session yet. Use Image or Video tabs first.")
    else:
        if st.button("Clear History", type="secondary"):
            st.session_state.history = []
            st.rerun()
        rows = []
        for r in st.session_state.history:
            row = {
                "source": r["source"],
                "when": r["when"].strftime("%H:%M:%S"),
                "detections_total": r["n_dets"],
                "secs": round(r["secs"], 3),
                "fps": round(r["fps"], 2),
            }
            for k, v in r["counts"].items():
                row[f"class:{k}"] = v
            rows.append(row)
        dfh = pd.DataFrame(rows).sort_values("when", ascending=False)
        st.dataframe(dfh, use_container_width=True)
        all_counts = defaultdict(int)
        for r in st.session_state.history:
            for k, v in r["counts"].items():
                all_counts[k] += v
        if all_counts:
            agg = pd.DataFrame({
                "class": list(all_counts.keys()),
                "count": list(all_counts.values())
            }).sort_values("count", ascending=False)
            st.subheader("Total counts across session")
            st.dataframe(agg, use_container_width=True)
            st.bar_chart(agg.set_index("class"))
        total_runs = len(st.session_state.history)
        total_dets = int(sum([r["n_dets"] for r in st.session_state.history]))
        avg_fps = np.mean([r["fps"] for r in st.session_state.history if r["fps"] > 0]) if total_runs else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Runs this session", total_runs)
        c2.metric("Detections total", total_dets)
        c3.metric("Average FPS", f"{avg_fps:.1f}")
        if st.button("Export History as CSV"):
            csv_data = pd.DataFrame(st.session_state.history).to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_data,
                file_name=f"defect_detection_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

st.markdown("---")
st.caption("Fabric Defect Detector powered by YOLOv8s - Built with Streamlit")
