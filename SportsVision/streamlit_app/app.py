"""
Streamlit app for SportsVision action spotting.

Three-pane layout:
- Left: match list (scrollable)
- Middle: video player with predict button
- Right: predicted actions sorted chronologically (scrollable)

Inference runs only when the user clicks "Predict actions" and uses GPU 7 if available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure project src is importable
import sys

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation import apply_nms  # noqa: E402
from config import Config, LABEL_NAMES, NUM_CLASSES  # noqa: E402
from models import ActionSpottingModel  # noqa: E402

# Paths and constants
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/home/o_a38510/ML/Dataset")
FEATURE_ROOT = REPO_ROOT / "data" / "features" / "baidu_2.0"
SPLITS_FILE = REPO_ROOT / "data" / "splits_baidu_2.0.csv"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "best_model.pth"
CONFIG_PATH = REPO_ROOT / "configs" / "confidence_baidu.yaml"
TARGET_GPU = 7


# Terminal loader for video list
def _log_loader(msg: str) -> None:
    print(f"[loader] {msg}", flush=True)


@st.cache_resource(show_spinner=False)
def load_config() -> Config:
    cfg = Config.from_yaml(str(CONFIG_PATH))
    return cfg


def _video_paths(video_name: str) -> Dict[int, Path]:
    base = DATASET_ROOT / video_name
    paths: Dict[int, Path] = {}
    for half in (1, 2):
        candidate = base / f"{half}_224p.mkv"
        if candidate.exists():
            paths[half] = candidate
    return paths


@st.cache_resource(show_spinner=False)
def load_video_index() -> List[Dict[str, object]]:
    _log_loader(f"Loading video list from {SPLITS_FILE} ...")
    df = pd.read_csv(SPLITS_FILE)
    videos: List[Dict[str, object]] = []
    for video_name in sorted(df["video_name"].unique()):
        paths = _video_paths(video_name)
        if not paths:
            continue
        league, season, match = video_name.split("/", maxsplit=2)
        videos.append(
            {
                "id": video_name,
                "label": match,
                "league": league,
                "season": season,
                "halves": sorted(paths.keys()),
            }
        )
    _log_loader(f"Ready. {len(videos)} videos with available MKVs found on Atlas.")
    return videos


@st.cache_resource(show_spinner=False)
def load_model(cfg: Config) -> Tuple[ActionSpottingModel, torch.device]:
    if torch.cuda.is_available():
        if torch.cuda.device_count() <= TARGET_GPU:
            raise RuntimeError(f"GPU {TARGET_GPU} not visible. Found {torch.cuda.device_count()} GPUs.")
        torch.cuda.set_device(TARGET_GPU)
        device = torch.device(f"cuda:{TARGET_GPU}")
    else:
        device = torch.device("cpu")

    cfg.device = str(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    model = ActionSpottingModel(cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def _load_features(video_name: str, half: int, cfg: Config) -> Optional[np.ndarray]:
    feat_path = FEATURE_ROOT / video_name / f"{half}_{cfg.data.feature_name}.npy"
    if not feat_path.exists():
        return None
    return np.load(feat_path, mmap_mode="r")


def run_inference(video_name: str, cfg: Config, model: ActionSpottingModel, device: torch.device) -> List[Dict[str, object]]:
    frame_rate = cfg.data.frame_rate
    chunk_size = int(cfg.data.chunk_duration * frame_rate)
    nms_window = int(cfg.eval.nms_window * frame_rate)
    conf_thresh = cfg.eval.confidence_threshold

    detections: List[Dict[str, object]] = []

    for half in (1, 2):
        features = _load_features(video_name, half, cfg)
        if features is None:
            continue

        num_frames = features.shape[0]
        scores = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)

        for start in range(0, num_frames, chunk_size):
            end = min(start + chunk_size, num_frames)
            chunk = features[start:end]
            actual_len = chunk.shape[0]

            if actual_len < chunk_size:
                pad = np.zeros((chunk_size - actual_len, chunk.shape[1]), dtype=chunk.dtype)
                chunk = np.concatenate([chunk, pad], axis=0)

            tensor = torch.from_numpy(chunk).unsqueeze(0).float().to(device)
            with torch.no_grad():
                outputs = model(tensor)
                conf = outputs["confidence"].sigmoid().cpu().numpy()[0, :actual_len]
            scores[start:end] = conf

        if nms_window > 0:
            scores = apply_nms(scores, nms_window, cfg.eval.nms_type)

        for frame_idx in range(scores.shape[0]):
            for class_idx, score in enumerate(scores[frame_idx]):
                if score < conf_thresh:
                    continue
                time_seconds = frame_idx / frame_rate
                absolute_seconds = (half - 1) * 45 * 60 + time_seconds
                detections.append(
                    {
                        "half": half,
                        "frame": int(frame_idx),
                        "class_index": int(class_idx),
                        "label": LABEL_NAMES[class_idx],
                        "score": float(score),
                        "time_seconds": float(time_seconds),
                        "absolute_seconds": float(absolute_seconds),
                        "display_time": _format_time(time_seconds, half),
                    }
                )

    detections = sorted(detections, key=lambda d: d["absolute_seconds"])
    return detections


def _format_time(seconds: float, half: int) -> str:
    total_seconds = int(round(seconds))
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"H{half} {minutes:02d}:{secs:02d}"


def init_state():
    st.session_state.setdefault("selected_video", None)
    st.session_state.setdefault("selected_half", 1)
    st.session_state.setdefault("jump_time", 0.0)
    st.session_state.setdefault("detections", [])
    st.session_state.setdefault("prediction_cache", {})


def select_video(video: dict):
    """Callback to select a video - triggers full rerun to update player."""
    st.session_state.selected_video = video
    st.session_state.selected_half = video["halves"][0]
    st.session_state.jump_time = 0.0
    st.session_state.detections = []


def video_list_content(videos: List[Dict], query: str):
    """Video list content - not a fragment so selecting updates the player."""
    filtered = [
        v
        for v in videos
        if query in v["label"].lower()
        or query in v["league"].lower()
        or query in v["season"].lower()
    ]
    with st.container(height=500):
        if not filtered:
            st.info("No videos found.")
        else:
            for v in filtered:
                label = f"{v['label']}  ·  {v['league']} · {v['season']}"
                st.button(label, key=f"vid-{v['id']}", use_container_width=True,
                          on_click=select_video, args=(v,))


@st.fragment
def actions_list_fragment():
    """Fragment for actions list - updates without rerunning whole page."""
    dets = st.session_state.detections
    with st.container(height=500):
        if not dets:
            st.info("Run a prediction to see actions.")
        else:
            for i, det in enumerate(dets):
                st.markdown(f"**{det['label']}** — {det['display_time']} — {det['score']:.2f}")


def video_player_content(cfg: Config):
    """Video player content."""
    vid = st.session_state.selected_video
    if not vid:
        st.info("Pick a match on the left to load the stream.")
        return

    halves = vid["halves"]
    current_half = st.session_state.selected_half
    if current_half not in halves:
        current_half = halves[0]
        st.session_state.selected_half = current_half

    half_choice = st.radio("Half", halves, index=halves.index(current_half),
                           horizontal=True, key="half_radio")
    if half_choice != st.session_state.selected_half:
        st.session_state.selected_half = half_choice
        st.session_state.jump_time = 0.0

    video_path = _video_paths(vid["id"]).get(st.session_state.selected_half)
    if not video_path:
        st.error("Video file not found for this half.")
    else:
        st.markdown(
            f"**{vid['label']}** — {vid['league']} · {vid['season']} (Half {st.session_state.selected_half})",
        )
        st.video(str(video_path), start_time=int(st.session_state.jump_time))

    # Predict button
    if st.button("Predict actions", key="predict-btn", type="primary", help="Runs inference on GPU 7"):
        with st.spinner("Running inference..."):
            if vid["id"] in st.session_state.prediction_cache:
                detections = st.session_state.prediction_cache[vid["id"]]
            else:
                model, device = load_model(cfg)
                detections = run_inference(vid["id"], cfg, model, device)
                st.session_state.prediction_cache[vid["id"]] = detections
            st.session_state.detections = detections
            if not detections:
                st.warning("No detections found or features missing for this game.")
            else:
                st.success(f"Found {len(detections)} actions.")
                st.rerun()  # Rerun to update actions list


def main():
    st.set_page_config(page_title="SportsVision Spotting Viewer", layout="wide")
    init_state()

    cfg = load_config()
    videos = load_video_index()

    st.markdown(
        """
        <style>
        /* Disable page-level scrolling */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
        .main, .stApp, section.main {
            overflow: hidden !important;
            height: 100vh !important;
            max-height: 100vh !important;
        }
        .main .block-container {
            padding-top: 10px;
            padding-bottom: 0;
            max-height: 100vh;
            overflow: hidden !important;
        }
        /* Style scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #a1a1a1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_list, col_player, col_actions = st.columns([0.9, 1.6, 1.0])

    # Left pane: video list
    with col_list:
        st.subheader("Videos")
        query = st.text_input("Search by team or league", "").strip().lower()
        video_list_content(videos, query)
        st.caption("Select a match to load its video.")

    # Middle pane: video player and predict
    with col_player:
        st.subheader("Player")
        video_player_content(cfg)

    # Right pane: actions
    with col_actions:
        st.subheader("Predicted actions")
        actions_list_fragment()

    # Footer info
    with st.expander("Run info", expanded=False):
        st.markdown(
            f"""
            - Model: `best_model.pth` using config `{CONFIG_PATH.name}`
            - Features: `{cfg.data.feature_name}` from `{FEATURE_ROOT}`
            - Inference device: GPU {TARGET_GPU} if available, else CPU
            - Frame rate: {cfg.data.frame_rate} fps
            """
        )


if __name__ == "__main__":
    main()
